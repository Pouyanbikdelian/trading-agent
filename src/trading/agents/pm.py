"""Agent PM — the committee's trading arm. SIMULATED ONLY, by construction.

Weekly cycle: read the week's committee rulings + the scorecard-graded
calibration + a fresh context snapshot, make ONE LLM call, get target
weights over a fixed ETF universe, clamp them through hard risk caps,
and mark a virtual portfolio to market under ``state/agent_pm/``.

Why this shape (concept: reuse, don't re-convene):

* The daily committee already debates and journals; re-running it for the
  PM would duplicate spend and lose the week's trajectory. The PM instead
  sees posture ACROSS the week — drift is signal a snapshot can't carry.
* Calibration goes in the prompt so track record, not eloquence, weighs
  the voices — same principle as the Manager charter.
* The instrument universe is a fixed whitelist. An LLM that hallucinates
  a ticker gets that weight dropped to cash, never a creative fill.

This module never imports the broker, never constructs an ``Order`` and
never touches the runner's order path — the same isolation contract as
the committee itself. Risk caps here mirror the spirit of rule #4
(hard-blocking, not advisory) even though only virtual money moves.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger
from trading.memory.store import MemoryStore

LlmFn = Callable[[str, str], dict[str, Any]]

# Match the paper strategy's capital so the dashboard race compares like
# for like. Override with AGENT_PM_START_EQUITY; takes effect on a fresh
# book (delete state/agent_pm/ to restart the sim).
START_EQUITY = float(os.getenv("AGENT_PM_START_EQUITY", "1000000"))
COST_BPS = 10.0  # commission + slippage on turnover, charitable but not free
MAX_WEIGHT_PER_NAME = 0.25  # ETFs are diversified; a quarter is the ceiling
MAX_WEIGHT_PER_STOCK = 0.10  # single names carry idiosyncratic risk — tighter
MAX_GROSS = 1.0  # long-only, no leverage
MAX_CLUSTER = 0.50  # max combined weight in one correlated cluster

# Correlated clusters for the concentration cap. Deliberately coarse and
# beta-based, not GICS-pure: AMZN/TSLA trade with tech beta, so for RISK
# purposes they live in the tech complex. Unknown symbols are uncapped at
# cluster level (still capped per-name) — the map covers the system's
# revealed bias (semis/mega-tech) plus the liquid ETF shelf.
CLUSTERS: dict[str, frozenset[str]] = {
    "tech_complex": frozenset(
        {
            "XLK",
            "SMH",
            "QQQ",
            "XLC",
            "NVDA",
            "AMD",
            "AVGO",
            "MU",
            "INTC",
            "WDC",
            "STX",
            "SNDK",
            "LITE",
            "CIEN",
            "GLW",
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
            "TSLA",
            "QCOM",
            "AMAT",
            "LRCX",
            "KLAC",
            "MRVL",
            "TXN",
            "ADI",
            "ASML",
            "SMCI",
            "DELL",
            "ANET",
            "CRM",
            "ORCL",
            "NOW",
            "TSM",
            "ARM",
            "PLTR",
        }
    ),
    "energy": frozenset({"XLE", "XOM", "CVX", "URA", "COP", "SLB", "OXY"}),
    "health": frozenset({"XLV", "IBB", "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV"}),
    "financials": frozenset({"XLF", "JPM", "V", "MA", "BAC", "GS", "MS", "WFC"}),
    "defense": frozenset({"ITA", "LMT", "RTX", "NOC", "GD", "BA"}),
}


def _cluster_of(sym: str) -> str | None:
    for name, members in CLUSTERS.items():
        if sym in members:
            return name
    return None


# Fixed, liquid ETF whitelist: broad + sectors + themes + defense assets.
UNIVERSE: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
    "SMH",
    "IBB",
    "ITA",
    "URA",
    "TLT",
    "IEF",
    "GLD",
)

PM_CHARTER = (
    "You are the Portfolio Manager of a small SIMULATED sleeve. You have "
    "just read a week of committee debates, each agent's graded track "
    "record, today's market context, and the sentinel's recent alert status. "
    "Decide the sleeve's allocation for the coming week.\n"
    "\n"
    "HARD RULES: long-only; weights sum to at most 1.0 (remainder is cash); "
    "only tickers from the allowed universes — max 0.25 per ETF, "
    "max 0.10 per single stock, max 0.50 combined in any one correlated "
    "cluster (semis + mega-cap tech = ONE cluster; excess cut to cash by code).\n"
    "\n"
    "ANTI-INERTIA (mandatory — state this explicitly in your rationale): "
    "The current book is NOT your starting point. Ask: starting from scratch "
    "today, would I build this portfolio? Holdings the committee is bearish "
    "on need an affirmative reason to keep — past gains are not a reason. "
    "A 5-of-7 or greater bearish committee consensus on a held cluster is a "
    "cluster EXIT signal: cut that cluster by at least half, not just trim "
    "one or two names. Do not rotate freed weight into a different sector "
    "just to stay invested — cash is a valid position.\n"
    "\n"
    "SENTINEL RULE (mandatory when sentinel_alert is present): If the "
    "sentinel fired CAUTION or ALARM in the last 24 hours, (a) cap total "
    "deployment at 70% — the 30% minimum goes to cash, not rotation; "
    "(b) do not open new sector positions to replace trimmed ones this cycle; "
    "freed weight stays in cash first.\n"
    "\n"
    "CREATIVE SCOUT RULE: If the creative or scout agent shows a bullish "
    "take with confidence ≥ 0.70 on a stock or sector NOT currently in the "
    "top three holdings by weight, allocate at least 5% there — this is "
    "the forcing function for portfolio evolution beyond existing themes.\n"
    "\n"
    "OPERATOR HOLDS (mandatory): symbols listed in operator_held_do_not_trade "
    "are the operator's long-term positions, pinned outside your mandate. "
    "NEVER allocate to them — any weight you place there is cut to cash by "
    "code. Do not count them as portfolio exposure either; they are not "
    "your book.\n"
    "\n"
    "STOCK PREFERENCE (mandatory): Individual stocks are the PRIMARY vehicle. "
    "When a sector thesis is clear, own the best 1-3 individual names in "
    "that sector — NOT the sector ETF. Examples: own LMT or RTX instead of "
    "ITA; own JPM or V instead of XLF; own LLY or UNH instead of XLV; own "
    "AMAT or ASML instead of SMH. Sector ETFs are last resort: use them "
    "only when the thesis is too diffuse to pick a winner, OR as a "
    "defensive hedge. Hard cap: maximum 3 ETF positions in the book at any "
    "time — if you want to express more than 3 sector views, express the "
    "rest via individual stocks.\n"
    "\n"
    "Trust calibration over stated confidence — an agent wrong all month is "
    "a fade. Social-signal sources with high trust scores (flagged by the "
    "creative agent's 'cited_lessons' or 'sources' fields) add weight to a "
    "thesis even when the committee majority is quiet on it. "
    "Respond ONLY with JSON: "
    '{"target_weights": {"<TICKER>": <0.0-0.25>, ...}, '
    '"rationale": "<5-7 sentences: the trade, the anti-inertia check result, '
    'stock-vs-ETF decisions, and whether the sentinel rule applied>", '
    '"watch": "<what would change your mind this week>"}'
)


def _stock_universe() -> tuple[str, ...]:
    """Single stocks the PM may hold — the union of '+'-joined universe
    names from AGENT_PM_STOCK_UNIVERSE (set empty to disable: ETF-only).

    Default covers S&P 500 + NASDAQ-100 + Russell 1000 via the generated
    constituents file (scripts/refresh_universes.py), with the hand-
    curated us_large_cap as a floor so the PM still has stocks when the
    generated file hasn't been refreshed yet. Missing universes degrade
    with a warning, never break the run. The whitelist remains the
    anti-hallucination guard: an off-list ticker is dropped to cash,
    never bought."""
    spec = os.getenv("AGENT_PM_STOCK_UNIVERSE", "sp500+nasdaq100+russell1000+us_large_cap")
    if not spec:
        return ()
    from trading.core.universes import load_universe

    syms: set[str] = set()
    for name in spec.split("+"):
        try:
            syms |= {i.symbol.upper() for i in load_universe(name.strip())}
        except Exception as e:
            logger.bind(component="agent_pm").warning(f"stock universe '{name}': {e}")
    return tuple(sorted(syms))


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    """The PM turns the committee's debate into actual sim-book allocations —
    a decision node, so it runs on the frontier model (``tier='frontier'``).
    An injected llm (tests) overrides this and keeps runs hermetic."""
    from trading.agents.llm import complete_json

    return complete_json(system, prompt, tier="frontier")


def _clamp_weights(
    raw: Any, stocks: tuple[str, ...] = (), blocked: frozenset[str] = frozenset()
) -> dict[str, float]:
    """Hard caps: whitelist, long-only, per-name max (tighter for single
    stocks than for ETFs), gross max. ``blocked`` symbols (operator
    ``/hold`` pins — Yan's long-term positions) are dropped to cash like
    any off-whitelist name: the PM may never allocate to a symbol the
    operator has pinned, in sim today and through the bridge later."""
    weights: dict[str, float] = {}
    if not isinstance(raw, dict):
        return weights
    for sym, w in raw.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        sym = str(sym).upper().strip()
        if w <= 0 or sym in blocked:
            continue
        if sym in UNIVERSE:
            weights[sym] = min(w, MAX_WEIGHT_PER_NAME)
        elif sym in stocks:
            weights[sym] = min(w, MAX_WEIGHT_PER_STOCK)
    # Sector/cluster concentration cap: six 0.9-correlated names at 8-10%
    # each is one big bet wearing six hats. Scale offending clusters down
    # proportionally; freed weight stays in cash (never redistributed —
    # the PM wanted concentration, it doesn't get diversification for free).
    by_cluster: dict[str, float] = {}
    for s, w in weights.items():
        c = _cluster_of(s)
        if c:
            by_cluster[c] = by_cluster.get(c, 0.0) + w
    for c, total in by_cluster.items():
        if total > MAX_CLUSTER:
            f = MAX_CLUSTER / total
            for s in list(weights):
                if _cluster_of(s) == c:
                    weights[s] = round(weights[s] * f, 4)
    gross = sum(weights.values())
    if gross > MAX_GROSS:
        weights = {s: w * MAX_GROSS / gross for s, w in weights.items()}
    return weights


def _fetch_closes(symbols: list[str]) -> dict[str, float]:
    """Latest closes via yfinance; missing symbols simply absent."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    try:
        import yfinance as yf

        raw = yf.download(
            " ".join(symbols),
            period="5d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        for sym in symbols:
            try:
                out[sym] = float(raw[sym]["Close"].dropna().iloc[-1])
            except Exception:
                continue
    except Exception as e:
        logger.bind(component="agent_pm").warning(f"price fetch failed: {e}")
    return out


def _load_portfolio(pm_dir: Path) -> dict[str, Any]:
    path = pm_dir / "portfolio.json"
    try:
        book = json.loads(path.read_text())
        book.setdefault("start_equity", START_EQUITY)
        return book
    except Exception:
        return {"cash": START_EQUITY, "holdings": {}, "history": [], "start_equity": START_EQUITY}


def _save(pm_dir: Path, name: str, payload: dict[str, Any]) -> None:
    pm_dir.mkdir(parents=True, exist_ok=True)
    path = pm_dir / name
    fd, tmp = tempfile.mkstemp(dir=pm_dir, prefix=f"{name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, default=str, indent=1)
    os.replace(tmp, path)


def mark_to_market(state_dir: Path, *, prices: dict[str, float] | None = None) -> dict[str, Any]:
    """Daily mark — no LLM, no trades. Appends an equity point (plus the
    SPY close as benchmark) so the observation period has a real curve,
    not four Monday dots. Idempotent per day."""
    pm_dir = Path(state_dir) / "agent_pm"
    if not (pm_dir / "portfolio.json").exists():
        return {"ok": False, "reason": "no PM book yet"}
    book = _load_portfolio(pm_dir)
    symbols = sorted({*book["holdings"], "SPY"})
    px = prices if prices is not None else _fetch_closes(symbols)
    unpriced = [s for s in book["holdings"] if s not in px]
    if unpriced or "SPY" not in px:
        return {"ok": False, "reason": f"missing prices: {unpriced or ['SPY']}"}
    equity = float(book["cash"]) + sum(q * px[s] for s, q in book["holdings"].items())
    entry = {
        "t": datetime.now(tz=timezone.utc).isoformat(),
        "equity": round(equity, 2),
        "spy": round(px["SPY"], 2),
    }
    today = entry["t"][:10]
    history = [h for h in book.get("history", []) if str(h.get("t", ""))[:10] != today]
    book["history"] = [*history, entry][-520:]
    _save(pm_dir, "portfolio.json", book)
    return {"ok": True, **entry}


def performance(state_dir: Path) -> dict[str, Any]:
    """Since-inception stats from the marked history: PM return, SPY
    return over the same window, max drawdown. Thin history -> thin dict."""
    book = _load_portfolio(Path(state_dir) / "agent_pm")
    hist = book.get("history", [])
    out: dict[str, Any] = {"points": len(hist)}
    if not hist:
        return out
    eq = [float(h["equity"]) for h in hist]
    out["equity"] = eq[-1]
    # Base on the equity the book opened with (persisted at creation):
    # eq[0] is unreliable because same-day marks replace the inception
    # entry, and the constant changes across config edits.
    base = float(book.get("start_equity") or eq[0])
    out["start_equity"] = base
    out["return_pct"] = (eq[-1] / base - 1.0) * 100
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    out["max_drawdown_pct"] = mdd * 100
    spy = [float(h["spy"]) for h in hist if h.get("spy")]
    if len(spy) >= 2:
        out["spy_return_pct"] = (spy[-1] / spy[0] - 1.0) * 100
    return out


def run_agent_pm(
    context: dict[str, Any],
    mem: MemoryStore,
    state_dir: Path,
    *,
    llm: LlmFn | None = None,
    prices: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One weekly PM cycle. ``prices`` injectable for hermetic tests."""
    llm = llm or _default_llm
    pm_dir = Path(state_dir) / "agent_pm"
    book = _load_portfolio(pm_dir)

    # --- mark current book to market. Price only what we hold (plus the
    # ETF shelf) — with a 1000-name stock whitelist, pricing everything
    # up front would be a pointless bulk download; targets the PM
    # actually picks are priced after the call.
    stocks = _stock_universe()
    symbols = sorted(set(book["holdings"]) | set(UNIVERSE))
    px = dict(prices) if prices is not None else _fetch_closes(symbols)
    equity = float(book["cash"]) + sum(
        qty * px[s] for s, qty in book["holdings"].items() if s in px
    )
    unpriced = [s for s in book["holdings"] if s not in px]
    if unpriced:
        logger.bind(component="agent_pm").warning(f"no price for held {unpriced}; skipping run")
        return {"ok": False, "reason": f"missing prices for held positions: {unpriced}"}

    # --- assemble the PM's evidence: week of rulings + calibration + context
    week = mem.journal_tail(6, kind="committee")

    # Sentinel state: the PM must know if a tripwire fired recently so the
    # sentinel rule in the charter can be applied explicitly.
    sentinel_state: dict[str, Any] = {}
    try:
        sentinel_path = Path(state_dir) / "sentinel.json"
        if sentinel_path.exists():
            raw = json.loads(sentinel_path.read_text())
            sentinel_state = {
                "last_alert_ts": raw.get("last_alert_ts"),
                "triggers": raw.get("triggers", []),
            }
    except Exception:
        pass

    # Operator pins: one list for the whole system (state/holds.json).
    # The PM is told about them AND hard-blocked from them — belt and
    # braces, same as every other cap in this module.
    from trading.runner.holds import load_holds

    held = frozenset(load_holds(Path(state_dir)))

    prompt = json.dumps(
        {
            "operator_held_do_not_trade": sorted(held),
            "sim_portfolio": {
                "equity": round(equity, 2),
                "cash": round(float(book["cash"]), 2),
                "holdings": book["holdings"],
                # Its own track record — a PM that can't see its P&L vs
                # benchmark can't learn from it.
                "performance": performance(state_dir),
            },
            "allowed_universe_etfs_max_25pct": list(UNIVERSE),
            "allowed_universe_stocks_max_10pct": (
                list(stocks)
                if len(stocks) <= 80
                else (
                    f"{len(stocks)} names: any current S&P 500, NASDAQ-100 or "
                    "Russell 1000 constituent (membership validated after you "
                    "answer; off-index tickers are dropped)"
                )
            ),
            "week_of_committee_rulings": week,
            "agent_calibration": mem.calibration(),
            "sentinel_alert": sentinel_state or None,
            "today_context": context,
        },
        default=str,
    )[:24000]

    try:
        out = llm(PM_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="agent_pm").warning(f"PM call failed: {e}")
        return {"ok": False, "reason": f"PM call failed: {e}"}

    weights = _clamp_weights(out.get("target_weights"), stocks, blocked=held)
    # Late pricing for newly targeted names not already marked.
    need_px = [s for s in weights if s not in px]
    if need_px and prices is None:
        px.update(_fetch_closes(need_px))
    unpriced_targets = [s for s in weights if s not in px]
    for s in unpriced_targets:
        weights.pop(s)  # no mark, no trade — that weight stays in cash
    dropped = sorted(set(map(str, (out.get("target_weights") or {}))) - set(weights))

    # --- rebalance at last close, costs on turnover
    new_holdings: dict[str, float] = {}
    target_value = 0.0
    for sym, w in weights.items():
        if sym not in px:
            continue
        qty = round(w * equity / px[sym], 4)
        if qty > 0:
            new_holdings[sym] = qty
            target_value += qty * px[sym]
    turnover = sum(
        abs(new_holdings.get(s, 0.0) - book["holdings"].get(s, 0.0)) * px[s]
        for s in set(new_holdings) | set(book["holdings"])
        if s in px
    )
    costs = turnover * COST_BPS / 10_000.0
    book = {
        "cash": round(equity - target_value - costs, 2),
        "holdings": new_holdings,
        "start_equity": float(book.get("start_equity") or START_EQUITY),
        "history": [
            *book.get("history", []),
            {"t": datetime.now(tz=timezone.utc).isoformat(), "equity": round(equity, 2)},
        ][-520:],
    }
    _save(pm_dir, "portfolio.json", book)

    result = {
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "equity": round(equity, 2),
        "weights": weights,
        "dropped": dropped,
        "turnover": round(turnover, 2),
        "costs": round(costs, 2),
        "rationale": str(out.get("rationale", ""))[:800],
        "watch": str(out.get("watch", ""))[:300],
    }
    _save(pm_dir, "last_run.json", result)
    mem.journal("agent_pm", {k: result[k] for k in ("equity", "weights", "rationale")}, actor="pm")
    return result


def format_pm_digest(result: dict[str, Any]) -> str:
    from trading.agents.committee import _clip

    if not result.get("ok"):
        return f"🤖 Agent PM did not trade: {result.get('reason', 'unknown')}"
    lines = [
        f"🧪 *Agent PM (simulated)* — equity ${result['equity']:,.0f}",
        "",
        "*Target book:* "
        + (", ".join(f"{s} {w:.0%}" for s, w in sorted(result["weights"].items())) or "all cash"),
        f"*Why:* {_clip(result.get('rationale', ''), 500)}",
        f"_Watching: {_clip(result.get('watch', ''), 200)}_",
    ]
    if result.get("dropped"):
        lines.append(f"_(dropped off-universe/invalid: {', '.join(result['dropped'])})_")
    lines.append(f"_turnover ${result['turnover']:,.0f} · costs ${result['costs']:,.2f}_")
    return "\n".join(lines)
