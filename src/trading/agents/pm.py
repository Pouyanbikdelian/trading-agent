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
* But a whitelist is permission, not a candidate list. Until 2026-08-05
  the prompt named no ticker the sleeve did not already own — the stock
  whitelist arrived as the *string* "1,607 names: any S&P 500 constituent"
  — so the model re-picked the book it was shown and the sleeve held the
  same names for weeks at healthy turnover. The ranked candidate ladder
  in ``today_context`` (see ``agents/candidates.py``) is the fix; the
  charter now points at it as the only sanctioned source of new names.

This module never imports the broker, never constructs an ``Order`` and
never touches the runner's order path — the same isolation contract as
the committee itself. Risk caps here mirror the spirit of rule #4
(hard-blocking, not advisory) even though only virtual money moves.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.copilot.mandates import STRENGTH_GUIDANCE
from trading.core.logging import logger
from trading.core.text import clip
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
# The charter has asked for "maximum 3 ETF positions" since the stock-
# preference block was written; nothing enforced it, and a book of six
# sector ETFs satisfied every cap that was actually code. Prose the code
# does not back is a suggestion, and this system's whole design premise
# is that limits bind.
MAX_ETF_POSITIONS = 3
# Cycles holding an identical set of names before the digest says so out
# loud. Three weekly cycles is a month of the same book — long enough that
# it is a decision, short enough to still be worth questioning.
STALE_BOOK_CYCLES = 3

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
    "WHOSE BOOK (read this before anything else): YOUR book is "
    "sim_portfolio.holdings and nothing else. Any key ending "
    "_NOT_YOUR_BOOK describes the operator's live brokerage account — a "
    "different portfolio, managed by other machinery, which you neither "
    "trade nor get credit for de-risking. It is context for reading the "
    "market, never a position of yours. Exiting a cluster you do not hold "
    "is not a decision; if you describe cutting something, it must appear "
    "in sim_portfolio.holdings.\n"
    "\n"
    "ANTI-INERTIA (mandatory — state this explicitly in your rationale): "
    "Apply this to sim_portfolio.holdings, name by name. The current book "
    "is NOT your starting point. Ask: starting from scratch "
    "today, would I build this portfolio? Holdings the committee is bearish "
    "on need an affirmative reason to keep — past gains are not a reason. "
    "If a majority of the agent takes in agent_takes are bearish on a held "
    "cluster, that is a cluster EXIT signal: cut that cluster by at least "
    "half, not just trim one or two names. Do not rotate freed weight into a "
    "different sector just to stay invested — cash is a valid position. "
    "Your rationale MUST name at least one symbol from "
    "sim_portfolio.holdings that you re-underwrote and say what would have "
    "made you drop it, and MUST account for every name you opened or "
    "closed this cycle. A rationale that discusses only names you do not "
    "hold has not done the work.\n"
    "\n"
    "CANDIDATE LADDER (your idea source): today_context.candidate_ladder is "
    "the live strategy's own ranked scoreboard over the trading universe — "
    "the same ranking the system trades off. It is where new names come "
    "from. Do NOT pick tickers from memory or from examples in these "
    "instructions; work from the ladder, the agent takes, and the operator's "
    "mandates. A high pctile_52w says where price sits in its range, not "
    "that reward-to-risk is good — rank is not a buy signal by itself. If "
    "the ladder is absent this cycle, say so in your rationale and prefer "
    "cash over re-picking the existing book by default.\n"
    "\n"
    "SENTINEL RULE (mandatory when sentinel_alert is present): If the "
    "sentinel fired CAUTION or ALARM in the last 24 hours, (a) cap total "
    "deployment at 70% — the 30% minimum goes to cash, not rotation; "
    "(b) do not open new sector positions to replace trimmed ones this cycle; "
    "freed weight stays in cash first.\n"
    "\n"
    "CREATIVE SCOUT RULE: agent_takes carries each specialist's own take, "
    "keyed by agent, with its falsifiable prediction. If the creative or "
    "scout agent shows a bullish take with confidence ≥ 0.70 on a stock or "
    "sector NOT currently in the top three holdings by weight, allocate at "
    "least 5% there — this is the forcing function for portfolio evolution "
    "beyond existing themes. Weigh every take by that agent's row in "
    "agent_calibration, not by how confidently it is phrased.\n"
    "\n"
    "OPERATOR MANDATES: " + STRENGTH_GUIDANCE + "\n"
    "\n"
    "OPERATOR HOLDS (mandatory): symbols listed in operator_held_do_not_trade "
    "are the operator's long-term positions, pinned outside your mandate. "
    "NEVER allocate to them — any weight you place there is cut to cash by "
    "code. Do not count them as portfolio exposure either; they are not "
    "your book.\n"
    "\n"
    "STOCK PREFERENCE (mandatory): Individual stocks are the PRIMARY vehicle. "
    "When a sector thesis is clear, own the best 1-3 individual names in "
    "that sector rather than the sector ETF — and pick those names from the "
    "candidate ladder, which is why it is given to you. Sector ETFs are last "
    "resort: use them only when the thesis is too diffuse to pick a winner, "
    "OR as a defensive hedge. Hard cap: maximum 3 ETF positions in the book "
    "at any time (enforced in code — excess ETF weight is cut to cash); if "
    "you want to express more than 3 sector views, express the rest via "
    "individual stocks.\n"
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


# How much history the PM reads. Rulings are the manager's synthesis;
# takes are the specialists behind it. Two committee cycles of takes
# (8 agents, Mon + Fri) is the window in which a scout idea is still
# actionable — older than that and it is a lesson, not a trade.
RULINGS_SEEN = 6
TAKES_SEEN = 16

# Characters of serialized JSON the PM prompt may occupy.
PROMPT_BUDGET = 24_000


# Keys in the shared agent context that describe the OPERATOR'S LIVE
# BROKERAGE ACCOUNT, not the PM's simulated sleeve — renamed on the way
# into the PM prompt.
#
# Why (observed 2026-08-05, first cycle after the candidate ladder
# shipped): the PM's book was JPM/V/XLV/LMT/GLD/PM/IBB, and its rationale
# opened "the sim book — it is effectively 100% semi/storage (AMD, INTC,
# STX, WDC, SNDK, DELL, CIEN, LITE) ... so I cut the entire semi/storage
# sleeve to zero". It cut a cluster it did not hold. Those semis are the
# live account, arriving as ``today_context.positions`` — and a bare key
# called "positions", sitting next to "holdings", reads as "your
# positions". So the model spent its whole ANTI-INERTIA obligation
# re-underwriting a portfolio it does not manage, considered the rule
# discharged, and left its own book untouched. That is the fixation
# surviving the fix that was supposed to end it.
#
# The names are deliberately shouty. This is a prompt, not an API: the
# key IS the documentation, and the failure mode was ambiguity.
OPERATOR_ACCOUNT_KEYS: dict[str, str] = {
    "positions": "operator_live_account_positions_NOT_YOUR_BOOK",
    "account": "operator_live_account_equity_NOT_YOUR_BOOK",
    "book_concentration": "operator_live_account_concentration_NOT_YOUR_BOOK",
}


def _relabel_operator_account(context: dict[str, Any]) -> dict[str, Any]:
    """Rename the live-account keys so they cannot be read as the sleeve.

    Shallow copy: the committee shares this context object and must keep
    seeing the original key names, since for the committee the live
    account IS the book under discussion.
    """
    if not isinstance(context, dict):
        return context
    return {OPERATOR_ACCOUNT_KEYS.get(k, k): v for k, v in context.items()}


def _recent_takes(mem: MemoryStore) -> list[dict[str, Any]]:
    """The specialists' own takes, newest first, compacted for the prompt.

    Newest-first matters: ``_budgeted_prompt`` trims list tails, so an
    overflow costs the oldest take rather than today's.
    """
    out: list[dict[str, Any]] = []
    for row in mem.journal_tail(TAKES_SEEN, kind="take"):
        payload = row.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        take: dict[str, Any] = {
            "ts": str(row.get("ts"))[:16],
            "agent": payload.get("agent") or row.get("actor"),
            "stance": payload.get("stance"),
            # The full take is several paragraphs; the PM needs the claim,
            # not the prose around it.
            "take": clip(str(payload.get("take", "")), 400),
        }
        pred = payload.get("prediction")
        if isinstance(pred, dict):
            # The falsifiable part — subject/direction/confidence is what
            # the CREATIVE SCOUT RULE actually keys on.
            take["prediction"] = {
                k: pred.get(k) for k in ("subject", "direction", "horizon_days", "confidence")
            }
        if payload.get("sources"):
            take["sources"] = list(payload["sources"])[:4]
        out.append(take)
    return out


def _budgeted_prompt(payload: dict[str, Any], budget: int = PROMPT_BUDGET) -> str:
    """Serialize under ``budget`` by DROPPING whole items, never slicing.

    The previous implementation was ``json.dumps(payload)[:24000]`` with
    ``today_context`` as the final key — so the first casualty of an
    overflow was the market data the decision is actually about, the
    second was the candidate ladder inside it, and what reached the model
    was JSON cut off mid-string. Silent, and exactly backwards.

    Trim order is the inverse of decision value, and mirrors the rule
    already written into ``context.py``: gossip goes first, then history.
    The book, the caps, the mandates and today's context survive every
    reduction, so the result is always both valid JSON and sufficient to
    decide on.
    """
    p = copy.deepcopy(payload)

    def fitted() -> str | None:
        s = json.dumps(p, default=str)
        return s if len(s) <= budget else None

    s = fitted()
    if s is not None:
        return s

    # 1. Gossip. Headlines are the largest and least decision-relevant
    #    block in the context by a wide margin.
    ctx = p.get("today_context")
    if isinstance(ctx, dict) and ctx.get("headlines"):
        for keep in (16, 6, 0):
            if keep:
                ctx["headlines"] = list(ctx["headlines"])[:keep]
            else:
                ctx.pop("headlines", None)
            s = fitted()
            if s is not None:
                return s

    # 2. History, oldest first: rulings, then takes. Both lists are
    #    newest-first, so popping the tail drops the stalest evidence.
    for key in ("recent_committee_rulings", "agent_takes"):
        seq = p.get(key)
        while isinstance(seq, list) and len(seq) > 1:
            seq.pop()
            s = fitted()
            if s is not None:
                return s

    # 3. Terminal fallback: shed whole keys in reverse-value order until it
    #    fits. Guaranteed to terminate with valid JSON — unlike a slice.
    for key in ("recent_committee_rulings", "agent_takes", "agent_calibration", "dossiers"):
        p.pop(key, None)
        s = fitted()
        if s is not None:
            return s
    if isinstance(ctx, dict):
        for key in ("dossiers", "source_trust", "established_lessons", "economy"):
            ctx.pop(key, None)
            s = fitted()
            if s is not None:
                return s
    return json.dumps(p, default=str)


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
    # ETF-count cap: keep the heaviest MAX_ETF_POSITIONS ETF sleeves, drop
    # the rest to cash. Ties break on ticker so the outcome is deterministic
    # (a cap that depends on dict ordering is not a cap).
    etfs = sorted((s for s in weights if s in UNIVERSE), key=lambda s: (-weights[s], s))
    for sym in etfs[MAX_ETF_POSITIONS:]:
        weights.pop(sym)
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
    # Persist the per-symbol marks alongside the equity point. /pm runs in
    # the bot's command path, which must not make a network call — a
    # yfinance round-trip there would block the poll loop. Storing the
    # closes we already fetched lets the digest show value and weight per
    # holding (a bare share count is unreadable) and lets it say how stale
    # those numbers are.
    book["marks"] = {s: round(float(px[s]), 4) for s in book["holdings"] if s in px}
    book["marks_ts"] = entry["t"]
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

    # A held name that yfinance failed to return used to abort the whole
    # cycle, leaving the book untouched. That turned a routine, transient
    # data gap into a silent no-trade week — and a run of them into a book
    # that hadn't moved in a month for reasons no one was reading. Fall
    # back to the last stored mark instead: a day-old close is a far better
    # basis for a weekly rebalance than not rebalancing at all. Only a name
    # with no mark anywhere is genuinely unpriceable, and only that aborts.
    stale_marks: list[str] = []
    prior_marks = book.get("marks") or {}
    for sym in book["holdings"]:
        if sym not in px and sym in prior_marks:
            px[sym] = float(prior_marks[sym])
            stale_marks.append(sym)
    if stale_marks:
        logger.bind(component="agent_pm").warning(
            f"using stored marks for unfetched holdings: {stale_marks}"
        )
    unpriced = [s for s in book["holdings"] if s not in px]
    if unpriced:
        logger.bind(component="agent_pm").warning(f"no price for held {unpriced}; skipping run")
        return {"ok": False, "reason": f"missing prices for held positions: {unpriced}"}

    equity = float(book["cash"]) + sum(
        qty * px[s] for s, qty in book["holdings"].items() if s in px
    )

    # --- assemble the PM's evidence: recent rulings + the takes behind
    # them + calibration + context.
    #
    # Rulings alone were not enough. ``committee`` journal rows carry only
    # the manager's synthesis ({"ruling", "disagreement"}); the individual
    # specialist takes are journaled separately under kind ``take``. So the
    # PM's two anti-inertia forcing functions — "creative or scout bullish
    # at ≥0.70 ⇒ allocate 5%" and the bearish-majority cluster exit — were
    # written against evidence that never reached the prompt, and the
    # calibration table had no takes to weigh. Both are now real inputs.
    rulings = mem.journal_tail(RULINGS_SEEN, kind="committee")
    takes = _recent_takes(mem)

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

    # Standing instructions the operator left from Telegram. Passed as
    # context for the decision — the weights this produces still go
    # through _clamp_weights, so a mandate can never lift a cap.
    from trading.copilot.mandates import for_context as _mandates_for_context

    payload: dict[str, Any] = {
        "operator_mandates": _mandates_for_context(Path(state_dir)),
        "operator_held_do_not_trade": sorted(held),
        "sim_portfolio": {
            "equity": round(equity, 2),
            "cash": round(float(book["cash"]), 2),
            "holdings": book["holdings"],
            # Its own track record — a PM that can't see its P&L vs
            # benchmark can't learn from it.
            "performance": performance(state_dir),
            # How long this book has gone without a name changing. The
            # digest reports it too; the PM sees it so a long streak is a
            # fact it must answer for, not a habit no one names.
            "cycles_since_name_change": int(book.get("cycles_since_name_change", 0)),
        },
        "allowed_universe_etfs_max_25pct": list(UNIVERSE),
        "allowed_universe_stocks_max_10pct": (
            list(stocks)
            if len(stocks) <= 80
            else (
                f"{len(stocks)} names: any current S&P 500, NASDAQ-100 or "
                "Russell 1000 constituent (membership validated after you "
                "answer; off-index tickers are dropped). Do not pick from "
                "memory — work from today_context.candidate_ladder."
            )
        ),
        "agent_takes": takes,
        "agent_calibration": mem.calibration(),
        "sentinel_alert": sentinel_state or None,
        "today_context": _relabel_operator_account(context),
        "recent_committee_rulings": rulings,
    }
    prompt = _budgeted_prompt(payload)

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
    prior_holdings = dict(book["holdings"])
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # Staleness counter. A book can churn weight every week and still hold
    # the same names for a quarter — which is what happened, and turnover
    # alone never showed it because turnover was healthy the whole time.
    # Count CYCLES WITH NO NAME CHANGE, and make the number visible in the
    # digest and in the PM's own next prompt.
    names_changed = set(new_holdings) != set(prior_holdings)
    cycles_stale = 0 if names_changed else int(book.get("cycles_since_name_change", 0)) + 1
    if cycles_stale >= STALE_BOOK_CYCLES:
        logger.bind(component="agent_pm").warning(
            f"book has held the same names for {cycles_stale} cycles: {sorted(new_holdings)}"
        )

    book = {
        "cash": round(equity - target_value - costs, 2),
        "holdings": new_holdings,
        "start_equity": float(book.get("start_equity") or START_EQUITY),
        "cycles_since_name_change": cycles_stale,
        "history": [
            *book.get("history", []),
            {"t": now_iso, "equity": round(equity, 2)},
        ][-520:],
        # Marks from this rebalance — see mark_to_market for why /pm needs
        # them persisted rather than fetched.
        "marks": {s: round(float(px[s]), 4) for s in new_holdings if s in px},
        "marks_ts": now_iso,
    }
    _save(pm_dir, "portfolio.json", book)

    result = {
        "ok": True,
        "ts": now_iso,
        "equity": round(equity, 2),
        "weights": weights,
        "dropped": dropped,
        "turnover": round(turnover, 2),
        "costs": round(costs, 2),
        # clip, not a raw slice: this text goes straight to Telegram and a
        # mid-word cut through a Markdown pair can cost the whole message.
        "rationale": clip(out.get("rationale", ""), 900),
        "watch": clip(out.get("watch", ""), 400),
        # What actually changed, so the digest can lead with the delta
        # instead of restating the whole book.
        "opened": sorted(set(new_holdings) - set(prior_holdings)),
        "closed": sorted(set(prior_holdings) - set(new_holdings)),
        "prior_holdings": prior_holdings,
        "cycles_since_name_change": cycles_stale,
        # Holdings marked from a stored close rather than a fresh fetch —
        # disclosed rather than quietly folded into the equity number.
        "stale_marks": stale_marks,
        "candidate_ladder_seen": bool((context or {}).get("candidate_ladder")),
        # The ladder's own as-of date, so a decision can be read back
        # against the tape it was actually made on.
        "candidate_ladder_as_of": ((context or {}).get("candidate_ladder") or {}).get("as_of"),
        "candidate_ladder_stale": bool(
            ((context or {}).get("candidate_ladder") or {}).get("staleness_warning")
        ),
    }
    _save(pm_dir, "last_run.json", result)
    mem.journal("agent_pm", {k: result[k] for k in ("equity", "weights", "rationale")}, actor="pm")
    return result


def _money(v: float) -> str:
    """Compact USD for a phone screen: $81.6k, $1.07M, $437k."""
    a = abs(v)
    if a >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    if a >= 1_000:
        return f"${v / 1_000:,.1f}k"
    return f"${v:,.0f}"


def _marks_age_note(book: dict[str, Any]) -> str:
    """'· marks 3d old' when the stored closes are stale, else ''.

    A weight computed from week-old closes is not wrong so much as
    unlabelled — say how old it is rather than quietly implying live."""
    ts = book.get("marks_ts")
    if not ts:
        return ""
    try:
        age = datetime.now(tz=timezone.utc) - datetime.fromisoformat(str(ts))
    except ValueError:
        return ""
    days = age.days
    if days >= 1:
        return f" · marks {days}d old"
    if age.total_seconds() >= 6 * 3600:
        return f" · marks {int(age.total_seconds() // 3600)}h old"
    return ""


def format_holdings(book: dict[str, Any]) -> list[str]:
    """Holding lines carrying their own units.

    A bare ``GLD: 285.839`` is unreadable — a 2026-07-29 transcript shows
    it posted ninety seconds before ``GLD 10%``, the same ticker in a
    different unit with neither one labelled. Share counts, dollar value
    and weight together remove the ambiguity; without marks we still say
    ``sh`` so the number is at least self-describing.
    """
    holdings: dict[str, float] = book.get("holdings", {}) or {}
    if not holdings:
        return ["  _all cash_"]
    marks: dict[str, float] = book.get("marks", {}) or {}
    cash = float(book.get("cash", 0.0))
    values = {s: q * marks[s] for s, q in holdings.items() if s in marks}
    equity = cash + sum(values.values())
    lines = []
    for sym, qty in sorted(holdings.items(), key=lambda kv: -values.get(kv[0], 0.0)):
        if sym in values and equity > 0:
            lines.append(
                f"  `{sym:<5}` {qty:,.1f} sh · {_money(values[sym])} · {values[sym] / equity:.1%}"
            )
        else:
            lines.append(f"  `{sym:<5}` {qty:,.1f} sh · _unmarked_")
    if equity > 0:
        lines.append(f"  `cash ` {_money(cash)} · {cash / equity:.1%}")
    else:
        lines.append(f"  `cash ` {_money(cash)}")
    return lines


def format_pm_digest(result: dict[str, Any], book: dict[str, Any] | None = None) -> str:
    """One message per rebalance — what changed, the resulting book, why.

    Previously the runner sent this *and* a separate book dump for the
    same event, ninety seconds apart, the first carrying the *previous*
    cycle's rationale. One event, one message, led by the delta.
    """
    if not result.get("ok"):
        return f"🤖 Agent PM did not trade: {result.get('reason', 'unknown')}"

    # Always name the book and the currency. This sim is quoted in USD
    # while the real account reports in CHF, and the two used to appear in
    # adjacent messages with neither one labelled.
    lines = [
        "🧪 *Agent PM — simulated book* (USD, not the trading account)",
        f"equity {_money(float(result['equity']))}",
    ]

    opened, closed = result.get("opened") or [], result.get("closed") or []
    if opened or closed:
        change = []
        if closed:
            change.append("exited " + ", ".join(f"`{s}`" for s in closed))
        if opened:
            change.append("opened " + ", ".join(f"`{s}`" for s in opened))
        lines.append("*Changed:* " + " · ".join(change))
    elif result.get("turnover"):
        lines.append("*Changed:* weights only, no names added or dropped")
    else:
        lines.append("*Changed:* nothing")

    # Say it when the book has stopped evolving. Weight churn reads as
    # activity in this digest; an unchanged name set does not, and that is
    # precisely the failure mode that went a month unnoticed.
    stale = int(result.get("cycles_since_name_change") or 0)
    if stale >= STALE_BOOK_CYCLES:
        lines.append(f"⚠️ *same names for {stale} cycles* — book is not evolving")
    if result.get("candidate_ladder_seen") is False:
        lines.append("⚠️ _no candidate ladder this cycle — the PM had no new-name feed_")
    elif result.get("candidate_ladder_stale"):
        lines.append(f"⚠️ _ladder ranks stale bars (as of {result.get('candidate_ladder_as_of')})_")
    if result.get("stale_marks"):
        lines.append(f"_(marked from stored closes: {', '.join(result['stale_marks'])})_")

    lines.append("")
    if book is not None:
        lines.append("*Book now:*")
        lines.extend(format_holdings(book))
    else:
        lines.append(
            "*Target book:* "
            + (
                ", ".join(f"`{s}` {w:.0%}" for s, w in sorted(result["weights"].items()))
                or "all cash"
            )
        )

    lines.extend(
        [
            "",
            f"*Why:* {clip(result.get('rationale', ''), 700)}",
            f"_Watching: {clip(result.get('watch', ''), 300)}_",
        ]
    )
    if result.get("dropped"):
        lines.append(f"_(dropped off-universe/invalid: {', '.join(result['dropped'])})_")
    lines.append(
        f"_turnover {_money(float(result['turnover']))} · costs {_money(float(result['costs']))}_"
    )
    return "\n".join(lines)
