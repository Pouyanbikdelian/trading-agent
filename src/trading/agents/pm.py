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

START_EQUITY = 100_000.0
COST_BPS = 10.0  # commission + slippage on turnover, charitable but not free
MAX_WEIGHT_PER_NAME = 0.25
MAX_GROSS = 1.0  # long-only, no leverage

# Fixed, liquid whitelist: broad + sectors + themes + defense assets.
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
    "record, and today's market context. Decide the sleeve's allocation "
    "for the coming week. Rules: long-only; weights sum to at most 1.0 "
    "(the remainder is cash); only tickers from the allowed universe; "
    "max 0.25 in any one name. Trust calibration over confidence — an "
    "agent who has been wrong all month is a fade. Be decisive: persistent "
    "committee drift, scout themes confirmed by relative momentum, and "
    "risk-officer warnings are all actionable. Respond ONLY with JSON: "
    '{"target_weights": {"<TICKER>": <0.0-0.25>, ...}, '
    '"rationale": "<4-6 sentences: the trade and the why>", '
    '"watch": "<what would change your mind this week>"}'
)


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


def _clamp_weights(raw: Any) -> dict[str, float]:
    """Hard caps: whitelist, long-only, per-name max, gross max."""
    weights: dict[str, float] = {}
    if not isinstance(raw, dict):
        return weights
    for sym, w in raw.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        sym = str(sym).upper().strip()
        if sym not in UNIVERSE or w <= 0:
            continue
        weights[sym] = min(w, MAX_WEIGHT_PER_NAME)
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
        return json.loads(path.read_text())
    except Exception:
        return {"cash": START_EQUITY, "holdings": {}, "history": []}


def _save(pm_dir: Path, name: str, payload: dict[str, Any]) -> None:
    pm_dir.mkdir(parents=True, exist_ok=True)
    path = pm_dir / name
    fd, tmp = tempfile.mkstemp(dir=pm_dir, prefix=f"{name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, default=str, indent=1)
    os.replace(tmp, path)


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

    # --- mark current book to market
    symbols = sorted(set(book["holdings"]) | set(UNIVERSE))
    px = prices if prices is not None else _fetch_closes(symbols)
    equity = float(book["cash"]) + sum(
        qty * px[s] for s, qty in book["holdings"].items() if s in px
    )
    unpriced = [s for s in book["holdings"] if s not in px]
    if unpriced:
        logger.bind(component="agent_pm").warning(f"no price for held {unpriced}; skipping run")
        return {"ok": False, "reason": f"missing prices for held positions: {unpriced}"}

    # --- assemble the PM's evidence: week of rulings + calibration + context
    week = mem.journal_tail(6, kind="committee")
    prompt = json.dumps(
        {
            "sim_portfolio": {
                "equity": round(equity, 2),
                "cash": round(float(book["cash"]), 2),
                "holdings": book["holdings"],
            },
            "allowed_universe": list(UNIVERSE),
            "week_of_committee_rulings": week,
            "agent_calibration": mem.calibration(),
            "today_context": context,
        },
        default=str,
    )[:24000]

    try:
        out = llm(PM_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="agent_pm").warning(f"PM call failed: {e}")
        return {"ok": False, "reason": f"PM call failed: {e}"}

    weights = _clamp_weights(out.get("target_weights"))
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
