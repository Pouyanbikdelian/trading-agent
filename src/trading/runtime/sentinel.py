"""The Sentinel — intraday tripwires with an LLM only behind the alarm.

Design (cost-discipline): during US market hours a 15-minute job computes
MECHANICAL triggers from quotes — no LLM, no spend, no judgment:

* SPY down more than SPY_TRIGGER_PCT vs yesterday's close;
* VIX up more than VIX_TRIGGER_RATIO vs yesterday's close;
* any held symbol (real book or sim sleeve) down more than
  HOLDING_TRIGGER_PCT on the day.

Only when a wire trips does ONE Sentinel agent run (~$0.02) to judge
severity: false alarm / caution / alarm. On 'alarm' the runner convenes
the full committee (existing flag path) and Telegram gets an explicit
heads-up with operator options (/mode defense, /flatten, /halt).

NOTHING here touches the order path — same isolation contract as every
agent. De-risking remains the operator's decision. Debounced so a bad
day produces one escalation, not sixteen.
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

STATE_FILENAME = "sentinel.json"
SPY_TRIGGER_PCT = -1.5  # day move that wakes the sentinel
VIX_TRIGGER_RATIO = 1.20  # VIX 20% above yesterday's close
HOLDING_TRIGGER_PCT = -5.0
DEBOUNCE_HOURS = 2.0

LlmFn = Callable[[str, str], dict[str, Any]]

SENTINEL_CHARTER = (
    "You are the Sentinel — the intraday risk watch on a systematic desk. "
    "A mechanical tripwire just fired; you decide if it matters. You will "
    "see the triggers, current book exposure and recent context. Most "
    "trips are noise: a single name gapping on earnings, a stale print. "
    "Real alarms are systemic: correlated selling, vol spike with breadth "
    "collapse, credit cracking. Be decisive and terse. Respond ONLY with "
    'JSON: {"severity": "false_alarm|caution|alarm", '
    '"assessment": "<2-3 sentences: what is happening and why it does or '
    'does not threaten the book>", '
    '"suggested_action": "<one concrete operator action, e.g. nothing / '
    'watch X / consider /mode defense / consider trimming Y>", '
    '"convene_committee": true|false}'
)


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


def _day_moves(symbols: list[str]) -> dict[str, float]:
    """{symbol: % move vs previous close}. Empty on any fetch failure —
    a blind sentinel must stay silent, not guess."""
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
                s = raw[sym]["Close"].dropna()
                if len(s) >= 2:
                    out[sym] = round((float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100, 2)
            except Exception:
                continue
    except Exception as e:
        logger.bind(component="sentinel").warning(f"quote fetch failed: {e}")
    return out


def _held_symbols(state_dir: Path) -> list[str]:
    """Union of the real paper book and the sim sleeve."""
    syms: set[str] = set()
    try:
        from trading.runner.state import RunnerStore

        snap = RunnerStore(state_dir / "runner.db").latest_snapshot()
        if snap:
            syms |= {p.instrument.symbol for p in snap.positions.values()}
    except Exception:
        pass
    try:
        book = json.loads((state_dir / "agent_pm" / "portfolio.json").read_text())
        syms |= set(book.get("holdings", {}))
    except Exception:
        pass
    return sorted(syms)


def check_triggers(state_dir: Path, *, moves: dict[str, float] | None = None) -> list[str]:
    """Mechanical pass — free, no LLM. Returns human-readable trigger strings."""
    held = _held_symbols(state_dir)
    moves = moves if moves is not None else _day_moves(["SPY", "^VIX", *held])
    triggers: list[str] = []
    spy = moves.get("SPY")
    if spy is not None and spy <= SPY_TRIGGER_PCT:
        triggers.append(f"SPY {spy:+.1f}% on the day")
    vix = moves.get("^VIX")
    if vix is not None and vix >= (VIX_TRIGGER_RATIO - 1.0) * 100:
        triggers.append(f"VIX +{vix:.0f}% vs yesterday")
    for sym in held:
        mv = moves.get(sym)
        if mv is not None and mv <= HOLDING_TRIGGER_PCT:
            triggers.append(f"{sym} {mv:+.1f}% (held)")
    return triggers


def _load_state(state_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(state_dir) / STATE_FILENAME).read_text())
    except Exception:
        return {}


def _save_state(state_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def run_sentinel(
    state_dir: Path,
    *,
    llm: LlmFn | None = None,
    moves: dict[str, float] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One watch cycle. Returns {'quiet': True} when nothing fired, else
    the sentinel's judgment (debounced)."""
    now = now or datetime.now(tz=timezone.utc)
    triggers = check_triggers(state_dir, moves=moves)
    if not triggers:
        return {"quiet": True}

    state = _load_state(state_dir)
    last = state.get("last_alert_ts")
    if last:
        try:
            age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600
            if age_h < DEBOUNCE_HOURS:
                return {"quiet": True, "debounced": True, "triggers": triggers}
        except Exception:
            pass

    llm = llm or _default_llm
    try:
        from trading.agents.context import build_context
        from trading.core.config import settings

        ctx = build_context(Path(state_dir), settings.data_dir)
    except Exception:
        ctx = {}
    prompt = json.dumps(
        {
            "triggers": triggers,
            "positions": ctx.get("positions", []),
            "vol_surface": ctx.get("vol_surface", {}),
            "macro_dial": ctx.get("macro_dial", {}),
        },
        default=str,
    )[:8000]
    try:
        verdict = llm(SENTINEL_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="sentinel").warning(f"sentinel call failed: {e}")
        # LLM down but wires tripped: escalate blindly — better a noisy
        # alert than a silent drawdown.
        verdict = {
            "severity": "caution",
            "assessment": f"sentinel LLM unavailable ({e}); raw triggers forwarded",
            "suggested_action": "review the book manually",
            "convene_committee": False,
        }
    _save_state(state_dir, {"last_alert_ts": now.isoformat(), "triggers": triggers})
    return {
        "quiet": False,
        "triggers": triggers,
        "severity": str(verdict.get("severity", "caution")),
        "assessment": str(verdict.get("assessment", ""))[:400],
        "suggested_action": str(verdict.get("suggested_action", ""))[:200],
        "convene_committee": bool(verdict.get("convene_committee")),
    }


def format_sentinel_alert(result: dict[str, Any]) -> str:
    icon = {"false_alarm": "🟡", "caution": "🟠", "alarm": "🚨"}.get(result["severity"], "🟠")
    lines = [
        f"{icon} *Sentinel — {result['severity'].replace('_', ' ').upper()}*",
        "*Tripped:* " + "; ".join(result["triggers"]),
        result["assessment"],
        f"_Suggested: {result['suggested_action']}_",
    ]
    if result.get("convene_committee"):
        lines.append("_Convening the committee now — digest follows._")
    return "\n".join(lines)
