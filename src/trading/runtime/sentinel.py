"""The Sentinel — intraday tripwires, plus a single late-day de-risk gate.

Design (cost-discipline): during US market hours a 15-minute job computes
MECHANICAL triggers from quotes — no LLM, no spend, no judgment:

* SPY down more than SPY_TRIGGER_PCT vs yesterday's close;
* VIX up more than VIX_TRIGGER_RATIO vs yesterday's close;
* any held symbol (real book or sim sleeve) down more than
  HOLDING_TRIGGER_PCT on the day.

When a wire trips, ONE Sentinel agent runs (~$0.02) to judge severity and
write a short assessment. This path is **INFORMATION ONLY** — it alerts the
operator and never convenes the committee. Alerts are debounced: at most one
every ``ALERT_DEBOUNCE_HOURS`` while the tripped SET is unchanged, but a
materially-changed set (a new name, or a name down another
``ESCALATION_STEP_PCT``) pings immediately.

The committee is EXPENSIVE and is convened from exactly three places, none of
them the intraday Sentinel: the twice-weekly schedule, the operator's
``/committee``, and the **late-day de-risk** gate in this module —
``run_late_day_derisk``. That gate runs once, ~50 minutes before the close
(mid-afternoon, so the noisy open is excluded): if a held name has fallen
``DERISK_DROP_PCT`` or more on the day, it convenes the committee a single
time. Nothing here fires after the close.

NOTHING in this module touches the order path — same isolation contract as
every agent. De-risking remains the operator's decision.
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
HOLDING_TRIGGER_PCT = -5.0  # informational alert wire for a held name

# --- Informational-alert cadence (no committee ever attached to this) ---
ALERT_DEBOUNCE_HOURS = 2.0
DEBOUNCE_HOURS = ALERT_DEBOUNCE_HOURS  # backwards-compatible alias
# A tripped name must fall another ESCALATION_STEP_PCT points beyond where it
# last alerted for the ping to bypass the debounce as "materially worse".
ESCALATION_STEP_PCT = 3.0

# --- Late-day de-risk gate (the only price-driven committee trigger) ---
# A held name down this much on the day, judged ~50 min before the close,
# convenes the committee once. Deliberately higher than the info wire.
DERISK_DROP_PCT = 10.0

LlmFn = Callable[[str, str], dict[str, Any]]

SENTINEL_CHARTER = (
    "You are the Sentinel — the intraday risk watch on a systematic desk. "
    "A mechanical tripwire just fired; you decide if it matters and describe "
    "it for the operator. You will see the triggers, current book exposure "
    "and recent context. Most trips are noise: a single name gapping on "
    "earnings, a stale print. Real alarms are systemic: correlated selling, "
    "vol spike with breadth collapse, credit cracking. Do NOT treat any of "
    "these as reassurance: a name near its 52-week high, large unrealized "
    "gains, or a subdued SPY IV — those describe a crowded, extended long "
    "with the most to give back, and quiet IV often precedes repricing, not "
    "safety. When two or more held names in the SAME sector trip together, "
    "your default is correlated selling (a real alarm), NOT 'idiosyncratic' — "
    "only call it idiosyncratic if you can name a stock-specific catalyst for "
    "each. Be decisive and terse. Respond ONLY with "
    'JSON: {"severity": "false_alarm|caution|alarm", '
    '"assessment": "<2-3 sentences: what is happening and why it does or '
    'does not threaten the book>", '
    '"suggested_action": "<one concrete operator action, e.g. nothing / '
    'watch X / consider /mode defense / consider trimming Y>"}'
)


def _envf(key: str, default: float) -> float:
    """Float env override; falls back to ``default`` on missing/garbage."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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


def _tripped_map(held: list[str], moves: dict[str, float]) -> dict[str, float]:
    """Structured tripwire result: ``{symbol: % move}`` for everything that
    breached a wire. ``"SPY"``/``"^VIX"`` are the systemic wires; the rest are
    held names. Content-aware debounce needs the levels, not just strings."""
    out: dict[str, float] = {}
    spy = moves.get("SPY")
    if spy is not None and spy <= SPY_TRIGGER_PCT:
        out["SPY"] = spy
    vix = moves.get("^VIX")
    if vix is not None and vix >= (VIX_TRIGGER_RATIO - 1.0) * 100:
        out["^VIX"] = vix
    for sym in held:
        mv = moves.get(sym)
        if mv is not None and mv <= HOLDING_TRIGGER_PCT:
            out[sym] = mv
    return out


def _triggers_from_map(tripped: dict[str, float]) -> list[str]:
    """Human-readable trigger strings; order stable (SPY, VIX, then held)."""
    out: list[str] = []
    if "SPY" in tripped:
        out.append(f"SPY {tripped['SPY']:+.1f}% on the day")
    if "^VIX" in tripped:
        out.append(f"VIX +{tripped['^VIX']:.0f}% vs yesterday")
    for sym, mv in tripped.items():
        if sym in ("SPY", "^VIX"):
            continue
        out.append(f"{sym} {mv:+.1f}% (held)")
    return out


def check_triggers(state_dir: Path, *, moves: dict[str, float] | None = None) -> list[str]:
    """Mechanical pass — free, no LLM. Returns human-readable trigger strings."""
    held = _held_symbols(state_dir)
    moves = moves if moves is not None else _day_moves(["SPY", "^VIX", *held])
    return _triggers_from_map(_tripped_map(held, moves))


def _material_change(tripped: dict[str, float], anchor: dict[str, float], step: float) -> bool:
    """True when ``tripped`` carries genuinely new risk vs ``anchor``: a symbol
    that was not tripped before, or one that has moved another ``step`` points
    in the risk direction beyond where the anchor recorded it (down for prices,
    up for VIX). An unchanged or recovering set is NOT material."""
    for sym, mv in tripped.items():
        prev = anchor.get(sym)
        if prev is None:
            return True
        if sym == "^VIX":
            if mv >= prev + step:
                return True
        elif mv <= prev - step:
            return True
    return False


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
    """One watch cycle — INFORMATION ONLY. Returns ``{'quiet': True}`` when
    nothing fired or the alert is debounced, else the sentinel's judgment for
    a caution ping. This never convenes the committee."""
    now = now or datetime.now(tz=timezone.utc)
    held = _held_symbols(state_dir)
    moves = moves if moves is not None else _day_moves(["SPY", "^VIX", *held])
    tripped = _tripped_map(held, moves)
    triggers = _triggers_from_map(tripped)
    if not triggers:
        return {"quiet": True}

    alert_debounce_h = _envf("SENTINEL_ALERT_DEBOUNCE_HOURS", ALERT_DEBOUNCE_HOURS)
    step = _envf("SENTINEL_ESCALATION_STEP_PCT", ESCALATION_STEP_PCT)

    state = _load_state(state_dir)
    today = now.date().isoformat()
    if state.get("day") != today:
        # New UTC day: reset the alert anchor but keep the alert clock so a
        # trip seconds after midnight doesn't double-ping.
        state = {"day": today, "last_alert_ts": state.get("last_alert_ts"), "alerted_moves": {}}

    alerted_anchor: dict[str, float] = state.get("alerted_moves") or {}
    material_alert = _material_change(tripped, alerted_anchor, step)

    last = state.get("last_alert_ts")
    if last and not material_alert:
        try:
            age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600
            if age_h < alert_debounce_h:
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
        # LLM down but wires tripped: still inform — better a noisy alert
        # than a silent drawdown.
        verdict = {
            "severity": "caution",
            "assessment": f"sentinel LLM unavailable ({e}); raw triggers forwarded",
            "suggested_action": "review the book manually",
        }

    # Persist alert state, preserving any same-day de-risk marker.
    state["day"] = today
    state["last_alert_ts"] = now.isoformat()
    state["alerted_moves"] = tripped
    state["triggers"] = triggers
    _save_state(state_dir, state)
    return {
        "quiet": False,
        "triggers": triggers,
        "severity": str(verdict.get("severity", "caution")),
        "assessment": str(verdict.get("assessment", ""))[:400],
        "suggested_action": str(verdict.get("suggested_action", ""))[:200],
    }


def run_late_day_derisk(
    state_dir: Path,
    *,
    moves: dict[str, float] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Once-a-day, late-session (~50 min before the close) safety gate. If a
    HELD name has fallen ``DERISK_DROP_PCT`` or more on the day, convene the
    committee ONE time. The noisy open is excluded because this runs
    mid-afternoon; after-close moves are excluded because it is scheduled
    before the close. Mechanical — no LLM here; the committee is the judgment.

    Returns ``{'quiet': True}`` when nothing qualifies (or it already fired
    today), else ``{'convene': True, 'symbols': {...}, 'drops': [...]}``."""
    now = now or datetime.now(tz=timezone.utc)
    drop = abs(_envf("SENTINEL_DERISK_DROP_PCT", DERISK_DROP_PCT))
    held = _held_symbols(state_dir)
    moves = moves if moves is not None else _day_moves(held)
    hits = {s: moves[s] for s in held if moves.get(s) is not None and moves[s] <= -drop}
    if not hits:
        return {"quiet": True}

    today = now.date().isoformat()
    state = _load_state(state_dir)
    if state.get("derisk_date") == today:
        return {"quiet": True, "already_convened": True, "symbols": hits}

    # Mark before convening so a runner restart near the fire time can't
    # double-convene.
    state["derisk_date"] = today
    state.setdefault("day", today)
    _save_state(state_dir, state)

    drops = [f"{s} {mv:+.1f}%" for s, mv in sorted(hits.items(), key=lambda kv: kv[1])]
    logger.bind(component="sentinel").info(f"late-day de-risk convening committee: {drops}")
    return {"quiet": False, "convene": True, "symbols": hits, "drops": drops}


def format_sentinel_alert(result: dict[str, Any]) -> str:
    icon = {"false_alarm": "🟡", "caution": "🟠", "alarm": "🚨"}.get(result["severity"], "🟠")
    lines = [
        f"{icon} *Sentinel — {result['severity'].replace('_', ' ').upper()}*",
        "*Tripped:* " + "; ".join(result["triggers"]),
        result["assessment"],
        f"_Suggested: {result['suggested_action']}_",
        "_Info only — reply /committee if you want the desk to debate it._",
    ]
    return "\n".join(lines)


def format_derisk_alert(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "🚨 *Late-day de-risk* — a holding is down hard into the close",
            "*Tripped (as of ~50 min pre-close):* " + "; ".join(result["drops"]),
            "_Convening the committee now — digest follows._",
        ]
    )
