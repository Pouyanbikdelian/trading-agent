"""The Sentinel — intraday tripwires with an LLM only behind the alarm.

Design (cost-discipline): during US market hours a 15-minute job computes
MECHANICAL triggers from quotes — no LLM, no spend, no judgment:

* SPY down more than SPY_TRIGGER_PCT vs yesterday's close;
* VIX up more than VIX_TRIGGER_RATIO vs yesterday's close;
* any held symbol (real book or sim sleeve) down more than
  HOLDING_TRIGGER_PCT on the day.

Only when a wire trips does ONE Sentinel agent run (~$0.02) to judge
severity: false alarm / caution / alarm.

Two INDEPENDENT gates keep a bad day from spamming — this is the whole
point of the module:

* Caution ping (cheap). At most one every ``ALERT_DEBOUNCE_HOURS`` while
  the tripped SET is unchanged, so you keep getting a heads-up that the
  names are still down. A materially-changed set — a NEW name trips, or an
  already-tripped name falls another ``ESCALATION_STEP_PCT`` — pings
  immediately regardless of the clock, because that is news.

* Committee (expensive). The full multi-voice debate is convened ONLY on a
  materially-changed set, and never more than ``MAX_COMMITTEE_PER_DAY``
  times per UTC day from here. A persistent drawdown on the SAME names
  re-pings but does NOT re-convene: the committee already ruled on those
  names, and re-running it every couple of hours on identical information
  is pure cost and noise. Genuine deterioration (a new leg down, a fresh
  name) still escalates.

All three cadence knobs are env-overridable so they can be tuned without a
code change: ``SENTINEL_ALERT_DEBOUNCE_HOURS``, ``SENTINEL_ESCALATION_STEP_PCT``,
``SENTINEL_COMMITTEE_MAX_PER_DAY``.

NOTHING here touches the order path — same isolation contract as every
agent. De-risking remains the operator's decision.
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

# --- Escalation cadence: two independent gates (see module docstring) ---
# Cheap caution ping: min spacing while the tripped set is unchanged.
ALERT_DEBOUNCE_HOURS = 2.0
DEBOUNCE_HOURS = ALERT_DEBOUNCE_HOURS  # backwards-compatible alias
# A tripped name must fall another ESCALATION_STEP_PCT points beyond where
# we last escalated for it to count as "materially worse".
ESCALATION_STEP_PCT = 3.0
# Hard ceiling on committee convenings triggered by the sentinel per UTC day.
# 2 leaves room for the initial alarm plus one genuine second-leg-down,
# on top of the always-on scheduled morning committee.
MAX_COMMITTEE_PER_DAY = 2

LlmFn = Callable[[str, str], dict[str, Any]]

SENTINEL_CHARTER = (
    "You are the Sentinel — the intraday risk watch on a systematic desk. "
    "A mechanical tripwire just fired; you decide if it matters. You will "
    "see the triggers, current book exposure and recent context. Most "
    "trips are noise: a single name gapping on earnings, a stale print. "
    "Real alarms are systemic: correlated selling, vol spike with breadth "
    "collapse, credit cracking. Do NOT treat any of these as reassurance: a "
    "name near its 52-week high, large unrealized gains, or a subdued SPY IV — "
    "those describe a crowded, extended long with the most to give back, and "
    "quiet IV often precedes repricing, not safety. When two or more held names "
    "in the SAME sector trip together, your default is correlated selling (a "
    "real alarm), NOT 'idiosyncratic' — only call it idiosyncratic if you can "
    "name a stock-specific catalyst for each. Be decisive and terse. Respond "
    "ONLY with "
    'JSON: {"severity": "false_alarm|caution|alarm", '
    '"assessment": "<2-3 sentences: what is happening and why it does or '
    'does not threaten the book>", '
    '"suggested_action": "<one concrete operator action, e.g. nothing / '
    'watch X / consider /mode defense / consider trimming Y>", '
    '"convene_committee": true|false}'
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


def _envi(key: str, default: int) -> int:
    """Int env override; falls back to ``default`` on missing/garbage."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
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
    held names. Content-aware escalation needs the levels, not just strings."""
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
    up for VIX). An unchanged or recovering set is NOT material — that is what
    stops the committee from re-running all day on the same names."""
    for sym, mv in tripped.items():
        prev = anchor.get(sym)
        if prev is None:
            return True  # a name/wire that wasn't tripped at the anchor
        if sym == "^VIX":
            if mv >= prev + step:  # vol spiking further
                return True
        elif mv <= prev - step:  # price falling further
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
    """One watch cycle. Returns ``{'quiet': True}`` when nothing fired or the
    caution ping is debounced, else the sentinel's judgment. ``convene_committee``
    in the result is the already-gated decision (content-aware + daily cap), so
    the caller can act on it directly."""
    now = now or datetime.now(tz=timezone.utc)
    held = _held_symbols(state_dir)
    moves = moves if moves is not None else _day_moves(["SPY", "^VIX", *held])
    tripped = _tripped_map(held, moves)
    triggers = _triggers_from_map(tripped)
    if not triggers:
        return {"quiet": True}

    alert_debounce_h = _envf("SENTINEL_ALERT_DEBOUNCE_HOURS", ALERT_DEBOUNCE_HOURS)
    step = _envf("SENTINEL_ESCALATION_STEP_PCT", ESCALATION_STEP_PCT)
    max_per_day = _envi("SENTINEL_COMMITTEE_MAX_PER_DAY", MAX_COMMITTEE_PER_DAY)

    state = _load_state(state_dir)
    today = now.date().isoformat()
    if state.get("day") != today:
        # New UTC day: reset the daily counters/anchors but keep the alert
        # clock so a trip seconds after midnight doesn't double-ping.
        state = {
            "day": today,
            "committee_count": 0,
            "convened_moves": {},
            "alerted_moves": {},
            "last_alert_ts": state.get("last_alert_ts"),
        }

    alerted_anchor: dict[str, float] = state.get("alerted_moves") or {}
    convened_anchor: dict[str, float] = state.get("convened_moves") or {}
    material_alert = _material_change(tripped, alerted_anchor, step)
    material_committee = _material_change(tripped, convened_anchor, step)

    # --- Gate 1: caution ping. Light cadence, bypassed on a new/worse set. ---
    last = state.get("last_alert_ts")
    if last and not material_alert:
        try:
            age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600
            if age_h < alert_debounce_h:
                return {"quiet": True, "debounced": True, "triggers": triggers}
        except Exception:
            pass

    # We're alerting: run the Sentinel LLM (~$0.02).
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

    # --- Gate 2: committee. New/worse info only, capped per UTC day. ---
    count = int(state.get("committee_count", 0))
    llm_wants = bool(verdict.get("convene_committee"))
    convene = llm_wants and material_committee and count < max_per_day
    suppressed_reason = ""
    if llm_wants and not convene:
        if not material_committee:
            suppressed_reason = "committee already ruled on these names today; no new deterioration"
        elif count >= max_per_day:
            suppressed_reason = f"daily committee cap ({max_per_day}) reached"
        logger.bind(component="sentinel").info(f"committee convene suppressed: {suppressed_reason}")

    _save_state(
        state_dir,
        {
            "day": today,
            "last_alert_ts": now.isoformat(),
            "alerted_moves": tripped,
            "convened_moves": tripped if convene else convened_anchor,
            "committee_count": count + (1 if convene else 0),
            "triggers": triggers,
        },
    )
    return {
        "quiet": False,
        "triggers": triggers,
        "severity": str(verdict.get("severity", "caution")),
        "assessment": str(verdict.get("assessment", ""))[:400],
        "suggested_action": str(verdict.get("suggested_action", ""))[:200],
        "convene_committee": convene,
        "committee_suppressed": bool(suppressed_reason),
        "committee_suppressed_reason": suppressed_reason,
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
    elif result.get("committee_suppressed"):
        reason = result.get("committee_suppressed_reason") or "no new deterioration"
        lines.append(f"_Committee held ({reason}). Reply /committee to force a debate._")
    return "\n".join(lines)
