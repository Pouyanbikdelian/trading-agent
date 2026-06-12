"""Position guards — ATR trailing stops + a portfolio profit ratchet.

Two layers, both OFF until configured (numbers are Yan's call, not code's):

* **Per-position trailing stop** (``GUARDS_ENABLED=true``): each position
  tracks its high-water mark; the stop trails it at a distance scaled by
  the symbol's own volatility — calm names get tight stops, semis get
  room. Anti-squeeze: the distance is clamped to [min, max] so neither a
  vol spike nor a sleepy tape produces a stop that's trivially hunted.
  Breach -> a CLOSE command through the EXISTING command pipeline
  (halt-aware, risk-checked, audited) + a Telegram note. Operator /hold
  pins are respected — a pinned position is never guard-exited.

* **Portfolio ratchet** (``GUARD_LOCK_ARM_PCT`` + ``GUARD_LOCK_GIVEBACK_PCT``):
  once account equity is up ARM%% from its baseline, the ratchet arms;
  if equity then gives back GIVEBACK%% of the peak GAIN, it alerts loudly
  and suggests defensive action. Advisory at portfolio level by design —
  wholesale de-risking is an operator decision; the per-position trails
  are the teeth.

Env knobs (all optional):
  GUARDS_ENABLED=true            master switch for trailing stops
  GUARD_ATR_MULT=3.0             stop distance in volatility units
  GUARD_TRAIL_MIN_PCT=8          anti-squeeze floor
  GUARD_TRAIL_MAX_PCT=20         disaster cap
  GUARD_TP_PCT=                  optional static take-profit (e.g. 35)
  GUARD_LOCK_ARM_PCT=            ratchet arms at +X% equity (e.g. 20)
  GUARD_LOCK_GIVEBACK_PCT=       alert after losing Y% of peak gain (e.g. 40)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

STATE_FILENAME = "guards.json"
EXIT_COOLDOWN_HOURS = 24.0  # one guard exit per symbol per day


def _env_f(name: str, default: float | None) -> float | None:
    raw = os.getenv(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def enabled() -> bool:
    return os.getenv("GUARDS_ENABLED", "false").lower() in ("true", "1", "yes")


def _vol_pct(data_dir: Path, symbol: str, lookback: int = 14) -> float | None:
    """Volatility proxy in percent: mean absolute daily move over the
    lookback, from the parquet cache. Close-to-close (no intraday H/L
    needed) — multiplied up by GUARD_ATR_MULT it serves the same role as
    an ATR distance."""
    try:
        from trading.runtime.portfolio_stats import _read_close

        s = _read_close(data_dir, symbol)
        if s is None or len(s) < lookback + 1:
            return None
        moves = s.pct_change().dropna().iloc[-lookback:]
        return float(moves.abs().mean()) * 100.0
    except Exception:
        return None


def _stop_distance_pct(data_dir: Path, symbol: str) -> float:
    mult = _env_f("GUARD_ATR_MULT", 3.0) or 3.0
    lo = _env_f("GUARD_TRAIL_MIN_PCT", 8.0) or 8.0
    hi = _env_f("GUARD_TRAIL_MAX_PCT", 20.0) or 20.0
    vol = _vol_pct(data_dir, symbol)
    if vol is None:
        return lo  # unknown vol -> tightest allowed, not widest
    return max(lo, min(hi, mult * vol))


def last_prices(symbols: list[str]) -> dict[str, float]:
    """Latest close per symbol via yfinance; missing symbols absent."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    try:
        import yfinance as yf

        raw = yf.download(
            " ".join(symbols),
            period="2d",
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
        logger.bind(component="guards").warning(f"price fetch failed: {e}")
    return out


def _load(state_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(state_dir) / STATE_FILENAME).read_text())
    except Exception:
        return {"positions": {}, "equity_baseline": None, "equity_hwm": None, "exits": {}}


def _save(state_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def check_guards(
    state_dir: Path,
    data_dir: Path,
    *,
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    equity: float | None,
    holds: set[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One guard pass. Pure decision logic — injectable inputs, no I/O
    beyond the guard state file. Returns exits to submit + alerts to send;
    the runner owns actually doing both."""
    now = now or datetime.now(tz=timezone.utc)
    holds = holds or set()
    state = _load(state_dir)
    pos_state: dict[str, Any] = state.setdefault("positions", {})
    exits_done: dict[str, str] = state.setdefault("exits", {})
    exit_orders: list[dict[str, Any]] = []
    alerts: list[str] = []

    tp_pct = _env_f("GUARD_TP_PCT", None)
    live_syms = set()
    for p in positions:
        sym = str(p["symbol"])
        live_syms.add(sym)
        px = prices.get(sym)
        if px is None or float(p.get("qty", 0)) <= 0:
            continue
        st = pos_state.get(sym) or {
            "hwm": max(px, float(p.get("avg_price", px))),
            "stop_pct": _stop_distance_pct(data_dir, sym),
        }
        st["hwm"] = max(float(st["hwm"]), px)
        stop_level = float(st["hwm"]) * (1.0 - float(st["stop_pct"]) / 100.0)
        st["stop_level"] = round(stop_level, 2)
        pos_state[sym] = st

        if sym in holds:
            continue  # operator pinned it; guards keep hands off
        last_exit = exits_done.get(sym)
        if last_exit:
            try:
                age_h = (now - datetime.fromisoformat(last_exit)).total_seconds() / 3600
                if age_h < EXIT_COOLDOWN_HOURS:
                    continue
            except Exception:
                pass

        avg = float(p.get("avg_price", px)) or px
        if px <= stop_level:
            exit_orders.append({"symbol": sym, "reason": "trailing_stop"})
            alerts.append(
                f"🛑 *Trailing stop* {sym}: {px:.2f} ≤ stop {stop_level:.2f} "
                f"(HWM {float(st['hwm']):.2f} − {float(st['stop_pct']):.1f}%) — closing"
            )
            exits_done[sym] = now.isoformat()
        elif tp_pct and px >= avg * (1.0 + tp_pct / 100.0):
            exit_orders.append({"symbol": sym, "reason": "take_profit"})
            alerts.append(
                f"🎯 *Take profit* {sym}: {px:.2f} ≥ +{tp_pct:.0f}% from {avg:.2f} — closing"
            )
            exits_done[sym] = now.isoformat()

    # Drop state for positions no longer held (re-entries start fresh).
    for sym in list(pos_state):
        if sym not in live_syms:
            del pos_state[sym]

    # --- portfolio ratchet (advisory)
    arm = _env_f("GUARD_LOCK_ARM_PCT", None)
    giveback = _env_f("GUARD_LOCK_GIVEBACK_PCT", None)
    if equity and arm and giveback:
        if not state.get("equity_baseline"):
            state["equity_baseline"] = equity
        base = float(state["equity_baseline"])
        state["equity_hwm"] = max(float(state.get("equity_hwm") or equity), equity)
        hwm = float(state["equity_hwm"])
        peak_gain = hwm - base
        armed = peak_gain >= base * arm / 100.0
        if armed and peak_gain > 0:
            kept = equity - base
            if kept <= peak_gain * (1.0 - giveback / 100.0) and not state.get("lock_alerted"):
                alerts.append(
                    f"🔒 *Profit ratchet*: gave back {giveback:.0f}% of the peak gain "
                    f"(peak +{peak_gain / base:+.1%}, now +{kept / base:+.1%}). "
                    f"Consider `/mode defense` or trimming — securing gains was the plan."
                )
                state["lock_alerted"] = True
            elif kept > peak_gain * (1.0 - giveback / 100.0):
                state["lock_alerted"] = False  # re-arm after recovery

    _save(state_dir, state)
    if exit_orders:
        logger.bind(component="guards").info(f"guard exits: {[e['symbol'] for e in exit_orders]}")
    return {"exits": exit_orders, "alerts": alerts}
