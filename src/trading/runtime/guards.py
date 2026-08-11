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
  GUARD_TRAIL_FLOOR=             per-position ratchet: distance never shrinks
                                 below FLOOR × the entry distance (e.g. 0.4)
  GUARD_TRAIL_TIGHTEN=           per-position ratchet: distance multiplier is
                                 max(FLOOR, 1 − TIGHTEN × gain). Both must be
                                 set (< 1 and > 0) or the trail stays classic.
                                 Backtested 2026-07-02 (scripts/guard_sweep.py):
                                 0.4 / 1.2 — breathe early, lock in late.
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
        loaded: dict[str, Any] = json.loads((Path(state_dir) / STATE_FILENAME).read_text())
        return loaded
    except Exception:
        return {"positions": {}, "equity_baseline": None, "equity_hwm": None, "exits": {}}


def _save(state_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


#: How far the quoted price may sit from the broker's own mark before we
#: refuse to act on it. A trailing stop turns one number into a
#: full-position market sell, and that number comes from yfinance — a
#: free feed that occasionally serves an unadjusted price across a split,
#: a stale bar, or a bad tick. Any of those is indistinguishable from a
#: crash to a naive comparison, and the guard's response is irreversible.
#:
#: The broker already gives us an independent mark for every position:
#: avg_price + unrealized_pnl / quantity. Two sources disagreeing by more
#: than this means we do not know the price, and "we do not know" must
#: not be a reason to sell.
PRICE_SANITY_PCT = 20.0


def _price_is_sane(quoted: float, mark: float | None, tol_pct: float) -> bool:
    """True when the quoted price is close enough to the broker's mark.

    No mark available (fresh position, zero quantity, missing PnL) means
    we cannot cross-check — return True rather than blocking the guards
    entirely. Half a safety net beats none, and a position with no mark
    is usually one we just opened.
    """
    if mark is None or mark <= 0 or quoted <= 0:
        return True
    return abs(quoted - mark) / mark * 100.0 <= tol_pct


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
    the runner owns actually doing both.

    Each position dict may carry ``mark`` — the broker's own valuation of
    the holding. When present it is used to sanity-check the quoted price
    before any exit; see ``PRICE_SANITY_PCT``.
    """
    now = now or datetime.now(tz=timezone.utc)
    holds = holds or set()
    state = _load(state_dir)
    pos_state: dict[str, Any] = state.setdefault("positions", {})
    exits_done: dict[str, str] = state.setdefault("exits", {})
    exit_orders: list[dict[str, Any]] = []
    alerts: list[str] = []
    # The per-name exit messages, also appended to `alerts` so existing
    # callers are unaffected. Tracked separately so the runner can swap
    # them for one roll-up when several fire at once.
    exit_alerts: list[str] = []

    tp_pct = _env_f("GUARD_TP_PCT", None)
    # Per-position profit ratchet: both knobs must be set sensibly, else the
    # trail is the classic fixed-distance one (bit-identical to before).
    r_floor = _env_f("GUARD_TRAIL_FLOOR", None)
    r_tighten = _env_f("GUARD_TRAIL_TIGHTEN", None)
    ratchet_on = (
        r_floor is not None and r_tighten is not None and 0.0 < r_floor < 1.0 and r_tighten > 0.0
    )
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
        eff_pct = float(st["stop_pct"])
        if ratchet_on:
            # Distance shrinks as the position's gain grows — a winner earns
            # a tighter leash. gain is measured off avg entry, from the HWM
            # so a pullback can't loosen an already-tightened trail.
            basis = float(p.get("avg_price", px)) or px
            gain = max(0.0, float(st["hwm"]) / basis - 1.0)
            eff_pct *= max(float(r_floor or 0.0), 1.0 - float(r_tighten or 0.0) * gain)
        stop_level = float(st["hwm"]) * (1.0 - eff_pct / 100.0)
        # A trail only ever rises: never publish a level below the last one.
        stop_level = max(stop_level, float(st.get("stop_level") or 0.0))
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

        # Cross-check the quoted price against the broker's own mark
        # BEFORE any exit decision. The state above (hwm, stop_level) is
        # still updated from the quote — a suspect print should not also
        # be allowed to ratchet the trail — but nothing gets sold on a
        # number two independent sources disagree about.
        mark = p.get("mark")
        tol = _env_f("GUARD_PRICE_SANITY_PCT", PRICE_SANITY_PCT) or PRICE_SANITY_PCT
        if not _price_is_sane(px, None if mark is None else float(mark), tol):
            alerts.append(
                f"⚠️ *{sym} price looks wrong* — quote {px:.2f} vs broker mark "
                f"{float(mark):.2f} ({abs(px - float(mark)) / float(mark) * 100:.0f}% apart, "
                f"limit {tol:.0f}%). Guard exits SKIPPED for this symbol.\n"
                "_Usually an unadjusted price across a split, a stale bar, or a bad "
                "tick. Check the symbol before trusting any stop on it._"
            )
            logger.bind(component="guards").warning(
                f"{sym}: quote {px} diverges from broker mark {mark}; skipping guard exit"
            )
            continue

        # Every exit carries the numbers the operator needs to decide what
        # to do next — proceeds, realized P&L, why. The runner only reads
        # `symbol` and `reason`, so the extra keys cost nothing there and
        # save the roll-up from re-deriving what is already in scope.
        qty = abs(float(p.get("qty", 0.0)))

        def _record(reason: str) -> dict[str, Any]:
            return {
                "symbol": sym,
                "reason": reason,
                "price": px,
                "qty": qty,
                "proceeds": qty * px,
                "entry": avg,
                "pnl_pct": (px / avg - 1.0) * 100.0 if avg else 0.0,
                "pnl_usd": (px - avg) * qty,
            }

        if px <= stop_level:
            exit_orders.append({**_record("trailing_stop"), "hwm": float(st["hwm"])})
            msg = (
                f"🛑 *Trailing stop* {sym}: {px:.2f} ≤ stop {stop_level:.2f} "
                f"(HWM {float(st['hwm']):.2f} − {eff_pct:.1f}%"
                f"{' ratcheted' if ratchet_on and eff_pct < float(st['stop_pct']) else ''}) — closing"
            )
            alerts.append(msg)
            exit_alerts.append(msg)
            exits_done[sym] = now.isoformat()
        elif tp_pct and px >= avg * (1.0 + tp_pct / 100.0):
            exit_orders.append(_record("take_profit"))
            msg = f"🎯 *Take profit* {sym}: {px:.2f} ≥ +{tp_pct:.0f}% from {avg:.2f} — closing"
            alerts.append(msg)
            exit_alerts.append(msg)
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
    # Value still at risk after these exits, for the roll-up's "deployed"
    # line. Computed here because positions/prices/equity are all in
    # scope; deriving it later would mean re-fetching a snapshot that has
    # not been written yet.
    exited = {e["symbol"] for e in exit_orders}
    remaining = sum(
        abs(float(p.get("qty", 0.0))) * float(prices.get(p["symbol"]) or 0.0)
        for p in positions
        if p["symbol"] not in exited
    )
    return {
        "exits": exit_orders,
        "alerts": alerts,
        "exit_alerts": exit_alerts,
        "rollup": format_exit_rollup(exit_orders, equity=equity, remaining_value=remaining, now=now),
    }


def _money(x: float) -> str:
    """`$12,430` / `−$1,240` — sign as a real minus, never a hyphen."""
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def format_exit_rollup(
    exits: list[dict[str, Any]],
    *,
    equity: float | None,
    remaining_value: float = 0.0,
    now: datetime | None = None,
) -> str | None:
    """One message for a multi-name stop-out, or None for 0-1 exits.

    A single exit already reads well as its own line; six of them arrive
    as six separate bubbles the operator has to add up by hand, at the
    exact moment they are least inclined to. What they need in one glance
    is: how much went to cash, what it cost, and what is left.

    The table sits in a fenced block on purpose — Telegram renders it
    monospaced so the columns line up, and does not parse entities inside
    it, so a symbol with an underscore cannot break the whole message.
    """
    if len(exits) < 2:
        return None

    proceeds = sum(float(e.get("proceeds") or 0.0) for e in exits)
    realized = sum(float(e.get("pnl_usd") or 0.0) for e in exits)
    stamp = (now or datetime.now(tz=timezone.utc)).strftime("%d %b %H:%M UTC")

    rows = []
    for e in sorted(exits, key=lambda x: float(x.get("proceeds") or 0.0), reverse=True):
        pnl = float(e.get("pnl_pct") or 0.0)
        rows.append(
            (
                str(e["symbol"]),
                f"{float(e.get('price') or 0.0):,.2f}",
                f"{'+' if pnl >= 0 else '−'}{abs(pnl):.1f}%",
                _money(float(e.get("proceeds") or 0.0)).replace("$", ""),
                "stop" if e.get("reason") == "trailing_stop" else "target",
            )
        )
    w = [max(len(r[i]) for r in rows) for i in range(5)]
    table = "\n".join(
        f"{r[0]:<{w[0]}}  {r[1]:>{w[1]}}  {r[2]:>{w[2]}}  {r[3]:>{w[3]}}  {r[4]}" for r in rows
    )

    stops = sum(1 for e in exits if e.get("reason") == "trailing_stop")
    kind = (
        "trailing stops"
        if stops == len(exits)
        else "profit targets"
        if stops == 0
        else f"{stops} stop{'s' if stops > 1 else ''}, "
        f"{len(exits) - stops} target{'s' if len(exits) - stops > 1 else ''}"
    )

    lines = [
        f"🛑 *Guard exits — {len(exits)} positions closed*",
        f"_{stamp} · {kind}_",
        "",
        f"```\n{table}\n```",
        f"*To cash* {_money(proceeds)}   *Realized* {_money(realized)}",
    ]
    if equity:
        lines.append(f"*Still deployed* {_money(remaining_value)} of {_money(equity)} "
                     f"({remaining_value / equity * 100:.0f}%)")
    lines += [
        "",
        "⚠️ *Nothing redeploys automatically.* The desk has not seen these exits.",
        "Reassess before buying back — `/cycle` alone would re-buy the same names:",
        "`/pm run` → `/pm` → `/cycle`",
    ]
    return "\n".join(lines)
