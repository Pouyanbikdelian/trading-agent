"""What left the book, why, and what the tape was doing that day.

The desk had no memory of its own trades at the moment it decided. It saw
the current book, distilled lessons, and nothing about outcomes — not
whether the last trade in a name made or lost money, not why the name
left, and not that the guards had sold anything at all. ``exits_done`` is
private to ``runtime.guards`` and gates re-EXIT, not re-ENTRY, so a
standing PM decision would happily re-buy a name stopped out that morning.

Two things are supplied here, and the split is deliberate:

* **Facts** — reason, entry, exit, realized P&L, days held, and the
  same-day move of both the name and SPY. Computed, not judged.
* **A stated rule** for reading those facts (market-wide vs specific),
  with the rule text alongside it, so an agent that disagrees can say so
  rather than inheriting a verdict it cannot inspect.

A hard label with no visible rule is how a heuristic becomes folklore.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

#: A name and the market are judged to have moved together when the name's
#: excess return over SPY on the exit day is no worse than this.
MARKET_WIDE_EXCESS_PCT = -3.0
#: Below this excess, the break is the name's own, whatever SPY did.
NAME_SPECIFIC_EXCESS_PCT = -6.0
#: SPY itself must be down at least this much for "market-wide" to mean
#: anything — otherwise a name down 2% on a flat tape reads as market-wide.
MARKET_DOWN_PCT = -1.0

_RULE = (
    f"market-wide = SPY down >{abs(MARKET_DOWN_PCT):.0f}% and the name within "
    f"{abs(MARKET_WIDE_EXCESS_PCT):.0f}pp of it; name-specific = the name "
    f"lagged SPY by more than {abs(NAME_SPECIFIC_EXCESS_PCT):.0f}pp; "
    "otherwise mixed. A rule of thumb, not a finding — disagree with it if "
    "the evidence says otherwise."
)


def _classify(name_pct: float | None, spy_pct: float | None) -> str:
    if name_pct is None or spy_pct is None:
        return "unknown (no price history for that day)"
    excess = name_pct - spy_pct
    if excess <= NAME_SPECIFIC_EXCESS_PCT:
        return "name-specific"
    if spy_pct <= MARKET_DOWN_PCT and excess >= MARKET_WIDE_EXCESS_PCT:
        return "market-wide"
    return "mixed"


def _day_move_pct(
    data_dir: Path, symbol: str, day: str, cache: dict[str, Any] | None = None
) -> float | None:
    """The symbol's close-to-close move on ``day`` (YYYY-MM-DD), or None.

    ``cache`` is per-call, not module-level: SPY is looked up once per
    exit, so a dozen exits meant a dozen reads of the same parquet, but a
    long-lived cache inside the runner would serve a stale series for the
    rest of the process's life.
    """
    if cache is not None and symbol in cache:
        pct = cache[symbol]
        if pct is None:
            return None
        for idx, val in zip(pct.index[::-1], pct.to_numpy()[::-1], strict=False):
            if str(idx)[:10] == day:
                return None if val != val else round(float(val), 2)
        return None
    try:
        from trading.runtime.portfolio_stats import _read_close

        s = _read_close(data_dir, symbol)
        if s is None or len(s) < 2:
            if cache is not None:
                cache[symbol] = None
            return None
        pct = s.pct_change() * 100.0
        if cache is not None:
            cache[symbol] = pct
        # Index may be tz-aware or naive; compare on the date string.
        for idx, val in zip(pct.index[::-1], pct.to_numpy()[::-1], strict=False):
            if str(idx)[:10] == day:
                return None if val != val else round(float(val), 2)  # NaN check
    except Exception as e:
        logger.bind(component="agents").warning(f"exits: day move for {symbol} failed ({e})")
    return None


def _as_dt(ts: Any) -> datetime | None:
    """``fills_with_symbols`` hands back epoch floats; tests hand datetimes.

    Accepting both keeps the walk testable without a sqlite fixture, and
    a float that silently fails an isinstance check is exactly how this
    returned an empty list while every unit test passed.
    """
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        return None


def _closed_positions(fills: list[dict[str, Any]], since: datetime) -> list[dict[str, Any]]:
    """Walk the fill stream and emit one record per position that went flat.

    Average cost, matching ``dashboard.live.realized_by_symbol`` and IBKR's
    default P&L view — the two must agree or the same trade shows two
    different numbers in two places the operator reads side by side.

    The walk starts from the beginning of the ledger, not from ``since``:
    a position opened months ago and closed yesterday needs its whole
    history to have a cost basis at all.
    """
    book: dict[str, dict[str, float]] = {}
    opened_at: dict[str, Any] = {}
    out: list[dict[str, Any]] = []

    for f in fills:
        sym = f["symbol"]
        b = book.setdefault(sym, {"qty": 0.0, "avg": 0.0})
        signed = float(f["qty"]) if f["side"] == "BUY" else -float(f["qty"])
        q0, a0 = b["qty"], b["avg"]
        if q0 == 0.0:
            opened_at[sym] = _as_dt(f["ts"])
        if q0 == 0.0 or (q0 > 0) == (signed > 0):
            total = q0 + signed
            b["avg"] = (a0 * q0 + float(f["price"]) * signed) / total if total else 0.0
            b["qty"] = total
            continue

        closed_qty = min(abs(signed), abs(q0))
        direction = 1.0 if q0 > 0 else -1.0
        realized = direction * closed_qty * (float(f["price"]) - a0)
        b["qty"] = q0 + signed
        if abs(b["qty"]) < 1e-9:
            b["qty"], b["avg"] = 0.0, 0.0
            ts = _as_dt(f["ts"])
            if ts is not None and ts >= since:
                out.append(
                    {
                        "symbol": sym,
                        "ts": ts,
                        "exit_price": round(float(f["price"]), 2),
                        "entry_price": round(a0, 2),
                        "qty": round(closed_qty, 4),
                        "realized_usd": round(realized, 2),
                        "realized_pct": round((float(f["price"]) / a0 - 1.0) * 100.0, 2)
                        if a0
                        else 0.0,
                        "opened_at": opened_at.get(sym),
                    }
                )
        elif q0 * b["qty"] < 0:
            b["avg"] = float(f["price"])
    return out


def _guard_reasons(state_dir: Path) -> dict[str, tuple[str, str]]:
    """symbol -> (reason, iso timestamp) from the guards' own state."""
    try:
        from trading.runtime.guards import _load, exit_reason, exit_stamp

        raw = _load(Path(state_dir)).get("exits") or {}
        out: dict[str, tuple[str, str]] = {}
        for sym, entry in raw.items():
            stamp = exit_stamp(entry)
            if stamp:
                out[sym] = (exit_reason(entry) or "guard exit", stamp)
        return out
    except Exception as e:
        logger.bind(component="agents").warning(f"exits: guard state unavailable ({e})")
        return {}


_REASON_TEXT = {
    "trailing_stop": "trailing stop — the guards sold it, not the desk",
    "take_profit": "take-profit target — the guards sold it, not the desk",
}


def recent_exits(
    state_dir: Path,
    data_dir: Path,
    *,
    days: int = 21,
    limit: int = 12,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Positions that went flat in the last ``days``, newest first.

    Never raises: this runs inside context assembly, and a desk that
    cannot convene because a parquet file is missing is worse than a desk
    that cannot see last week's exits.
    """
    now = now or datetime.now(tz=timezone.utc)
    since = now - timedelta(days=days)
    try:
        from trading.dashboard.live import fills_with_symbols

        fills = fills_with_symbols(Path(state_dir) / "orders.db")
    except Exception as e:
        logger.bind(component="agents").warning(f"exits: fills unavailable ({e})")
        return []

    try:
        closed = _closed_positions(fills, since)
    except Exception as e:
        logger.bind(component="agents").warning(f"exits: walk failed ({e})")
        return []

    guards = _guard_reasons(state_dir)
    px_cache: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for rec in sorted(closed, key=lambda r: r["ts"], reverse=True)[:limit]:
        sym, ts = rec["symbol"], rec["ts"]
        day = ts.strftime("%Y-%m-%d")

        reason = "rebalance or operator close"
        g = guards.get(sym)
        if g:
            try:
                # Same trading day is the match: the guard writes its stamp
                # when it decides, the fill lands minutes later.
                if abs((datetime.fromisoformat(g[1]) - ts).total_seconds()) <= 36 * 3600:
                    reason = _REASON_TEXT.get(g[0], g[0])
            except Exception:
                pass

        name_move = _day_move_pct(data_dir, sym, day, px_cache)
        spy_move = _day_move_pct(data_dir, "SPY", day, px_cache)
        held = None
        if isinstance(rec.get("opened_at"), datetime):
            held = max(0, (ts - rec["opened_at"]).days)

        out.append(
            {
                "symbol": sym,
                "exited": day,
                "days_held": held,
                "why_it_left": reason,
                "entry_price": rec["entry_price"],
                "exit_price": rec["exit_price"],
                "realized_pct": rec["realized_pct"],
                "realized_usd": rec["realized_usd"],
                "on_the_day": {
                    "name_move_pct": name_move,
                    "spy_move_pct": spy_move,
                    "excess_vs_spy_pp": (
                        round(name_move - spy_move, 2)
                        if name_move is not None and spy_move is not None
                        else None
                    ),
                    "reads_as": _classify(name_move, spy_move),
                },
            }
        )
    return out


def exits_note() -> str:
    """The prose that travels with the block, so it is read correctly."""
    return (
        "Positions that LEFT the book recently, with what the tape was doing "
        "that day. Read it before re-buying anything: a guard exit is not the "
        "desk's decision and the desk was never told about it, so a name here "
        "may still sit in the standing target list. "
        f"How 'reads as' is derived — {_RULE} "
        "A market-wide exit says little about the name and may be worth "
        "re-entering; a name-specific one is evidence about the thesis. "
        "Judge it; do not just restate the label."
    )
