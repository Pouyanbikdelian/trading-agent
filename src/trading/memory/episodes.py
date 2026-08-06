"""Closed round-trips → the ``episodes`` table.

The missing half of the learning loop, found 2026-08-06. The memory
spine's stated design rule is "Everything gradeable… Skill is a number
attached to memory, not a vibe", and the `episodes` table is where a
lesson is supposed to meet reality: a lesson accumulates supporting or
contradicting *episodes* and is promoted to `established` at +3 net.

`MemoryStore.add_episode` was written, indexed and documented — and
never called from anywhere in `src/`. So the table stayed empty, and the
historian filled the evidence slot with a synthetic ``week_tag``
instead. Lesson promotion therefore ran on **an LLM voting weekly on its
own lesson book**, with no contact with realised P&L at all.

This module derives episodes from the fill ledger, which is the only
place a completed round-trip actually exists: `runner.db` snapshots show
what is held now, and a position that closed has already vanished from
them. Walking fills in time order and marking the moment a symbol's net
quantity returns to zero reconstructs entry, exit and P&L exactly.

Idempotent by construction — re-running never duplicates an episode, so
it is safe on the nightly grader's schedule where a crash mid-pass would
otherwise double-count.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def _round_trips(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Completed long round-trips, oldest first.

    A leg opens when net quantity leaves zero and closes when it returns.
    Partial fills fold into weighted averages; a position scaled into and
    out of over three weeks is ONE episode, because that is what the
    desk actually decided — not six.

    Shorts are ignored for now: the system is long-only by risk config,
    and inventing accounting for a side we do not trade would be untested
    code on a path that matters.
    """
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for f in fills:
        by_symbol.setdefault(str(f.get("symbol", "?")).upper(), []).append(f)

    episodes: list[dict[str, Any]] = []
    for symbol, rows in by_symbol.items():
        qty = 0.0
        cost = 0.0  # running cost basis of the open leg
        opened_ts: float | None = None
        exit_value = 0.0
        exit_qty = 0.0
        for f in sorted(rows, key=lambda r: float(r["ts"])):
            side = str(f.get("side", "")).upper()
            # ``fills_with_symbols`` emits ``qty``; accept ``quantity`` too
            # so a caller handing raw ledger rows is not silently ignored.
            n = abs(float(f.get("qty", f.get("quantity")) or 0.0))
            px = float(f.get("price") or 0.0)
            if n <= 0 or px <= 0:
                continue
            if side.startswith("B"):
                if qty <= 0:
                    opened_ts, cost, exit_value, exit_qty = float(f["ts"]), 0.0, 0.0, 0.0
                qty += n
                cost += n * px
            elif side.startswith("S"):
                if qty <= 0:
                    continue  # a sell with nothing open: not our episode
                closed = min(n, qty)
                exit_value += closed * px
                exit_qty += closed
                qty -= closed
                if qty <= 1e-9 and exit_qty > 0 and opened_ts is not None:
                    entry_px = cost / (exit_qty or 1.0)
                    exit_px = exit_value / exit_qty
                    episodes.append(
                        {
                            "symbol": symbol,
                            "ts_open": datetime.fromtimestamp(opened_ts, tz=timezone.utc),
                            "ts_close": datetime.fromtimestamp(float(f["ts"]), tz=timezone.utc),
                            "entry_px": round(entry_px, 6),
                            "exit_px": round(exit_px, 6),
                            "pnl_pct": round(exit_px / entry_px - 1.0, 6) if entry_px else None,
                            "qty": round(exit_qty, 6),
                        }
                    )
                    qty = cost = exit_value = exit_qty = 0.0
                    opened_ts = None
    return sorted(episodes, key=lambda e: e["ts_close"])


def _entry_pctile_52w(data_dir: Path, symbol: str, when: datetime) -> float | None:
    """Where the entry sat in its own 52-week range, 0=low 1=high.

    This is the field that makes an episode teachable rather than just
    accounted: the quant charter's first hard rule is that buying at the
    100th percentile is maximally far from any trend stop, and this is
    how a lesson could ever be evidenced against that claim.
    """
    try:
        from trading.runtime.portfolio_stats import _read_close, close_at

        s = _read_close(data_dir, symbol)
        if s is None or len(s) < 60:
            return None
        px = close_at(s, when)
        if px is None:
            return None
        # The 52 weeks BEFORE the entry, not the whole series — using
        # bars from after the trade would score the decision with
        # information the decision did not have.
        import pandas as pd

        ts = pd.Timestamp(when)
        tz = getattr(s.index, "tz", None)
        ts = ts.tz_localize(tz) if (tz is not None and ts.tzinfo is None) else ts
        yr = s[s.index <= ts].iloc[-252:]
        if len(yr) < 60:
            return None
        lo, hi = float(yr.min()), float(yr.max())
        return round((px - lo) / (hi - lo), 3) if hi > lo else None
    except Exception:
        return None


def record_closed_episodes(mem: Any, state_dir: Path, data_dir: Path) -> int:
    """Write any completed round-trip not already recorded. Returns the count."""
    try:
        from trading.dashboard.live import fills_with_symbols

        fills = fills_with_symbols(Path(state_dir) / "orders.db")
    except Exception:
        logger.bind(component="memory").exception("episode recording: fills unavailable")
        return 0
    if not fills:
        return 0

    # Dedupe on (symbol, close timestamp to the second). Cheap, and a
    # symbol cannot close two distinct round-trips in the same second.
    try:
        seen = {
            (str(r["symbol"]).upper(), int(float(r["ts_close"])))
            for r in mem.episodes_for(limit=5000)
        }
    except Exception:
        seen = set()

    written = 0
    for ep in _round_trips(fills):
        key = (ep["symbol"], int(ep["ts_close"].timestamp()))
        if key in seen:
            continue
        try:
            mem.add_episode(
                symbol=ep["symbol"],
                ts_open=ep["ts_open"],
                ts_close=ep["ts_close"],
                entry_px=ep["entry_px"],
                exit_px=ep["exit_px"],
                pnl_pct=ep["pnl_pct"],
                entry_pctile_52w=_entry_pctile_52w(data_dir, ep["symbol"], ep["ts_open"]),
                context={"qty": ep["qty"], "source": "fill_ledger"},
                tags="round_trip",
            )
            written += 1
        except Exception:
            logger.bind(component="memory", symbol=ep["symbol"]).exception("add_episode failed")
    if written:
        logger.bind(component="memory").info(f"recorded {written} closed episode(s)")
    return written
