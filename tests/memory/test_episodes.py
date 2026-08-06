"""Round-trip reconstruction — the missing half of the learning loop.

`MemoryStore.add_episode` was written, indexed and documented, and never
called from anywhere in `src/`. The `episodes` table — the place a lesson
is supposed to meet realised P&L — stayed empty, so the historian filled
the evidence slot with a synthetic week tag and lesson promotion ran on
an LLM voting on its own book.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.memory.episodes import _round_trips


def _fill(ts: float, symbol: str, side: str, qty: float, px: float) -> dict:
    # Key names match what dashboard.live.fills_with_symbols actually
    # emits — ``qty``, not ``quantity``. The first cut of this module read
    # ``quantity`` and therefore treated every real fill as zero-sized,
    # which the unit tests happily passed because they used the same
    # wrong key. Fixtures that agree with the code instead of the source
    # prove nothing.
    return {"ts": ts, "symbol": symbol, "side": side, "qty": qty, "price": px}


DAY = 86_400.0


class TestRoundTrips:
    def test_simple_buy_then_sell(self) -> None:
        eps = _round_trips(
            [_fill(0, "AAPL", "BUY", 10, 100.0), _fill(DAY, "AAPL", "SELL", 10, 110.0)]
        )
        assert len(eps) == 1
        e = eps[0]
        assert e["symbol"] == "AAPL"
        assert e["entry_px"] == 100.0 and e["exit_px"] == 110.0
        assert e["pnl_pct"] == pytest.approx(0.10)
        assert e["ts_open"] == datetime.fromtimestamp(0, tz=timezone.utc)

    def test_scaling_in_and_out_is_one_episode(self) -> None:
        """A position built over three weeks and sold in two tranches is
        ONE decision, not four. Splitting it would let a single thesis
        vote four times on the same lesson."""
        eps = _round_trips(
            [
                _fill(0, "MSFT", "BUY", 10, 100.0),
                _fill(DAY, "MSFT", "BUY", 10, 120.0),
                _fill(2 * DAY, "MSFT", "SELL", 5, 130.0),
                _fill(3 * DAY, "MSFT", "SELL", 15, 140.0),
            ]
        )
        assert len(eps) == 1
        assert eps[0]["entry_px"] == pytest.approx(110.0)  # weighted
        assert eps[0]["exit_px"] == pytest.approx((5 * 130 + 15 * 140) / 20)
        assert eps[0]["qty"] == 20

    def test_still_open_position_is_not_an_episode(self) -> None:
        """An episode requires an exit — an open position has no outcome
        to learn from yet."""
        assert _round_trips([_fill(0, "NVDA", "BUY", 10, 100.0)]) == []
        assert (
            _round_trips([_fill(0, "NVDA", "BUY", 10, 100.0), _fill(DAY, "NVDA", "SELL", 4, 110.0)])
            == []
        )

    def test_two_separate_round_trips_in_the_same_name(self) -> None:
        eps = _round_trips(
            [
                _fill(0, "GLD", "BUY", 10, 100.0),
                _fill(DAY, "GLD", "SELL", 10, 110.0),
                _fill(5 * DAY, "GLD", "BUY", 10, 120.0),
                _fill(6 * DAY, "GLD", "SELL", 10, 90.0),
            ]
        )
        assert len(eps) == 2
        assert [round(e["pnl_pct"], 4) for e in eps] == [0.10, -0.25]

    def test_sell_with_nothing_open_is_ignored(self) -> None:
        """Long-only by risk config; a stray sell must not invent a short
        episode with nonsense accounting."""
        assert _round_trips([_fill(0, "TSLA", "SELL", 10, 100.0)]) == []

    def test_zero_and_negative_prices_are_skipped(self) -> None:
        eps = _round_trips(
            [
                _fill(0, "X", "BUY", 10, 0.0),
                _fill(DAY, "X", "BUY", 10, 100.0),
                _fill(2 * DAY, "X", "SELL", 10, 110.0),
            ]
        )
        assert len(eps) == 1 and eps[0]["entry_px"] == 100.0

    def test_output_is_ordered_by_close(self) -> None:
        eps = _round_trips(
            [
                _fill(0, "A", "BUY", 1, 10.0),
                _fill(9 * DAY, "A", "SELL", 1, 11.0),
                _fill(DAY, "B", "BUY", 1, 10.0),
                _fill(2 * DAY, "B", "SELL", 1, 11.0),
            ]
        )
        assert [e["symbol"] for e in eps] == ["B", "A"]


def test_recording_is_idempotent(tmp_path) -> None:
    """Runs nightly; a crash mid-pass must not double-count an episode."""
    import json
    import sqlite3

    from trading.memory.episodes import record_closed_episodes
    from trading.memory.store import MemoryStore

    db = tmp_path / "orders.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE orders (client_order_id TEXT PRIMARY KEY, side TEXT, instrument_json TEXT);
           CREATE TABLE fills (ts REAL, order_id TEXT, quantity REAL, price REAL, commission REAL);"""
    )
    inst = json.dumps({"symbol": "AAPL"})
    conn.execute("INSERT INTO orders VALUES ('o1','BUY',?)", (inst,))
    conn.execute("INSERT INTO orders VALUES ('o2','SELL',?)", (inst,))
    conn.execute("INSERT INTO fills VALUES (0,'o1',10,100.0,0)")
    conn.execute("INSERT INTO fills VALUES (86400,'o2',10,110.0,0)")
    conn.commit()
    conn.close()

    mem = MemoryStore(tmp_path / "memory")
    assert record_closed_episodes(mem, tmp_path, tmp_path / "data") == 1
    assert record_closed_episodes(mem, tmp_path, tmp_path / "data") == 0
    assert len(mem.episodes_for()) == 1
