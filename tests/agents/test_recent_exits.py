"""The desk must be able to see what left the book, and why.

Until this existed the PM decided from the present state alone: it could
not see whether the last trade in a name made or lost money, why the name
left, or that the guards had sold anything at all. ``exits_done`` is
private to ``runtime.guards`` and gates re-EXIT, not re-ENTRY — so a
standing decision would happily re-buy a name stopped out that morning.

Facts and judgement are kept apart on purpose: the numbers are computed,
and the rule for reading them travels WITH them so an agent can disagree
with the label instead of inheriting a verdict it cannot inspect.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.agents.exits import _classify, _closed_positions, exits_note, recent_exits

NOW = datetime(2026, 8, 11, 20, 0, tzinfo=timezone.utc)


def _fill(sym, side, qty, price, ts):
    return {"symbol": sym, "side": side, "qty": qty, "price": price, "ts": ts, "commission": 0.0}


class TestFindingClosedPositions:
    def test_a_round_trip_is_reported_once(self) -> None:
        fills = [
            _fill("VST", "BUY", 60, 100.0, NOW - timedelta(days=30)),
            _fill("VST", "SELL", 60, 90.0, NOW - timedelta(days=1)),
        ]

        out = _closed_positions(fills, NOW - timedelta(days=21))

        assert len(out) == 1
        assert out[0]["symbol"] == "VST"
        assert out[0]["entry_price"] == 100.0
        assert out[0]["realized_usd"] == -600.0
        assert out[0]["realized_pct"] == -10.0

    def test_an_open_position_is_not_an_exit(self) -> None:
        fills = [
            _fill("MU", "BUY", 100, 50.0, NOW - timedelta(days=10)),
            _fill("MU", "SELL", 40, 60.0, NOW - timedelta(days=1)),
        ]

        assert _closed_positions(fills, NOW - timedelta(days=21)) == []

    def test_cost_basis_walks_the_whole_ledger_not_just_the_window(self) -> None:
        """A position opened months ago has no basis if the walk starts at
        the window edge — it would look like a naked short."""
        fills = [
            _fill("GS", "BUY", 10, 400.0, NOW - timedelta(days=300)),
            _fill("GS", "SELL", 10, 500.0, NOW - timedelta(days=2)),
        ]

        out = _closed_positions(fills, NOW - timedelta(days=21))

        assert out[0]["entry_price"] == 400.0
        assert out[0]["realized_usd"] == 1000.0

    def test_scaling_in_averages_the_basis(self) -> None:
        fills = [
            _fill("X", "BUY", 10, 100.0, NOW - timedelta(days=9)),
            _fill("X", "BUY", 10, 200.0, NOW - timedelta(days=8)),
            _fill("X", "SELL", 20, 150.0, NOW - timedelta(days=1)),
        ]

        out = _closed_positions(fills, NOW - timedelta(days=21))

        assert out[0]["entry_price"] == 150.0
        assert out[0]["realized_usd"] == 0.0

    def test_an_old_exit_falls_outside_the_window(self) -> None:
        fills = [
            _fill("OLD", "BUY", 10, 10.0, NOW - timedelta(days=300)),
            _fill("OLD", "SELL", 10, 12.0, NOW - timedelta(days=90)),
        ]

        assert _closed_positions(fills, NOW - timedelta(days=21)) == []

    def test_a_reopened_name_reports_each_round_trip(self) -> None:
        fills = [
            _fill("R", "BUY", 10, 100.0, NOW - timedelta(days=20)),
            _fill("R", "SELL", 10, 110.0, NOW - timedelta(days=15)),
            _fill("R", "BUY", 10, 120.0, NOW - timedelta(days=10)),
            _fill("R", "SELL", 10, 90.0, NOW - timedelta(days=2)),
        ]

        out = _closed_positions(fills, NOW - timedelta(days=21))

        assert [r["realized_usd"] for r in out] == [100.0, -300.0]


class TestReadingTheTape:
    """The operator's question: was it a market-wide selloff or the name?"""

    def test_a_broad_selloff_reads_as_market_wide(self) -> None:
        assert _classify(-5.0, -4.5) == "market-wide"

    def test_a_name_breaking_on_a_calm_tape_reads_as_specific(self) -> None:
        assert _classify(-9.0, -0.2) == "name-specific"

    def test_a_name_falling_much_harder_than_the_market_is_specific(self) -> None:
        """Down on a down day is not an excuse if it fell three times as far."""
        assert _classify(-15.0, -3.0) == "name-specific"

    def test_a_small_drop_on_a_flat_tape_is_not_market_wide(self) -> None:
        assert _classify(-2.0, -0.1) == "mixed"

    def test_missing_history_is_admitted_not_guessed(self) -> None:
        assert "unknown" in _classify(None, -3.0)
        assert "unknown" in _classify(-3.0, None)


class TestTheNoteTravelsWithTheNumbers:
    def test_the_rule_is_stated_in_the_note(self) -> None:
        """A hard label with no visible rule is how a heuristic becomes
        folklore. The agent must be able to argue with it."""
        note = exits_note()

        assert "market-wide" in note and "name-specific" in note
        assert "rule of thumb" in note or "do not just restate" in note

    def test_it_warns_that_a_guard_exit_was_never_told_to_the_desk(self) -> None:
        assert "never told" in exits_note()


class TestEndToEnd:
    def _db(self, path: Path, *, sell_ts: datetime) -> None:
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE orders (
                client_order_id TEXT PRIMARY KEY, instrument_json TEXT NOT NULL,
                side TEXT NOT NULL, quantity REAL NOT NULL, order_type TEXT NOT NULL,
                limit_price REAL, stop_price REAL, tif TEXT NOT NULL,
                created_at REAL NOT NULL, status TEXT NOT NULL, broker_order_id TEXT);
            CREATE TABLE fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL,
                ts REAL NOT NULL, quantity REAL NOT NULL, price REAL NOT NULL,
                commission REAL NOT NULL DEFAULT 0, venue TEXT);
            """
        )
        inst = '{"symbol": "VST", "asset_class": "equity"}'
        for coid, side, qty, price, ts in (
            ("b1", "BUY", 60, 151.5, NOW - timedelta(days=30)),
            ("s1", "SELL", 60, 139.12, sell_ts),
        ):
            conn.execute(
                "INSERT INTO orders VALUES (?,?,?,?,'MARKET',NULL,NULL,'DAY',?,'FILLED',NULL)",
                (coid, inst, side, qty, ts.timestamp()),
            )
            conn.execute(
                "INSERT INTO fills (order_id, ts, quantity, price, commission, venue) "
                "VALUES (?,?,?,?,0,'SMART')",
                (coid, ts.timestamp(), qty, price),
            )
        conn.commit()
        conn.close()

    def test_a_guard_stop_is_attributed_to_the_guards(self, tmp_path) -> None:
        """The point of the whole block: the desk learns it did not do this."""
        import json

        sell = NOW - timedelta(days=1)
        self._db(tmp_path / "orders.db", sell_ts=sell)
        (tmp_path / "guards.json").write_text(
            json.dumps({"exits": {"VST": {"at": sell.isoformat(), "reason": "trailing_stop"}}})
        )

        out = recent_exits(tmp_path, tmp_path, now=NOW)

        assert len(out) == 1
        assert "trailing stop" in out[0]["why_it_left"]
        assert "guards sold it" in out[0]["why_it_left"]

    def test_without_a_guard_record_it_is_not_blamed_on_the_guards(self, tmp_path) -> None:
        self._db(tmp_path / "orders.db", sell_ts=NOW - timedelta(days=1))

        out = recent_exits(tmp_path, tmp_path, now=NOW)

        assert out[0]["why_it_left"] == "rebalance or operator close"

    def test_the_realized_numbers_reach_the_desk(self, tmp_path) -> None:
        self._db(tmp_path / "orders.db", sell_ts=NOW - timedelta(days=1))

        rec = recent_exits(tmp_path, tmp_path, now=NOW)[0]

        assert rec["entry_price"] == 151.5
        assert rec["exit_price"] == 139.12
        assert rec["realized_pct"] == pytest.approx(-8.17, abs=0.01)
        assert rec["days_held"] == 29

    def test_missing_price_history_does_not_break_it(self, tmp_path) -> None:
        """No parquet cache in tmp_path — the block must still be produced."""
        self._db(tmp_path / "orders.db", sell_ts=NOW - timedelta(days=1))

        rec = recent_exits(tmp_path, tmp_path, now=NOW)[0]

        assert rec["on_the_day"]["name_move_pct"] is None
        assert "unknown" in rec["on_the_day"]["reads_as"]

    def test_a_missing_database_returns_empty_not_an_exception(self, tmp_path) -> None:
        assert recent_exits(tmp_path / "nope", tmp_path, now=NOW) == []


class TestWiring:
    def test_the_pm_is_instructed_to_use_it(self) -> None:
        """A context key no prompt mentions is a key the model skims past."""
        src = Path("src/trading/agents/pm.py").read_text()

        assert "RECENT EXITS RULE" in src
        assert "excess_vs_spy_pp" in src

    def test_context_assembly_includes_it(self) -> None:
        src = Path("src/trading/agents/context.py").read_text()

        assert 'ctx["recent_exits"]' in src
        assert "recent_exits_note" in src
