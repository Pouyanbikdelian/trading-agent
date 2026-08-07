"""One execution, one ledger row — however many times we reconcile.

``_reconcile_from`` widens the fill window back to the oldest STILL-OPEN
order, on purpose, so an order placed after the close and filled at the
next open is not missed. ``save_fill`` was a plain INSERT, so every cycle
re-stored every fill in that window.

With ``CRON=5 21 * * FRI`` against a 20:00 UTC close, an order queued to
the next open is the normal case. The duplicates were not inert: the
ledger feeds ``fills_with_symbols`` -> ``realized_by_symbol`` (the Live
tab's realized PnL and fees) and ``memory/episodes.py`` (the historian's
trade reconstruction). Both counted the same execution repeatedly.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from trading.core.types import (
    AssetClass,
    Fill,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)
from trading.execution.store import OrderStore

TS = datetime(2026, 8, 7, 20, 55, tzinfo=timezone.utc)


def _order(store: OrderStore, coid: str = "o-1") -> Order:
    order = Order(
        client_order_id=coid,
        instrument=Instrument(symbol="VST", asset_class=AssetClass.EQUITY),
        side=Side.SELL,
        quantity=63,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=TS,
    )
    store.save_order(order)
    return order


def _fill(**kw) -> Fill:
    base = {"order_id": "o-1", "ts": TS, "quantity": 63.0, "price": 139.12, "commission": 1.0}
    return Fill(**{**base, **kw})


def _count(store: OrderStore) -> int:
    return int(store.conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0])


def test_the_same_execution_reconciled_twice_is_stored_once(tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    _order(store)
    f = _fill(exec_id="0001f4e8.68a1b2c3.01.01")

    store.save_fill(f, client_order_id="o-1")
    store.save_fill(f, client_order_id="o-1")

    assert _count(store) == 1


def test_five_reconciliation_passes_still_yield_one_row(tmp_path):
    """An order left open all week is re-read on every cycle."""
    store = OrderStore(tmp_path / "orders.db")
    _order(store)
    f = _fill(exec_id="exec-abc")

    for _ in range(5):
        store.save_fill(f, client_order_id="o-1")

    assert _count(store) == 1


def test_distinct_executions_are_both_kept(tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    _order(store)

    store.save_fill(_fill(exec_id="exec-1", quantity=20.0), client_order_id="o-1")
    store.save_fill(_fill(exec_id="exec-2", quantity=43.0), client_order_id="o-1")

    assert _count(store) == 2


def test_a_broker_without_exec_ids_still_dedupes(tmp_path):
    """The Simulator supplies no execId; the composite key covers it."""
    store = OrderStore(tmp_path / "orders.db")
    _order(store)
    f = _fill()

    store.save_fill(f, client_order_id="o-1")
    store.save_fill(f, client_order_id="o-1")

    assert _count(store) == 1


def test_partial_fills_at_different_times_are_not_collapsed(tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    _order(store)

    store.save_fill(_fill(quantity=20.0), client_order_id="o-1")
    store.save_fill(_fill(quantity=20.0, ts=TS + timedelta(seconds=1)), client_order_id="o-1")

    assert _count(store) == 2


def test_same_size_different_price_is_a_distinct_fill(tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    _order(store)

    store.save_fill(_fill(quantity=20.0, price=139.12), client_order_id="o-1")
    store.save_fill(_fill(quantity=20.0, price=139.44), client_order_id="o-1")

    assert _count(store) == 2


def test_load_fills_round_trips_the_exec_id(tmp_path):
    store = OrderStore(tmp_path / "orders.db")
    _order(store)
    store.save_fill(_fill(exec_id="exec-xyz"), client_order_id="o-1")

    loaded = store.load_fills(client_order_id="o-1")

    assert len(loaded) == 1
    assert loaded[0].quantity == 63.0


class TestMigratingADeployedLedger:
    """The live orders.db already holds duplicates. The UNIQUE index cannot
    be created until they are repaired, so the migration removes them —
    the only destructive statement in the store, and deliberate."""

    def _legacy_db(self, path) -> None:
        """A pre-migration fills table with the same execution stored 3x."""
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE orders (
                client_order_id TEXT PRIMARY KEY, instrument_json TEXT NOT NULL,
                side TEXT NOT NULL, quantity REAL NOT NULL, order_type TEXT NOT NULL,
                limit_price REAL, stop_price REAL, tif TEXT NOT NULL,
                created_at REAL NOT NULL, status TEXT NOT NULL, broker_order_id TEXT);
            CREATE TABLE fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL, ts REAL NOT NULL, quantity REAL NOT NULL,
                price REAL NOT NULL, commission REAL NOT NULL DEFAULT 0, venue TEXT);
            """
        )
        for _ in range(3):
            conn.execute(
                "INSERT INTO fills (order_id, ts, quantity, price, commission, venue) "
                "VALUES ('o-1', 1000.0, 63.0, 139.12, 1.0, 'SMART')"
            )
        conn.execute(
            "INSERT INTO fills (order_id, ts, quantity, price, commission, venue) "
            "VALUES ('o-1', 2000.0, 10.0, 140.0, 1.0, 'SMART')"
        )
        conn.commit()
        conn.close()

    def test_duplicates_are_collapsed_to_one_row_each(self, tmp_path):
        db = tmp_path / "orders.db"
        self._legacy_db(db)

        store = OrderStore(db)

        rows = store.conn.execute("SELECT ts, quantity FROM fills ORDER BY ts").fetchall()
        assert [(r[0], r[1]) for r in rows] == [(1000.0, 63.0), (2000.0, 10.0)]

    def test_the_migration_is_idempotent(self, tmp_path):
        db = tmp_path / "orders.db"
        self._legacy_db(db)

        OrderStore(db).conn.close()
        store = OrderStore(db)

        assert _count(store) == 2

    def test_after_migration_new_duplicates_are_refused(self, tmp_path):
        db = tmp_path / "orders.db"
        self._legacy_db(db)
        store = OrderStore(db)

        f = Fill(
            order_id="o-1",
            ts=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            quantity=63.0,
            price=139.12,
            commission=1.0,
        )
        store.save_fill(f, client_order_id="o-1")

        assert _count(store) == 2


class TestPartialFills:
    """``OrderStatus.PARTIAL`` was defined, listed in ``OPEN_STATUSES``, and
    set by nothing. The cycle marked an order FILLED on its first fill
    event, and IBKR reports each execution separately.

    Not merely cosmetic: ``oldest_open_created_at`` drives the
    reconciliation window, so an order going terminal early stops holding
    it open — and the remainder of a partial fill could then arrive after
    the window had closed and never be recorded at all.
    """

    def test_a_partial_fill_leaves_the_order_open(self, tmp_path):
        from trading.core.types import OrderStatus

        store = OrderStore(tmp_path / "orders.db")
        _order(store)  # 63 shares

        store.save_fill(_fill(quantity=20.0, exec_id="e1"), client_order_id="o-1")
        status = store.settle_status("o-1")

        assert status == OrderStatus.PARTIAL
        assert store.oldest_open_created_at() is not None

    def test_the_completing_fill_closes_it(self, tmp_path):
        from trading.core.types import OrderStatus

        store = OrderStore(tmp_path / "orders.db")
        _order(store)

        store.save_fill(_fill(quantity=20.0, exec_id="e1"), client_order_id="o-1")
        store.settle_status("o-1")
        store.save_fill(_fill(quantity=43.0, exec_id="e2"), client_order_id="o-1")
        status = store.settle_status("o-1")

        assert status == OrderStatus.FILLED
        assert store.oldest_open_created_at() is None

    def test_a_single_full_fill_goes_straight_to_filled(self, tmp_path):
        from trading.core.types import OrderStatus

        store = OrderStore(tmp_path / "orders.db")
        _order(store)

        store.save_fill(_fill(quantity=63.0, exec_id="e1"), client_order_id="o-1")

        assert store.settle_status("o-1") == OrderStatus.FILLED

    def test_float_dust_does_not_leave_it_forever_partial(self, tmp_path):
        from trading.core.types import OrderStatus

        store = OrderStore(tmp_path / "orders.db")
        _order(store)

        store.save_fill(_fill(quantity=63.0 - 1e-12, exec_id="e1"), client_order_id="o-1")

        assert store.settle_status("o-1") == OrderStatus.FILLED

    def test_an_unknown_order_is_reported_not_guessed(self, tmp_path):
        store = OrderStore(tmp_path / "orders.db")
        assert store.settle_status("never-existed") is None
