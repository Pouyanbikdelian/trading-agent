"""Overnight fills — the order that filled and was never marked filled.

Found via the copilot 2026-07-16, open until 2026-08-06. The cycle
reconciled with ``broker.get_fills(since=ts_start)`` — only fills that
arrived during the submitting cycle. An order placed after the close that
filled at the next open was invisible: by the time the next cycle ran,
its ``since`` was already past the fill. Those rows stayed ``submitted``
in orders.db permanently (all of 2026-06-10 and 2026-07-14 are still like
that on the live box).

Cosmetic on paper. On live it means the system's view of what it holds
silently diverges from the broker's — it can believe it holds nothing and
buy the same name again.

The fix reconciles from the oldest STILL-OPEN order instead, which is
self-healing and back-fills the stuck rows without a migration.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from trading.execution.store import OrderStore

NOW = datetime(2026, 8, 6, 21, 5, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> OrderStore:
    return OrderStore(tmp_path / "orders.db")


def _order(cid: str, *, created: datetime, symbol: str = "AAPL") -> Order:
    return Order(
        client_order_id=cid,
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY),
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=created,
    )


class TestOldestOpenCreatedAt:
    def test_none_when_nothing_is_open(self, store: OrderStore) -> None:
        store.save_order(_order("o1", created=NOW - timedelta(days=2)))
        store.update_status("o1", OrderStatus.FILLED)
        assert store.oldest_open_created_at() is None

    def test_finds_the_oldest_submitted_order(self, store: OrderStore) -> None:
        """The overnight order: submitted yesterday, still not filled."""
        old = NOW - timedelta(days=1)
        store.save_order(_order("o1", created=old))
        store.update_status("o1", OrderStatus.SUBMITTED)
        store.save_order(_order("o2", created=NOW))
        store.update_status("o2", OrderStatus.SUBMITTED)
        assert store.oldest_open_created_at() == old

    def test_partial_and_pending_count_as_open(self, store: OrderStore) -> None:
        """A partially filled order can still fill the rest."""
        for i, status in enumerate((OrderStatus.PENDING, OrderStatus.PARTIAL)):
            store.save_order(_order(f"o{i}", created=NOW - timedelta(days=i + 1)))
            store.update_status(f"o{i}", status)
        assert store.oldest_open_created_at() == NOW - timedelta(days=2)

    def test_terminal_statuses_do_not_hold_the_window_open(self, store: OrderStore) -> None:
        """Otherwise one old rejection would widen the reconciliation
        window forever."""
        for i, status in enumerate(
            (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
        ):
            store.save_order(_order(f"o{i}", created=NOW - timedelta(days=30)))
            store.update_status(f"o{i}", status)
        assert store.oldest_open_created_at() is None


class TestStaleOpenOrders:
    def test_lists_only_the_old_ones(self, store: OrderStore) -> None:
        store.save_order(_order("fresh", created=NOW - timedelta(hours=2)))
        store.update_status("fresh", OrderStatus.SUBMITTED)
        store.save_order(_order("stale", created=NOW - timedelta(days=9), symbol="MSFT"))
        store.update_status("stale", OrderStatus.SUBMITTED)

        stale = store.open_orders_older_than(NOW - timedelta(days=3))

        assert [o.client_order_id for o, _, _ in stale] == ["stale"]

    def test_filled_old_orders_are_not_stale(self, store: OrderStore) -> None:
        store.save_order(_order("done", created=NOW - timedelta(days=30)))
        store.update_status("done", OrderStatus.FILLED)
        assert store.open_orders_older_than(NOW - timedelta(days=3)) == []


class TestReconcileWindow:
    """``Cycle._reconcile_from`` — pure arithmetic over the store, tested
    without standing up a Cycle."""

    @staticmethod
    def _window(store: OrderStore, ts_start: datetime) -> datetime:
        """Run the REAL method against a stand-in carrying only what it
        touches — the store and the two class constants. Constructing a
        whole Cycle would need a broker, a risk manager and a price
        cache to test three lines of date arithmetic."""
        from trading.runner.cycle import Cycle

        holder = type(
            "H",
            (),
            {
                "order_store": store,
                "RECONCILE_LOOKBACK": Cycle.RECONCILE_LOOKBACK,
                "STALE_ORDER_AGE": Cycle.STALE_ORDER_AGE,
            },
        )()
        return Cycle._reconcile_from(holder, ts_start)  # type: ignore[arg-type]

    def test_no_open_orders_reconciles_from_this_cycle(self, store: OrderStore) -> None:
        assert self._window(store, NOW) == NOW

    def test_an_overnight_order_widens_the_window_to_cover_it(self, store: OrderStore) -> None:
        """The whole bug in one assertion: yesterday's order must be
        inside today's reconciliation window."""
        yesterday = NOW - timedelta(days=1)
        store.save_order(_order("o1", created=yesterday))
        store.update_status("o1", OrderStatus.SUBMITTED)
        assert self._window(store, NOW) == yesterday

    def test_the_window_is_floored_so_one_wedged_row_cannot_run_away(
        self, store: OrderStore
    ) -> None:
        from trading.runner.cycle import Cycle

        store.save_order(_order("ancient", created=NOW - timedelta(days=400)))
        store.update_status("ancient", OrderStatus.SUBMITTED)
        assert self._window(store, NOW) == NOW - Cycle.RECONCILE_LOOKBACK

    def test_never_later_than_the_cycle_start(self, store: OrderStore) -> None:
        """A clock skew that put an order in the future must not skip
        reconciliation entirely."""
        store.save_order(_order("future", created=NOW + timedelta(hours=3)))
        store.update_status("future", OrderStatus.SUBMITTED)
        assert self._window(store, NOW) == NOW

    def test_a_broken_store_degrades_to_the_old_behaviour(self) -> None:
        """Reconciliation must never be what kills a cycle."""
        from trading.runner.cycle import Cycle

        class Boom:
            def oldest_open_created_at(self):
                raise RuntimeError("db locked")

        holder = type(
            "H",
            (),
            {"order_store": Boom(), "RECONCILE_LOOKBACK": Cycle.RECONCILE_LOOKBACK},
        )()
        assert Cycle._reconcile_from(holder, NOW) == NOW  # type: ignore[arg-type]
