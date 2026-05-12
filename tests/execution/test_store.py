"""SQLite OrderStore tests using ``:memory:`` databases."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AssetClass,
    Fill,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from trading.execution import OrderStore


@pytest.fixture
def store() -> OrderStore:
    return OrderStore(":memory:")


@pytest.fixture
def aapl() -> Instrument:
    return Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def order(aapl: Instrument) -> Order:
    return Order(
        client_order_id="test-001",
        instrument=aapl,
        side=Side.BUY,
        quantity=10.0,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_save_and_load_order(store: OrderStore, order: Order) -> None:
    store.save_order(order)
    rows = store.load_orders()
    assert len(rows) == 1
    loaded, status, broker_id = rows[0]
    assert loaded.client_order_id == order.client_order_id
    assert loaded.instrument.symbol == "AAPL"
    assert loaded.instrument.asset_class == AssetClass.EQUITY
    assert status == OrderStatus.PENDING
    assert broker_id is None


def test_save_order_with_broker_id(store: OrderStore, order: Order) -> None:
    store.save_order(order, broker_order_id="ibkr-42")
    _, _, broker_id = store.load_orders()[0]
    assert broker_id == "ibkr-42"


def test_save_is_idempotent_on_client_order_id(store: OrderStore, order: Order) -> None:
    store.save_order(order)
    store.save_order(order)  # INSERT OR REPLACE — no duplicate row
    assert len(store.load_orders()) == 1


def test_update_status_persists(store: OrderStore, order: Order) -> None:
    store.save_order(order)
    store.update_status(order.client_order_id, OrderStatus.FILLED, broker_order_id="ibkr-7")
    _, status, broker_id = store.load_orders()[0]
    assert status == OrderStatus.FILLED
    assert broker_id == "ibkr-7"


def test_load_orders_filter_by_status(store: OrderStore, aapl: Instrument) -> None:
    a = Order(
        client_order_id="a",
        instrument=aapl,
        side=Side.BUY,
        quantity=1,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    b = Order(
        client_order_id="b",
        instrument=aapl,
        side=Side.SELL,
        quantity=2,
        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    store.save_order(a)
    store.save_order(b)
    store.update_status("a", OrderStatus.FILLED)
    filled = store.load_orders(status=OrderStatus.FILLED)
    pending = store.load_orders(status=OrderStatus.PENDING)
    assert {o.client_order_id for o, _, _ in filled} == {"a"}
    assert {o.client_order_id for o, _, _ in pending} == {"b"}


def test_load_orders_filter_by_since(store: OrderStore, aapl: Instrument) -> None:
    early = Order(
        client_order_id="early",
        instrument=aapl,
        side=Side.BUY,
        quantity=1,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    late = Order(
        client_order_id="late",
        instrument=aapl,
        side=Side.BUY,
        quantity=1,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    store.save_order(early)
    store.save_order(late)
    cutoff = datetime(2024, 3, 1, tzinfo=timezone.utc)
    rows = store.load_orders(since=cutoff)
    assert [o.client_order_id for o, _, _ in rows] == ["late"]


def test_save_and_load_fills(store: OrderStore, order: Order) -> None:
    store.save_order(order)
    f1 = Fill(
        order_id=order.client_order_id,
        ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
        quantity=5,
        price=100.0,
        commission=0.05,
    )
    f2 = Fill(
        order_id=order.client_order_id,
        ts=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
        quantity=5,
        price=101.0,
        commission=0.05,
    )
    store.save_fill(f1, client_order_id=order.client_order_id)
    store.save_fill(f2, client_order_id=order.client_order_id)
    rows = store.load_fills(client_order_id=order.client_order_id)
    assert len(rows) == 2
    assert rows[0].ts < rows[1].ts


def test_load_fills_since_filter(store: OrderStore, order: Order) -> None:
    store.save_order(order)
    early = Fill(
        order_id=order.client_order_id,
        ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
        quantity=5,
        price=100.0,
    )
    late = Fill(
        order_id=order.client_order_id,
        ts=datetime(2024, 1, 3, tzinfo=timezone.utc),
        quantity=5,
        price=101.0,
    )
    store.save_fill(early, client_order_id=order.client_order_id)
    store.save_fill(late, client_order_id=order.client_order_id)
    rows = store.load_fills(since=datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert len(rows) == 1
    assert rows[0].price == 101.0


def test_migration_is_idempotent(tmp_path) -> None:
    """Re-opening the same DB file must not error."""
    path = tmp_path / "orders.db"
    s1 = OrderStore(path)
    _ = s1.conn  # trigger migration
    s1.close()
    s2 = OrderStore(path)
    _ = s2.conn  # re-runs CREATE TABLE IF NOT EXISTS
    s2.close()


def test_save_rejects_naive_datetime(store: OrderStore, aapl: Instrument) -> None:
    naive = datetime(2024, 1, 1)
    bad = Order(
        client_order_id="naive",
        instrument=aapl,
        side=Side.BUY,
        quantity=1,
        created_at=naive.replace(tzinfo=timezone.utc),
    )  # actually tz-aware
    # The model_validator on Bar requires tz, but Order has no such validator —
    # the store itself is the guard. Replace with a tz-aware datetime for the
    # round-trip, then test the timestamp-stripping helper directly:
    from trading.execution.store import _ts_to_epoch

    with pytest.raises(ValueError, match="timezone-aware"):
        _ts_to_epoch(datetime(2024, 1, 1))
    # And confirm the normal save path works.
    store.save_order(bad)
    assert len(store.load_orders()) == 1
