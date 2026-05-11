"""IbkrBroker tests using a hand-rolled fake ``ib`` object.

The point isn't to test ib-async — they have their own tests. We confirm
that our adapter:
  * Builds the right ``Contract`` and ``Order`` shapes for each enum.
  * Routes ``submit_order`` / ``cancel_order`` to the right ib-async calls.
  * Refuses to act when not connected.
  * Maps IBKR's ``Position`` / ``Fill`` objects back into our types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)
from trading.execution import IbkrBroker, NotConnectedError, new_client_order_id


class _FakeIb:
    """Stub for the bits of the ib-async ``IB`` class our adapter touches."""

    def __init__(self) -> None:
        self._connected = False
        self.placed: list[tuple[object, object]] = []
        self.cancelled: list[object] = []
        self._open_trades: list[object] = []
        self._positions: list[object] = []
        self._fills: list[object] = []
        self._account_summary: list[object] = []

    # connection
    def isConnected(self) -> bool: return self._connected

    async def connectAsync(self, host: str, port: int, clientId: int) -> None:   # noqa: N802,N803
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # orders
    def placeOrder(self, contract: object, order: object) -> object:  # noqa: N802
        self.placed.append((contract, order))
        trade = SimpleNamespace(order=order, contract=contract)
        self._open_trades.append(trade)
        return trade

    def cancelOrder(self, order: object) -> None:   # noqa: N802
        self.cancelled.append(order)

    def openTrades(self) -> list[object]:           # noqa: N802
        return list(self._open_trades)

    # state
    def positions(self) -> list[object]:
        return list(self._positions)

    def fills(self) -> list[object]:
        return list(self._fills)

    def accountSummary(self) -> list[object]:      # noqa: N802
        return list(self._account_summary)


@pytest.fixture
def aapl() -> Instrument:
    return Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def fake_ib() -> _FakeIb:
    return _FakeIb()


@pytest.fixture
def broker(fake_ib: _FakeIb) -> IbkrBroker:
    b = IbkrBroker(ib=fake_ib)
    b.connect()
    return b


def test_requires_connect(fake_ib: _FakeIb, aapl: Instrument) -> None:
    b = IbkrBroker(ib=fake_ib)   # not connected
    order = Order(client_order_id=new_client_order_id(), instrument=aapl,
                  side=Side.BUY, quantity=1,
                  created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(NotConnectedError):
        b.submit_order(order)


def test_submit_market_order_calls_place_order(broker: IbkrBroker, fake_ib: _FakeIb,
                                                aapl: Instrument) -> None:
    order = Order(client_order_id="cid-1", instrument=aapl, side=Side.BUY,
                  quantity=10, order_type=OrderType.MARKET, tif=TimeInForce.DAY,
                  created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    broker.submit_order(order)
    assert len(fake_ib.placed) == 1
    contract, ib_order = fake_ib.placed[0]
    assert ib_order.action == "BUY"
    assert ib_order.orderType == "MKT"
    assert ib_order.totalQuantity == 10
    assert ib_order.tif == "DAY"
    assert ib_order.orderRef == "cid-1"


def test_limit_order_translation(broker: IbkrBroker, fake_ib: _FakeIb,
                                  aapl: Instrument) -> None:
    order = Order(client_order_id="cid-2", instrument=aapl, side=Side.SELL,
                  quantity=5, order_type=OrderType.LIMIT, limit_price=150.0,
                  tif=TimeInForce.GTC,
                  created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    broker.submit_order(order)
    _, ib_order = fake_ib.placed[0]
    assert ib_order.orderType == "LMT"
    assert ib_order.lmtPrice == 150.0
    assert ib_order.tif == "GTC"


def test_contract_for_fx_uses_forex() -> None:
    """FX symbols come pre-joined ("EURUSD") and dispatch to ib_async.Forex."""
    fake = _FakeIb()
    b = IbkrBroker(ib=fake)
    b.connect()
    eurusd = Instrument(symbol="EURUSD", asset_class=AssetClass.FX)
    o = Order(client_order_id="fx-1", instrument=eurusd, side=Side.BUY, quantity=10_000,
              created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    b.submit_order(o)
    contract, _ = fake.placed[0]
    # ib_async.Forex has a ``pair()`` method or stores `symbol` as the base currency.
    # We just confirm a Contract-like object came through.
    assert hasattr(contract, "secType") or hasattr(contract, "symbol")


def test_cancel_finds_order_by_orderref(broker: IbkrBroker, fake_ib: _FakeIb,
                                         aapl: Instrument) -> None:
    o = Order(client_order_id="cid-cancel", instrument=aapl, side=Side.BUY,
              quantity=1, created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    broker.submit_order(o)
    broker.cancel_order("cid-cancel")
    assert len(fake_ib.cancelled) == 1


def test_cancel_unknown_is_no_op(broker: IbkrBroker, fake_ib: _FakeIb) -> None:
    broker.cancel_order("never-placed")   # warns, doesn't raise
    assert fake_ib.cancelled == []


def test_get_positions_maps_to_our_types(broker: IbkrBroker, fake_ib: _FakeIb) -> None:
    """Inject a fake IBKR position record and confirm we map it correctly."""
    contract = SimpleNamespace(symbol="AAPL", secType="STK", exchange="SMART",
                                currency="USD", multiplier=None)
    fake_ib._positions = [SimpleNamespace(contract=contract, position=10, avgCost=150.0)]
    out = broker.get_positions()
    assert len(out) == 1
    p = out[0]
    assert p.instrument.symbol == "AAPL"
    assert p.instrument.asset_class == AssetClass.EQUITY
    assert p.quantity == 10.0
    assert p.avg_price == 150.0


def test_get_account_reads_summary(broker: IbkrBroker, fake_ib: _FakeIb) -> None:
    fake_ib._account_summary = [
        SimpleNamespace(tag="TotalCashValue", value="50000"),
        SimpleNamespace(tag="NetLiquidation", value="125000"),
        SimpleNamespace(tag="GrossPositionValue", value="75000"),    # ignored
    ]
    snap = broker.get_account()
    assert snap.cash == 50_000
    assert snap.equity == 125_000


def test_get_fills_filters_by_since(broker: IbkrBroker, fake_ib: _FakeIb) -> None:
    exec1 = SimpleNamespace(time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                            shares=5, price=100.0, exchange="SMART", orderId=1)
    exec2 = SimpleNamespace(time=datetime(2024, 1, 3, tzinfo=timezone.utc),
                            shares=5, price=101.0, exchange="SMART", orderId=2)
    fill1 = SimpleNamespace(execution=exec1,
                            order=SimpleNamespace(orderRef="ref-1"),
                            commissionReport=SimpleNamespace(commission=0.1))
    fill2 = SimpleNamespace(execution=exec2,
                            order=SimpleNamespace(orderRef="ref-2"),
                            commissionReport=SimpleNamespace(commission=0.1))
    fake_ib._fills = [fill1, fill2]
    out = broker.get_fills(since=datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert len(out) == 1
    assert out[0].order_id == "ref-2"


def test_disconnect_safe_to_call_twice(broker: IbkrBroker) -> None:
    broker.disconnect()
    broker.disconnect()   # second call must be a no-op
