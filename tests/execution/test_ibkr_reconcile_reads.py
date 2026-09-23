"""IBKR reconciliation reads: strict (no heal/restart), mapped, bounded."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading.execution.base import NotConnectedError
from trading.execution.ibkr import BrokerTimeoutError, IbkrBroker

T0 = datetime(2026, 9, 28, 13, 30, tzinfo=timezone.utc)


def _exec(ref: str, ts: datetime, qty: float = 5, exec_id: str = "e1") -> SimpleNamespace:
    return SimpleNamespace(
        execution=SimpleNamespace(
            orderRef=ref,
            orderId=1,
            time=ts,
            shares=qty,
            price=101.5,
            exchange="NYSE",
            execId=exec_id,
        ),
        commissionReport=SimpleNamespace(commission=1.0, currency="USD"),
    )


class _Ib:
    def __init__(self, *, executions=(), completed=(), hang: bool = False) -> None:
        self.executions, self.completed, self.hang = list(executions), list(completed), hang
        self.calls: list[tuple] = []

    def isConnected(self) -> bool:
        return True

    async def reqExecutionsAsync(self, *args):
        self.calls.append(("executions", args))
        if self.hang:
            await asyncio.get_running_loop().create_future()
        return self.executions

    async def reqCompletedOrdersAsync(self, api_only):
        self.calls.append(("completed", (api_only,)))
        return self.completed


def _broker(ib: _Ib, monkeypatch) -> IbkrBroker:
    broker = IbkrBroker(ib=ib)
    broker._connected = True
    broker._ensure_ib_loop_thread()

    def _forbidden(*_a, **_k):
        raise AssertionError("reconciliation reads must not use the self-healing path")

    monkeypatch.setattr(broker, "_ensure_connected", _forbidden)
    monkeypatch.setattr(broker, "_bounded", _forbidden)
    monkeypatch.setattr(broker, "_reconnect_session", _forbidden)
    monkeypatch.setattr(broker, "_trigger_gateway_restart", _forbidden)
    return broker


def test_executions_are_mapped_and_filtered_by_since(monkeypatch) -> None:
    ib = _Ib(executions=[_exec("a", T0), _exec("b", T0 - timedelta(days=3), exec_id="e2")])
    fills = _broker(ib, monkeypatch).get_executions(since=T0 - timedelta(days=1))
    assert [f.order_id for f in fills] == ["a"]
    f = fills[0]
    assert (f.quantity, f.price, f.exec_id, f.commission_currency) == (5.0, 101.5, "e1", "USD")


def test_completed_orders_map_status_perm_id_and_skip_foreign(monkeypatch) -> None:
    trades = [
        SimpleNamespace(
            order=SimpleNamespace(orderRef="cyc-1", permId=987),
            orderStatus=SimpleNamespace(status="Cancelled", filled=0),
        ),
        SimpleNamespace(
            order=SimpleNamespace(orderRef="", permId=5),  # placed in TWS by hand
            orderStatus=SimpleNamespace(status="Filled", filled=3),
        ),
    ]
    ib = _Ib(completed=trades)
    reports = _broker(ib, monkeypatch).get_completed_orders()
    assert len(reports) == 1
    r = reports[0]
    assert (r.client_order_id, r.broker_order_id, r.is_cancelled) == ("cyc-1", "987", True)
    assert ib.calls == [("completed", (False,))]


def test_hung_request_times_out_without_healing(monkeypatch) -> None:
    broker = _broker(_Ib(hang=True), monkeypatch)
    broker.RECONCILE_TIMEOUT_S = 0.1
    with pytest.raises(BrokerTimeoutError):
        broker.get_executions()


def test_disconnected_client_is_refused_not_reconnected(monkeypatch) -> None:
    ib = _Ib()
    broker = _broker(ib, monkeypatch)
    broker._connected = False
    with pytest.raises(NotConnectedError):
        broker.get_completed_orders()
    assert ib.calls == []


def test_completed_filled_quantity_comes_from_the_order(monkeypatch) -> None:
    """ib-async's completed-order status carries no fill size; the Order does."""
    trades = [
        SimpleNamespace(
            order=SimpleNamespace(orderRef="cyc-2", permId=11, filledQuantity=4.0),
            orderStatus=SimpleNamespace(status="Cancelled", filled=0.0),
        ),
        SimpleNamespace(
            order=SimpleNamespace(
                orderRef="cyc-3", permId=12, filledQuantity=1.7976931348623157e308
            ),
            orderStatus=SimpleNamespace(status="Cancelled", filled=0.0),
        ),
    ]
    reports = _broker(_Ib(completed=trades), monkeypatch).get_completed_orders()
    assert [r.filled_quantity for r in reports] == [4.0, 0.0]  # UNSET_DOUBLE ignored
