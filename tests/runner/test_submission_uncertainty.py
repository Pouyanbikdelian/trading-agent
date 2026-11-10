"""A failed acknowledgement is not evidence that IBKR rejected an order."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from trading.core.types import AssetClass, Fill, Instrument, Order, OrderStatus, Side
from trading.execution import OrderStore
from trading.execution.base import (
    BrokerError,
    OrderNotSentDisconnectedError,
    OrderNotSentError,
)
from trading.execution.ibkr import BrokerTimeoutError
from trading.risk.limits import RiskLimits
from trading.risk.manager import RiskManager
from trading.runner import NullAlerts
from trading.runner.cycle import Cycle


@pytest.mark.parametrize(
    "error",
    [
        BrokerTimeoutError("ack timeout"),
        ConnectionError("socket lost"),
        RuntimeError("unexpected acknowledgement failure"),
    ],
)
@pytest.mark.parametrize("defensive", [False, True])
def test_uncertain_submission_stops_batch_and_preserves_reconciliation(tmp_path, error, defensive):
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    orders = [
        Order(
            client_order_id=s,
            instrument=Instrument(symbol=s, asset_class=AssetClass.EQUITY),
            side=Side.BUY,
            quantity=1,
            created_at=now,
        )
        for s in ["AAPL", "MSFT"]
    ]
    attempted = []

    def submit(order):
        attempted.append(order)  # Simulate acceptance followed by lost acknowledgement.
        raise error

    store = OrderStore(tmp_path / "orders.db")
    risk = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
    cycle = SimpleNamespace(
        risk_manager=risk,
        order_store=store,
        alerts=NullAlerts(),
        broker=SimpleNamespace(
            submit_order=submit, tick=lambda *a: None, get_fills=lambda **kw: []
        ),
        _build_latest_bars=lambda *a: {},
        _reconcile_from=lambda ts: ts,
        _warn_on_stale_open_orders=lambda ts: None,
    )
    count, fills, halted, failed = Cycle._submit_and_reconcile(
        cycle, orders, pd.DataFrame(), now, allow_when_halted=defensive
    )
    assert attempted == orders[:1]
    assert count == 0 and failed and halted
    assert fills == []
    assert risk.is_halted()
    assert store.oldest_open_created_at() == now
    cycle.RECONCILE_LOOKBACK = timedelta(days=7)
    assert Cycle._reconcile_from(cycle, now + timedelta(days=1)) == now
    assert "outcome unknown" in risk.state.reason
    # A late fill can now settle the retained row instead of falling outside
    # the next reconciliation window when the row was falsely terminal.
    fill = Fill(order_id="AAPL", ts=now + timedelta(hours=1), quantity=1, price=100)
    store.save_fill(fill, client_order_id="AAPL")
    assert store.settle_status("AAPL") == OrderStatus.FILLED
    assert store.oldest_open_created_at() is None
    assert risk.is_halted()  # Reconciliation does not grant operator consent.


def test_explicit_broker_rejection_remains_terminal(tmp_path):
    now = datetime(2026, 11, 8, tzinfo=timezone.utc)
    order = Order(
        client_order_id="rejected",
        instrument=Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY),
        side=Side.BUY,
        quantity=1,
        created_at=now,
    )

    def reject(order):
        raise BrokerError("IBKR rejected: insufficient funds")

    store = OrderStore(tmp_path / "orders.db")
    risk = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
    cycle = SimpleNamespace(
        risk_manager=risk,
        order_store=store,
        alerts=NullAlerts(),
        broker=SimpleNamespace(
            submit_order=reject, tick=lambda *a: None, get_fills=lambda **kw: []
        ),
        _build_latest_bars=lambda *a: {},
        _reconcile_from=lambda ts: ts,
        _warn_on_stale_open_orders=lambda ts: None,
    )
    assert Cycle._submit_and_reconcile(cycle, [order], pd.DataFrame(), now)[3]
    assert store.oldest_open_created_at() is None
    assert not risk.is_halted()


def _two_orders(now):
    return [
        Order(
            client_order_id=s,
            instrument=Instrument(symbol=s, asset_class=AssetClass.EQUITY),
            side=Side.BUY,
            quantity=1,
            created_at=now,
        )
        for s in ["AAPL", "MSFT"]
    ]


def _cycle_with(submit, tmp_path, alerts):
    store = OrderStore(tmp_path / "orders.db")
    risk = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
    cycle = SimpleNamespace(
        risk_manager=risk,
        order_store=store,
        alerts=alerts,
        broker=SimpleNamespace(
            submit_order=submit, tick=lambda *a: None, get_fills=lambda **kw: []
        ),
        _build_latest_bars=lambda *a: {},
        _reconcile_from=lambda ts: ts,
        _warn_on_stale_open_orders=lambda ts: None,
    )
    return cycle, store, risk


class _Alerts(NullAlerts):
    def __init__(self):
        self.critical_msgs, self.error_msgs = [], []

    def critical(self, msg):
        self.critical_msgs.append(msg)

    def error(self, msg):
        self.error_msgs.append(msg)


def test_a_dead_gateway_before_the_send_neither_halts_nor_leaves_a_ghost_row(tmp_path):
    """2026-11-10. The gateway is down or waiting for 2FA, so the adapter
    fails before placeOrder. Nothing reached IBKR: there is no outcome to
    reconcile. The 8 November rule classified this by its "not connected"
    text as outcome-unknown — halting the desk, telling the operator IBKR
    "may have received" the order, and leaving a PENDING row that could
    not be retired for a week. The most common broker failure there is."""
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    attempted = []

    def submit(order):
        attempted.append(order.client_order_id)
        raise OrderNotSentDisconnectedError(
            "buy 1 AAPL was not sent to IBKR: IbkrBroker is not connected"
        )

    alerts = _Alerts()
    cycle, store, risk = _cycle_with(submit, tmp_path, alerts)
    count, _fills, halted, failed = Cycle._submit_and_reconcile(
        cycle, _two_orders(now), pd.DataFrame(), now
    )
    assert attempted == ["AAPL"]  # the rest of the batch would fail the same way
    assert count == 0 and failed
    assert not halted and not risk.is_halted()
    assert store.oldest_open_created_at() is None  # no ghost row
    assert len(alerts.critical_msgs) == 1
    assert "not sent" in alerts.critical_msgs[0]
    assert "may have received" not in alerts.critical_msgs[0]


def test_a_local_refusal_before_the_send_does_not_stop_the_batch(tmp_path):
    """One order the adapter cannot build says nothing about the next."""
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    attempted = []

    def submit(order):
        attempted.append(order.client_order_id)
        if order.client_order_id == "AAPL":
            raise OrderNotSentError("buy 1 AAPL was not sent to IBKR: unsupported asset class")
        return order

    alerts = _Alerts()
    cycle, _store, risk = _cycle_with(submit, tmp_path, alerts)
    count, _fills, halted, failed = Cycle._submit_and_reconcile(
        cycle, _two_orders(now), pd.DataFrame(), now
    )
    assert attempted == ["AAPL", "MSFT"]
    assert count == 1 and failed and not halted
    assert not risk.is_halted()
    assert not alerts.critical_msgs and len(alerts.error_msgs) == 1
