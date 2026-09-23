"""Daily broker reconciliation: fills, broker cancels, permIds, honest gaps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.core.types import (
    AssetClass,
    BrokerOrderReport,
    Fill,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from trading.execution.store import OrderStore
from trading.runtime.broker_reconcile import reconcile_with_broker

NOW = datetime(2026, 9, 28, 20, 30, tzinfo=timezone.utc)


def _order(coid: str, sym: str = "NVDA", qty: float = 10, days_ago: float = 3) -> Order:
    return Order(
        client_order_id=coid,
        instrument=Instrument(symbol=sym, asset_class=AssetClass.EQUITY),
        side=Side.BUY,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=NOW - timedelta(days=days_ago),
    )


def _fill(coid: str, qty: float, exec_id: str, days_ago: float = 2) -> Fill:
    return Fill(
        order_id=coid,
        ts=NOW - timedelta(days=days_ago),
        quantity=qty,
        price=100.0,
        commission=1.0,
        commission_currency="USD",
        exec_id=exec_id,
    )


class _Broker:
    def __init__(self, *, executions=(), completed=None, working=()) -> None:
        self.executions = list(executions)
        self.completed = completed
        self.working = list(working)
        self.submitted: list[object] = []
        self.since: datetime | None = None

    def get_executions(self, *, since=None):
        self.since = since
        return list(self.executions)

    def get_open_orders(self):
        return list(self.working)

    def submit_order(self, order):  # pragma: no cover - must never be called
        self.submitted.append(order)
        raise AssertionError("reconciliation must never submit")

    def cancel_order(self, coid):  # pragma: no cover - must never be called
        raise AssertionError("reconciliation must never cancel")


class _CompletedBroker(_Broker):
    def get_completed_orders(self):
        return list(self.completed or [])


@pytest.fixture
def store(tmp_path: Path) -> OrderStore:
    s = OrderStore(tmp_path / "orders.db")
    for coid in ("a", "b", "c", "d"):
        s.save_order(_order(coid))
        s.update_status(coid, OrderStatus.SUBMITTED)
    return s


def _status(store: OrderStore, coid: str) -> OrderStatus:
    return {o.client_order_id: st for o, st, _ in store.load_orders()}[coid]


def test_overnight_fill_is_recorded_and_settled(store: OrderStore) -> None:
    broker = _CompletedBroker(
        executions=[_fill("a", 10, "e1"), _fill("zz-manual", 5, "e9")],
        completed=[BrokerOrderReport(client_order_id="a", broker_order_id="777", status="Filled")],
        working=[_order("b"), _order("c"), _order("d")],
    )

    report = reconcile_with_broker(store, broker, now=NOW)

    assert _status(store, "a") == OrderStatus.FILLED
    assert report.fills_recorded == 1 and report.unmatched_executions == 1
    assert report.settled == {"a": "filled"}
    assert {o.client_order_id: b for o, _s, b in store.load_orders()}["a"] == "777"
    assert broker.submitted == []


def test_rerunning_is_idempotent(store: OrderStore) -> None:
    broker = _CompletedBroker(executions=[_fill("a", 4, "e1")], completed=[], working=[])
    reconcile_with_broker(store, broker, now=NOW)
    reconcile_with_broker(store, broker, now=NOW)
    assert len(store.load_fills(client_order_id="a")) == 1
    assert _status(store, "a") == OrderStatus.PARTIAL


def test_broker_cancel_is_recorded_as_cancelled(store: OrderStore) -> None:
    broker = _CompletedBroker(
        completed=[
            BrokerOrderReport(client_order_id="b", broker_order_id="42", status="Cancelled"),
            BrokerOrderReport(client_order_id="c", status="Inactive"),
        ],
        working=[_order("a"), _order("d")],
    )
    report = reconcile_with_broker(store, broker, now=NOW)
    assert _status(store, "b") == OrderStatus.CANCELLED
    assert _status(store, "c") == OrderStatus.CANCELLED
    assert sorted(report.cancelled) == ["b", "c"]


def test_partial_then_cancelled_keeps_its_fills_and_goes_terminal(store: OrderStore) -> None:
    broker = _CompletedBroker(
        executions=[_fill("a", 3, "e1")],
        completed=[BrokerOrderReport(client_order_id="a", status="Cancelled", filled_quantity=3)],
        working=[],
    )
    reconcile_with_broker(store, broker, now=NOW)
    assert _status(store, "a") == OrderStatus.CANCELLED
    assert len(store.load_fills(client_order_id="a")) == 1


def test_filled_without_visible_executions_is_reported_not_guessed(store: OrderStore) -> None:
    broker = _CompletedBroker(
        completed=[BrokerOrderReport(client_order_id="d", status="Filled", filled_quantity=10)],
        working=[_order("a"), _order("b"), _order("c")],
    )
    report = reconcile_with_broker(store, broker, now=NOW)
    assert report.filled_without_executions == ["d"]
    assert _status(store, "d") == OrderStatus.SUBMITTED  # no fill price, no FILLED claim
    assert report.needs_attention


def test_unknown_to_broker_is_reported_never_retired(store: OrderStore) -> None:
    broker = _CompletedBroker(completed=[], working=[_order("a")])
    report = reconcile_with_broker(store, broker, now=NOW)
    assert report.unseen_open == ["b", "c", "d"]
    for coid in ("b", "c", "d"):
        assert _status(store, coid) == OrderStatus.SUBMITTED
    assert "orders resolve" in report.summary()


def test_without_completed_orders_capability_nothing_is_called_unseen(store: OrderStore) -> None:
    """A broker that cannot list completed orders proves nothing about absence."""
    broker = _Broker(working=[])
    report = reconcile_with_broker(store, broker, now=NOW)
    assert report.unseen_open == []


def test_falls_back_to_session_fills(store: OrderStore) -> None:
    class _SessionOnly:
        def get_fills(self, *, since=None):
            return [_fill("a", 10, "e1")]

        def get_open_orders(self):
            return []

    report = reconcile_with_broker(store, _SessionOnly(), now=NOW)
    assert report.sources == ["session_fills"]
    assert _status(store, "a") == OrderStatus.FILLED


def test_window_starts_at_oldest_open_order_capped(tmp_path: Path) -> None:
    s = OrderStore(tmp_path / "orders.db")
    s.save_order(_order("old", days_ago=40))
    broker = _CompletedBroker(completed=[], working=[])
    reconcile_with_broker(s, broker, now=NOW)
    assert broker.since == NOW - timedelta(days=14)


def test_no_open_orders_means_no_broker_calls(tmp_path: Path) -> None:
    class _Boom:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected broker call {name}")

    report = reconcile_with_broker(OrderStore(tmp_path / "o.db"), _Boom(), now=NOW)
    assert not report.changed and not report.needs_attention


def test_naive_now_is_rejected(store: OrderStore) -> None:
    with pytest.raises(ValueError):
        reconcile_with_broker(store, _Broker(), now=datetime(2026, 9, 28))


def test_broker_read_error_propagates(store: OrderStore) -> None:
    class _Down(_Broker):
        def get_executions(self, *, since=None):
            raise ConnectionError("gateway down")

    with pytest.raises(ConnectionError):
        reconcile_with_broker(store, _Down(), now=NOW)
    assert _status(store, "a") == OrderStatus.SUBMITTED


def test_runner_job_alerts_on_change_and_debounces_repeat_gaps(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import trading.runner.runner as runner_module
    from trading.runner.runner import Runner

    monkeypatch.setattr(runner_module, "settings", SimpleNamespace(state_dir=tmp_path))
    store = OrderStore(tmp_path / "orders.db")
    store.save_order(_order("stale"))
    sent: list[tuple[str, str]] = []
    alerts = SimpleNamespace(
        info=lambda m: sent.append(("info", m)), warning=lambda m: sent.append(("warning", m))
    )
    fake = SimpleNamespace(
        cycle=SimpleNamespace(order_store=store),
        broker=_CompletedBroker(completed=[], working=[]),
        alerts=alerts,
        _RECONCILE_STATE_FILE=Runner._RECONCILE_STATE_FILE,
    )

    Runner._run_broker_reconcile(fake)  # type: ignore[arg-type]
    Runner._run_broker_reconcile(fake)  # type: ignore[arg-type]

    assert [level for level, _ in sent] == ["warning"]
    assert "stale" in sent[0][1]
