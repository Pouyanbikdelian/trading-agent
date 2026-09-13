"""An operator's own trades must reach the ledger.

2026-08-11: `/sell UNH` submitted at 14:14:43, and `/orders` a minute
later listed only the ten buys the cycle had placed. The command
handlers call ``broker.submit_order`` directly and nothing wrote to
orders.db, so `/buy` `/sell` `/close` `/flatten` were invisible to
`/orders`, `/pending`, fill reconciliation, realized P&L, and
``agents.exits.recent_exits``.

The cycle already saved its orders, so the gap was exactly the trades
the operator placed by hand — the ones least reconstructable later.

Fixed by wrapping the broker, not by patching six handlers: the seventh
would have been missed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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
from trading.execution.store import OrderStore
from trading.risk import RiskLimits, RiskManager
from trading.runtime.command_processor import _recording, _RecordingBroker

TS = datetime(2026, 8, 11, 14, 14, 43, tzinfo=timezone.utc)


def _order(sym: str = "UNH", side: Side = Side.SELL, qty: float = 1.0) -> Order:
    return Order(
        client_order_id=f"cmd-{sym}",
        instrument=Instrument(symbol=sym, asset_class=AssetClass.EQUITY),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=TS,
    )


class _Broker:
    def __init__(self) -> None:
        self.submitted: list[Order] = []

    def submit_order(self, order: Order) -> str:
        self.submitted.append(order)
        return "ok"

    def get_positions(self) -> list:
        return []


class TestTheOrderReachesTheLedger:
    def test_a_command_sell_is_recorded(self, tmp_path) -> None:
        """The regression, in the exact shape it occurred."""
        store = OrderStore(tmp_path / "orders.db")
        rec = _RecordingBroker(_Broker(), store)

        rec.submit_order(_order())

        rows = store.conn.execute("SELECT client_order_id FROM orders").fetchall()
        assert [r[0] for r in rows] == ["cmd-UNH"]

    def test_it_is_still_submitted(self, tmp_path) -> None:
        """Bookkeeping must not replace the trade."""
        inner = _Broker()
        rec = _RecordingBroker(inner, OrderStore(tmp_path / "orders.db"))

        rec.submit_order(_order())

        assert len(inner.submitted) == 1

    def test_the_order_is_saved_before_it_is_sent(self, tmp_path) -> None:
        """An order sent and then lost to a crash is worse than a PENDING
        row with no fill."""
        seen: list[str] = []
        store = OrderStore(tmp_path / "orders.db")

        class _Watch:
            def submit_order(self, order):
                n = store.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
                seen.append(f"rows_at_submit={n}")
                return "ok"

        _RecordingBroker(_Watch(), store).submit_order(_order())

        assert seen == ["rows_at_submit=1"]

    def test_a_ledger_failure_does_not_block_the_trade(self, tmp_path) -> None:
        """Never lose a fill because bookkeeping broke."""

        class _BadStore:
            def save_order(self, order):
                raise RuntimeError("disk full")

        inner = _Broker()

        _RecordingBroker(inner, _BadStore()).submit_order(_order())

        assert len(inner.submitted) == 1

    def test_halt_gate_wraps_the_actual_submit_not_just_command_start(self, tmp_path) -> None:
        """A halt that lands after command dispatch still blocks the broker call."""
        inner = _Broker()
        manager = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
        manager.halt("continuous monitor")
        rec = _RecordingBroker(
            inner,
            OrderStore(tmp_path / "orders.db"),
            risk_manager=manager,
        )

        with pytest.raises(RuntimeError, match="halt engaged"):
            rec.submit_order(_order())

        assert inner.submitted == []


class TestItIsATransparentProxy:
    def test_other_broker_calls_pass_through(self, tmp_path) -> None:
        """Handlers call get_positions, cancel, account summary — the
        proxy must be invisible for everything but submit."""
        inner = _Broker()
        rec = _RecordingBroker(inner, OrderStore(tmp_path / "orders.db"))

        assert rec.get_positions() == []

    def test_unknown_attributes_delegate(self, tmp_path) -> None:
        inner = SimpleNamespace(anything=lambda: "delegated", submit_order=lambda o: None)
        rec = _RecordingBroker(inner, OrderStore(tmp_path / "orders.db"))

        assert rec.anything() == "delegated"

    def test_a_broken_ledger_path_degrades_to_the_raw_broker(self, tmp_path) -> None:
        """A dashboard-style degrade: trading continues unrecorded rather
        than not at all."""
        inner = _Broker()

        out = _recording(inner, tmp_path / "definitely" / "not" / "here")

        out.submit_order(_order())
        assert len(inner.submitted) == 1


def test_the_processor_uses_the_recording_broker() -> None:
    """A proxy nothing routes through records nothing."""
    src = Path("src/trading/runtime/command_processor.py").read_text()

    # The proxy is deliberately created inside the execution lock now: the
    # halt state is re-read there before choosing ordinary vs reduce-only
    # command execution.  Both order paths must still use the same ledger
    # wrapper; only non-order recovery commands use the raw broker.
    assert "used = _recording(" in src
    assert "risk_manager=risk_manager" in src
    assert "result = handler(cmd, used)" in src
    assert "result = _execute_halted_reduce_only(cmd, used)" in src


def test_every_order_submitting_command_is_covered() -> None:
    """The proxy is applied on the needs_lock branch, which is exactly
    _ORDER_SUBMITTING_COMMANDS — so a new order command inherits it."""
    src = Path("src/trading/runtime/command_processor.py").read_text()

    assert "needs_lock = cmd.type in _ORDER_SUBMITTING_COMMANDS" in src
    i_flag = src.index("needs_lock = cmd.type in _ORDER_SUBMITTING_COMMANDS")
    i_use = src.index("used = _recording(")
    assert i_flag < i_use


class TestTheRowIsSettledOnTheBrokersAnswer:
    """Saving the row was only half the job.

    Until 2026-09-13 the recorder wrote PENDING and never touched the row
    again, so every operator order stayed PENDING for life. PENDING means
    "created locally, not sent" — a direct lie about an order the broker
    had acknowledged, and one that left `/pending`, `/orders` and the
    reconciliation matcher working from a false premise. Five such rows
    from August 2026 were still open a month later.
    """

    @staticmethod
    def _status(store: OrderStore, client_order_id: str = "cmd-UNH") -> str:
        row = store.conn.execute(
            "SELECT status FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return str(row["status"])

    def test_an_acknowledged_order_is_marked_submitted(self, tmp_path) -> None:
        store = OrderStore(tmp_path / "orders.db")

        _RecordingBroker(_Broker(), store).submit_order(_order())

        assert self._status(store) == "submitted"

    def test_a_refused_order_is_marked_rejected_and_still_raises(self, tmp_path) -> None:
        """A row that can never fill must not masquerade as working."""
        store = OrderStore(tmp_path / "orders.db")

        class _Refuses:
            def submit_order(self, order):
                raise RuntimeError("insufficient margin")

        rec = _RecordingBroker(_Refuses(), store)

        with pytest.raises(RuntimeError, match="insufficient margin"):
            rec.submit_order(_order())

        assert self._status(store) == "rejected"

    def test_the_brokers_own_id_is_kept_when_it_returns_one(self, tmp_path) -> None:
        """Reconciliation matches executions by broker id; drop it and an
        old row can never be tied back to its fill."""
        store = OrderStore(tmp_path / "orders.db")

        class _Acks:
            def submit_order(self, order):
                return SimpleNamespace(broker_order_id="ib-5512")

        _RecordingBroker(_Acks(), store).submit_order(_order())

        row = store.conn.execute(
            "SELECT broker_order_id FROM orders WHERE client_order_id = 'cmd-UNH'"
        ).fetchone()
        assert row["broker_order_id"] == "ib-5512"

    def test_a_settle_failure_never_blocks_the_trade(self, tmp_path) -> None:
        inner = _Broker()

        class _HalfBrokenStore:
            def save_order(self, order): ...

            def update_status(self, *a, **kw):
                raise RuntimeError("db locked")

        _RecordingBroker(inner, _HalfBrokenStore()).submit_order(_order())

        assert len(inner.submitted) == 1

    def test_an_unsaved_order_is_not_settled(self, tmp_path) -> None:
        """No row was written, so there is nothing to update — and a blind
        UPDATE would silently do nothing anyway."""
        touched: list[str] = []

        class _WriteOnlyFails:
            def save_order(self, order):
                raise RuntimeError("disk full")

            def update_status(self, *a, **kw):
                touched.append("update")

        _RecordingBroker(_Broker(), _WriteOnlyFails()).submit_order(_order())

        assert touched == []

    def test_no_operator_order_is_left_claiming_it_was_never_sent(self, tmp_path) -> None:
        """The property that matters: after a normal command, nothing in
        the ledger still says PENDING."""
        store = OrderStore(tmp_path / "orders.db")
        rec = _RecordingBroker(_Broker(), store)

        for sym in ("V", "MU", "AMD"):
            rec.submit_order(_order(sym))

        rows = store.conn.execute("SELECT status FROM orders").fetchall()
        assert [r["status"] for r in rows] == ["submitted"] * 3
