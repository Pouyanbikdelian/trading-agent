r"""Orders the ledger can no longer account for.

2026-09-11, production. A cycle warned:

    ⚠️ 5 order(s) still open — never reached a terminal state:
      `V` sell 3 (pending, 29d old) ... `MU` sell 4 (pending, 18d old)
    _Local position view may differ from the broker. Check /orders._

`/orders`, three minutes later:

    no orders in the last 7 days.

Both statements were true. Together they were useless, and the operator
had no way to tell whether five live orders were sitting at IBKR. They
were not: IBKR showed no working orders and no V/MU/AMD position. The
rows were ledger orphans — written PENDING by the manual-order path,
never transitioned, and by then far outside the 14-day reconciliation
window, so no future cycle could ever settle them.

Three separate defects, one symptom. This file covers the two that are
about *reading and retiring* the rows; the missing PENDING → SUBMITTED
transition is in ``test_command_orders_recorded.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
from trading.runtime.command_processor import RESOLVE_MIN_AGE_DAYS, _h_resolve_orders
from trading.runtime.commands import Command, CommandType

NOW = datetime(2026, 9, 11, 17, 20, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _frozen_resolve_clock(monkeypatch):
    """The fixtures are dated relative to NOW; so is the handler's clock."""
    import trading.runtime.command_processor as cp

    monkeypatch.setattr(cp, "_utcnow", lambda: NOW)


def _order(sym: str, side: Side = Side.SELL, *, days_ago: float, qty: float = 3.0) -> Order:
    return Order(
        client_order_id=f"cmd-{sym}-{days_ago:g}",
        instrument=Instrument(symbol=sym, asset_class=AssetClass.EQUITY),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=NOW - timedelta(days=days_ago),
    )


def _store_with_the_august_rows(tmp_path: Path) -> tuple[OrderStore, list[Order]]:
    """The five real rows, as they sat on 2026-09-11."""
    store = OrderStore(tmp_path / "orders.db")
    orders = [
        _order("V", Side.SELL, days_ago=29),
        _order("MU", Side.BUY, days_ago=29),
        _order("AMD", Side.BUY, days_ago=29, qty=6),
        _order("AMD", Side.SELL, days_ago=18, qty=8),
        _order("MU", Side.SELL, days_ago=18, qty=4),
    ]
    for o in orders:
        store.save_order(o)
    return store, orders


class TestOpenRowsAreNeverHiddenByAge:
    def test_the_store_returns_every_open_row_however_old(self, tmp_path) -> None:
        store, orders = _store_with_the_august_rows(tmp_path)

        assert len(store.open_orders()) == len(orders)

    def test_terminal_rows_are_excluded(self, tmp_path) -> None:
        store, orders = _store_with_the_august_rows(tmp_path)
        store.update_status(orders[0].client_order_id, OrderStatus.FILLED)

        symbols = {o.instrument.symbol for o, _st, _b in store.open_orders()}
        assert symbols == {"MU", "AMD"}

    def test_orders_command_no_longer_answers_no_orders(self, tmp_path, monkeypatch) -> None:
        """The exact contradiction, closed."""
        from trading.bot import telegram as telegram_module

        store, _orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            telegram_module,
            "settings",
            SimpleNamespace(state_dir=tmp_path, trading_env="research"),
        )

        out = telegram_module._cmd_orders()

        assert "no orders in the last 7 days" not in out
        assert "In flight: 5 order(s)" in out
        assert "predate the broker's execution history" in out
        assert "V" in out and "MU" in out and "AMD" in out

    def test_pending_command_shows_a_month_old_order(self, tmp_path, monkeypatch) -> None:
        """`/pending` filtered to two days, so the only orders worth
        chasing were the ones it could not show."""
        from trading.bot import telegram as telegram_module

        store, _orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            telegram_module,
            "settings",
            SimpleNamespace(state_dir=tmp_path, trading_env="research"),
        )

        out = telegram_module._cmd_pending_orders()

        assert "5 order(s) in flight" in out
        assert "no orders currently in flight" not in out
        assert "d ago" in out  # days, not "41760m ago"


class TestResolvingRequiresBrokerEvidence:
    @staticmethod
    def _cmd() -> Command:
        return Command(id="c1", type=CommandType.RESOLVE_ORDERS, args={}, requested_by="yan")

    @staticmethod
    def _broker(working: list[Order] | None = None) -> SimpleNamespace:
        return SimpleNamespace(get_open_orders=lambda: list(working or []))

    def test_a_broker_that_cannot_be_read_resolves_nothing(self, tmp_path, monkeypatch) -> None:
        """No evidence, no resolution — and nothing written on the way out."""
        import trading.core.config as config_module

        store, _orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )

        def unreachable():
            raise RuntimeError("not connected to IB Gateway")

        with pytest.raises(RuntimeError, match="not connected"):
            _h_resolve_orders(self._cmd(), SimpleNamespace(get_open_orders=unreachable))

        reopened = OrderStore(tmp_path / "orders.db")
        assert len(reopened.open_orders()) == 5

    def test_the_august_rows_are_retired_without_inventing_an_outcome(
        self, tmp_path, monkeypatch
    ) -> None:
        import trading.core.config as config_module

        store, _orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )

        out = _h_resolve_orders(self._cmd(), self._broker([]))

        assert out["n_resolved"] == 5
        assert out["broker_working"] == 0
        reopened = OrderStore(tmp_path / "orders.db")
        assert reopened.open_orders() == []
        statuses = {
            r["status"] for r in reopened.conn.execute("SELECT status FROM orders").fetchall()
        }
        assert statuses == {"unreconciled"}
        # The point of the whole exercise: no guessed outcome.
        assert "filled" not in statuses and "cancelled" not in statuses

    def test_every_resolution_carries_an_audit_note(self, tmp_path, monkeypatch) -> None:
        import trading.core.config as config_module

        store, orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )

        _h_resolve_orders(self._cmd(), self._broker([]))

        reopened = OrderStore(tmp_path / "orders.db")
        note, at = reopened.resolution_note(orders[0].client_order_id)
        assert note is not None
        assert "yan" in note
        assert "NOT a fill" in note
        assert at is not None

    def test_an_order_the_broker_still_recognises_is_left_alone(
        self, tmp_path, monkeypatch
    ) -> None:
        """A false skip leaves a row alarming. A false resolution silences
        a live order. Only one of those is recoverable."""
        import trading.core.config as config_module

        store, orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )

        out = _h_resolve_orders(self._cmd(), self._broker([orders[0]]))

        assert out["n_resolved"] == 4
        assert out["n_skipped"] == 1
        reopened = OrderStore(tmp_path / "orders.db")
        still_open = [o.client_order_id for o, _st, _b in reopened.open_orders()]
        assert still_open == [orders[0].client_order_id]

    def test_a_matching_symbol_and_side_is_enough_to_skip(self, tmp_path, monkeypatch) -> None:
        """The broker may report an order under its own id. Matching the
        leg as well means an unrecognised id cannot cause a wrong retire."""
        import trading.core.config as config_module

        store, _orders = _store_with_the_august_rows(tmp_path)
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )
        broker_side = _order("V", Side.SELL, days_ago=0)
        broker_side = broker_side.model_copy(update={"client_order_id": "broker-9931"})

        out = _h_resolve_orders(self._cmd(), self._broker([broker_side]))

        assert out["n_skipped"] == 1
        assert any("V" in name for name in out["skipped"])

    def test_a_row_still_inside_the_window_is_never_touched(self, tmp_path, monkeypatch) -> None:
        """Resolving something reconciliation can still settle destroys
        information for no reason."""
        import trading.core.config as config_module

        store = OrderStore(tmp_path / "orders.db")
        fresh = _order("NVDA", Side.BUY, days_ago=2)
        store.save_order(fresh)
        store.save_order(_order("V", Side.SELL, days_ago=29))
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )

        out = _h_resolve_orders(self._cmd(), self._broker([]))

        assert out["n_resolved"] == 1
        reopened = OrderStore(tmp_path / "orders.db")
        assert [o.instrument.symbol for o, _st, _b in reopened.open_orders()] == ["NVDA"]

    def test_the_minimum_age_cannot_be_argued_below_the_lookback(
        self, tmp_path, monkeypatch
    ) -> None:
        import trading.core.config as config_module

        store = OrderStore(tmp_path / "orders.db")
        store.save_order(_order("NVDA", Side.BUY, days_ago=1))
        store.close()
        monkeypatch.setattr(
            config_module,
            "settings",
            config_module.settings.model_copy(update={"state_dir": tmp_path}),
        )
        cmd = Command(
            id="c1",
            type=CommandType.RESOLVE_ORDERS,
            args={"older_than_days": 0},
            requested_by="yan",
        )

        out = _h_resolve_orders(cmd, self._broker([]))

        assert out["older_than_days"] == RESOLVE_MIN_AGE_DAYS
        assert out["n_resolved"] == 0


class TestAResolvedRowStopsAlarming:
    def test_it_leaves_the_open_set_and_the_reconciliation_window(self, tmp_path) -> None:
        """The alarm and the lookback both key off OPEN_STATUSES, so this
        single transition is what actually ends the month of noise."""
        store, orders = _store_with_the_august_rows(tmp_path)
        assert store.oldest_open_created_at() is not None

        for o in orders:
            store.mark_unreconciled(o.client_order_id, note="broker had nothing working", at=NOW)

        assert store.open_orders() == []
        assert store.open_orders_older_than(NOW) == []
        assert store.oldest_open_created_at() is None
        assert OrderStatus.UNRECONCILED not in OrderStore.OPEN_STATUSES


class TestTheAlarmSaysWhichRowsCanStillHeal:
    """The old message told the operator to "check /orders" and implied
    waiting might help. For rows past the reconciliation window it never
    could, and nothing in the text said so — which is why the same five
    rows alarmed on every cycle for a month."""

    @staticmethod
    def _cycle(store: OrderStore):
        from trading.runner.cycle import Cycle

        class _Alerts:
            def __init__(self) -> None:
                self.errors: list[str] = []

            def error(self, m: str) -> None:
                self.errors.append(m)

        class _Bare:
            RECONCILE_LOOKBACK = Cycle.RECONCILE_LOOKBACK
            STALE_ORDER_AGE = Cycle.STALE_ORDER_AGE
            _warn_on_stale_open_orders = Cycle._warn_on_stale_open_orders

            def __init__(self) -> None:
                self.alerts = _Alerts()
                self.order_store = store

        return _Bare()

    def test_beyond_window_rows_are_named_as_unhealable(self, tmp_path) -> None:
        store, _orders = _store_with_the_august_rows(tmp_path)
        cycle = self._cycle(store)

        cycle._warn_on_stale_open_orders(NOW)

        (msg,) = cycle.alerts.errors
        assert "5 order(s) still open" in msg
        assert "no future cycle can settle them" in msg
        assert "/orders resolve" in msg

    def test_a_recently_stale_row_is_not_declared_dead(self, tmp_path) -> None:
        store = OrderStore(tmp_path / "orders.db")
        store.save_order(_order("NVDA", Side.BUY, days_ago=4))
        cycle = self._cycle(store)

        cycle._warn_on_stale_open_orders(NOW)

        (msg,) = cycle.alerts.errors
        assert "may settle on their own" in msg
        assert "/orders resolve" not in msg

    def test_a_healthy_ledger_stays_silent(self, tmp_path) -> None:
        store = OrderStore(tmp_path / "orders.db")
        store.save_order(_order("NVDA", Side.BUY, days_ago=0))
        cycle = self._cycle(store)

        cycle._warn_on_stale_open_orders(NOW)

        assert cycle.alerts.errors == []

    def test_resolved_rows_end_the_noise(self, tmp_path) -> None:
        """End to end: the month of repeated alarms stops."""
        store, orders = _store_with_the_august_rows(tmp_path)
        for o in orders:
            store.mark_unreconciled(o.client_order_id, note="broker had nothing working", at=NOW)
        cycle = self._cycle(store)

        cycle._warn_on_stale_open_orders(NOW)

        assert cycle.alerts.errors == []
