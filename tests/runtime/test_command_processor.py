"""Tests for the command processor's halt and reduce-only safety gates.

A halt must stop exposure increases, but it must not turn the account into a
prison: an explicitly requested, broker-verified close remains available for
the operator and the position guards.  The exit path is intentionally much
narrower than an ordinary SELL and fails closed if its safety proof is absent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
    TimeInForce,
)
from trading.risk import RiskLimits, RiskManager
from trading.runner.alerts import TelegramAlerts
from trading.runtime.command_processor import process_pending
from trading.runtime.commands import Command, CommandType, submit


class _FakeBroker:
    """Records submit_order calls so tests can assert what was/wasn't sent."""

    def __init__(self) -> None:
        self.submitted: list[Order] = []
        self.positions: list[Position] = []
        self.strict_positions: list[Position] | None = None
        self.open_orders: list[Order] = []
        self.open_orders_error: Exception | None = None
        self.strict_position_reads = 0
        self.connected = True

    def submit_order(self, order: Order) -> None:
        self.submitted.append(order)

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_positions_strict(self) -> list[Position]:
        self.strict_position_reads += 1
        return list(self.positions if self.strict_positions is None else self.strict_positions)

    def get_open_orders(self) -> list[Order]:
        if self.open_orders_error is not None:
            raise self.open_orders_error
        return list(self.open_orders)


class _RecordingAlerts(TelegramAlerts):
    """In-process alert sink — captures everything for assertions."""

    def __init__(self) -> None:
        super().__init__(token=None, chat_id=None, enabled=False)
        self.messages: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))

    def critical(self, msg: str) -> None:
        self.messages.append(("critical", msg))


@pytest.fixture
def state_dir(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path


@pytest.fixture
def risk_manager(tmp_path: Path) -> RiskManager:
    return RiskManager(
        RiskLimits(
            max_position_pct=0.10,
            max_gross_exposure=1.0,
            max_net_exposure=1.0,
            max_sector_exposure=0.30,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.15,
        ),
        halt_state_path=tmp_path / "halt.json",
    )


def _position(symbol: str, quantity: float) -> Position:
    return Position(
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY),
        quantity=quantity,
        avg_price=100.0,
    )


def _working_order(symbol: str, side: Side, quantity: float) -> Order:
    return Order(
        client_order_id=f"working-{symbol}-{side.value}-{quantity}",
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY),
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=datetime.now(tz=timezone.utc),
    )


def _stock_instrument(
    symbol: str,
    *,
    asset_class: AssetClass,
    exchange: str,
    currency: str = "USD",
) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_class=asset_class,
        exchange=exchange,
        currency=currency,
    )


def _working_order_for_instrument(instrument: Instrument, side: Side, quantity: float) -> Order:
    return Order(
        client_order_id=f"working-{instrument.symbol}-{side.value}-{quantity}",
        instrument=instrument,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=datetime.now(tz=timezone.utc),
    )


def test_halt_gate_blocks_buy_when_halted(state_dir, risk_manager) -> None:
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    risk_manager.halt("manual: testing halt gate")

    cmd = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 10})
    submit(cmd, state_dir)

    n = process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert n == 1
    # No order should have reached the broker.
    assert broker.submitted == []
    # The operator should see a refusal — not silent.
    error_messages = [m for level, m in alerts.messages if level == "error"]
    assert len(error_messages) == 1
    assert "halted" in error_messages[0].lower()


def test_halt_gate_blocks_exposure_increasing_order_types(state_dir, risk_manager) -> None:
    """BUY, discretionary SELL, and FX remain unavailable while halted."""
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    risk_manager.halt("safety test")

    submit(Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 1}), state_dir)
    # A SELL remains blocked even if the spelling looks like a full exit.
    # Only the explicit CLOSE command can enter the strictly verified path.
    submit(Command.new(CommandType.SELL, args={"symbol": "AAPL", "qty": "all"}), state_dir)
    submit(
        Command.new(
            CommandType.FX_CONVERT,
            args={"from_ccy": "CHF", "to_ccy": "USD", "amount": 25000},
        ),
        state_dir,
    )

    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert broker.submitted == []
    assert len([m for level, m in alerts.messages if level == "error"]) == 3


def test_halted_close_uses_fresh_position_and_is_reduce_only(state_dir, risk_manager) -> None:
    """A halted /close must size from a fresh broker snapshot, not cache state."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 1)]  # stale-looking fallback view
    broker.strict_positions = [_position("AAPL", 10)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.CLOSE, args={"symbol": "AAPL"}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.strict_position_reads == 1
    assert len(broker.submitted) == 1
    order = broker.submitted[0]
    assert order.instrument.symbol == "AAPL"
    assert order.side == Side.SELL
    assert order.quantity == 10
    assert order.client_order_id.startswith("halted-close-")
    assert any("Close AAPL" in msg for level, msg in alerts.messages if level == "info")


def test_halted_guard_close_uses_the_same_reduce_only_path(state_dir, risk_manager) -> None:
    """Trailing-stop commands are CLOSE commands and stay available in a halt."""
    broker = _FakeBroker()
    broker.positions = [_position("GEV", 9)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(
        Command.new(
            CommandType.CLOSE,
            args={"symbol": "GEV"},
            requested_by="guard:trailing_stop",
        ),
        state_dir,
    )
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert [(o.instrument.symbol, o.side, o.quantity) for o in broker.submitted] == [
        ("GEV", Side.SELL, 9)
    ]


def test_halted_close_buys_only_the_actual_short_to_cover(state_dir, risk_manager) -> None:
    """The reduce-only proof works in both directions, including legacy shorts."""
    broker = _FakeBroker()
    broker.positions = [_position("TSLA", -3)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.CLOSE, args={"symbol": "TSLA"}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert [(o.instrument.symbol, o.side, o.quantity) for o in broker.submitted] == [
        ("TSLA", Side.BUY, 3)
    ]


def test_halted_close_nets_verified_working_exit_without_flipping(state_dir, risk_manager) -> None:
    """Only the uncovered live amount is allowed after a working exit."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 10)]
    broker.open_orders = [_working_order("AAPL", Side.SELL, 4)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.CLOSE, args={"symbol": "AAPL"}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert [(o.side, o.quantity) for o in broker.submitted] == [(Side.SELL, 6)]


def test_halted_close_recognises_equivalent_etf_and_equity_contract_metadata(
    state_dir, risk_manager
) -> None:
    """IBKR venue/class metadata must not hide an existing same-stock exit."""
    broker = _FakeBroker()
    broker.positions = [
        Position(
            instrument=_stock_instrument("GLD", asset_class=AssetClass.ETF, exchange="ARCA"),
            quantity=10,
            avg_price=100.0,
        )
    ]
    broker.open_orders = [
        _working_order_for_instrument(
            _stock_instrument("GLD", asset_class=AssetClass.EQUITY, exchange="SMART"),
            Side.SELL,
            10,
        )
    ]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.CLOSE, args={"symbol": "GLD"}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    assert not [msg for level, msg in alerts.messages if level == "error"]
    assert any("already covered" in msg for level, msg in alerts.messages if level == "info")


def test_halted_close_rejects_a_conflicting_working_order(state_dir, risk_manager) -> None:
    """A working BUY means a new SELL cannot be proven reduce-only."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 10)]
    broker.open_orders = [_working_order("AAPL", Side.BUY, 1)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.CLOSE, args={"symbol": "AAPL"}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    errors = [msg for level, msg in alerts.messages if level == "error"]
    assert len(errors) == 1
    assert "increase exposure" in errors[0]


def test_halted_close_fails_closed_when_working_orders_cannot_be_verified(
    state_dir, risk_manager
) -> None:
    """No exit order is safe when an existing opposing order is unknowable."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 10)]
    broker.open_orders_error = ConnectionError("gateway unavailable")
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.CLOSE, args={"symbol": "AAPL"}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    errors = [msg for level, msg in alerts.messages if level == "error"]
    assert len(errors) == 1
    assert "cannot verify broker working orders" in errors[0]


def test_halted_flatten_reduces_long_and_short_positions(state_dir, risk_manager) -> None:
    """A verified flatten routes each actual position toward zero, never through it."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 10), _position("TSLA", -3)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.FLATTEN), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert {(o.instrument.symbol, o.side, o.quantity) for o in broker.submitted} == {
        ("AAPL", Side.SELL, 10),
        ("TSLA", Side.BUY, 3),
    }
    assert len({o.client_order_id for o in broker.submitted}) == 2


def test_halted_flatten_normalises_equivalent_etf_and_equity_contract_metadata(
    state_dir, risk_manager
) -> None:
    """Flatten must recognise an ARCA ETF exit reported as SMART/EQUITY."""
    broker = _FakeBroker()
    broker.positions = [
        Position(
            instrument=_stock_instrument("GLD", asset_class=AssetClass.ETF, exchange="ARCA"),
            quantity=10,
            avg_price=100.0,
        )
    ]
    broker.open_orders = [
        _working_order_for_instrument(
            _stock_instrument("GLD", asset_class=AssetClass.EQUITY, exchange="SMART"),
            Side.SELL,
            10,
        )
    ]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.FLATTEN), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    assert not [msg for level, msg in alerts.messages if level == "error"]
    assert any("already covered" in msg for level, msg in alerts.messages if level == "info")


def test_halted_flatten_does_not_conflate_same_symbol_with_different_currency(
    state_dir, risk_manager
) -> None:
    """Currency remains part of the identity, so a separate listing cannot be hidden."""
    broker = _FakeBroker()
    broker.positions = [
        Position(
            instrument=_stock_instrument("GLD", asset_class=AssetClass.ETF, exchange="ARCA"),
            quantity=10,
            avg_price=100.0,
        )
    ]
    broker.open_orders = [
        _working_order_for_instrument(
            _stock_instrument(
                "GLD", asset_class=AssetClass.EQUITY, exchange="SMART", currency="CHF"
            ),
            Side.SELL,
            10,
        )
    ]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.FLATTEN), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    errors = [msg for level, msg in alerts.messages if level == "error"]
    assert len(errors) == 1
    assert "could create exposure" in errors[0]


def test_halted_flatten_rejects_orphan_or_conflicting_working_orders(
    state_dir, risk_manager
) -> None:
    """No partial flatten may claim safety while a queued order can reopen risk."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 10)]
    broker.open_orders = [_working_order("MSFT", Side.BUY, 1)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")

    submit(Command.new(CommandType.FLATTEN), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    errors = [msg for level, msg in alerts.messages if level == "error"]
    assert len(errors) == 1
    assert "could create exposure" in errors[0]


def test_halted_close_still_obeys_the_order_ttl(state_dir, risk_manager) -> None:
    """The emergency exit exception never resurrects an old /close."""
    broker = _FakeBroker()
    broker.positions = [_position("AAPL", 10)]
    alerts = _RecordingAlerts()
    risk_manager.halt("daily loss")
    stale = replace(
        Command.new(CommandType.CLOSE, args={"symbol": "AAPL"}),
        requested_at=(datetime.now(tz=timezone.utc) - timedelta(minutes=16)).isoformat(),
    )

    submit(stale, state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    assert broker.submitted == []
    assert any("expired" in msg for level, msg in alerts.messages if level == "error")


def test_halt_gate_allows_non_order_commands_when_halted(state_dir, risk_manager) -> None:
    """CANCEL, REFRESH, RECONNECT are recovery actions — allowed during halt."""
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    risk_manager.halt("safety test")

    # REFRESH_DATA is a no-op handler so it'll just succeed.
    submit(Command.new(CommandType.REFRESH_DATA), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    # No halt-related error — the refresh handler ran.
    halt_errors = [m for level, m in alerts.messages if level == "error" and "halt" in m.lower()]
    assert halt_errors == []


def test_halt_gate_allows_buy_when_not_halted(state_dir, risk_manager) -> None:
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    assert not risk_manager.is_halted()

    cmd = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 10})
    submit(cmd, state_dir)

    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert len(broker.submitted) == 1
    assert broker.submitted[0].instrument.symbol == "AAPL"
    assert broker.submitted[0].quantity == 10


def test_no_risk_manager_skips_halt_gate(state_dir) -> None:
    """Back-compat: callers that don't pass a risk_manager get the old behavior.
    This keeps the existing test surface and lets us roll the change out
    without touching every call site."""
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    cmd = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 5})
    submit(cmd, state_dir)
    process_pending(broker, state_dir, alerts)  # no risk_manager kwarg
    assert len(broker.submitted) == 1


def test_buy_passes_through_correct_order(state_dir, risk_manager) -> None:
    """Sanity check: when allowed, the order built carries the right
    side/qty/symbol/instrument metadata so the broker submits a real order."""
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    submit(
        Command.new(CommandType.BUY, args={"symbol": "MSFT", "qty": 25, "limit": 500.0}),
        state_dir,
    )
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert len(broker.submitted) == 1
    o = broker.submitted[0]
    assert o.instrument == Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY)
    assert o.quantity == 25
    assert o.limit_price == 500.0


def test_halt_gate_picks_up_resume_from_disk_between_calls(state_dir, risk_manager) -> None:
    """Regression for 2026-05-22 follow-up.

    Trader and bot are separate processes; both read/write
    ``state/halt.json``. After the bot's /resume rewrites the file, a
    queued /buy must NOT still be refused by the trader's stale
    in-memory halt state. process_pending must reload halt.json before
    enforcing the gate.
    """
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    risk_manager.halt("test: halt then external unhalt")
    assert risk_manager.is_halted()

    # External process (bot) clears halt.json on disk while the trader
    # still has the old state in memory.
    halt_path = risk_manager._halt_path  # type: ignore[attr-defined]
    halt_path.write_text(
        '{"halted": false, "reason": "", "halted_at": null, '
        '"equity_high_watermark": 0.0, "daily_equity_open": 0.0, '
        '"last_day": null}'
    )
    # In-memory still says halted — only a reload should flip it.
    assert risk_manager.is_halted()

    cmd = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 10})
    submit(cmd, state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)

    # process_pending must have reloaded halt.json and let the BUY through.
    assert len(broker.submitted) == 1
    assert broker.submitted[0].instrument.symbol == "AAPL"
    # And the in-memory state should now reflect the on-disk reality.
    assert not risk_manager.is_halted()


# ---------------------------------------------------------------------------
# TTL gate — order-submitting commands expire.
# A /flatten typed while the runner was down must not fire when the runner
# comes back hours later. Stale May-2026 commands found in a July audit
# motivated this gate.
# ---------------------------------------------------------------------------


def _aged(cmd_type: CommandType, *, minutes_old: float, args: dict | None = None):
    """Build a command whose requested_at is `minutes_old` minutes in the past."""
    import dataclasses
    from datetime import datetime, timedelta, timezone

    cmd = Command.new(cmd_type, args=args)
    old_ts = (datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_old)).isoformat()
    return dataclasses.replace(cmd, requested_at=old_ts)


def test_ttl_gate_blocks_stale_order_commands(state_dir, risk_manager) -> None:
    broker = _FakeBroker()
    alerts = _RecordingAlerts()

    submit(_aged(CommandType.FLATTEN, minutes_old=60), state_dir)
    submit(
        _aged(
            CommandType.FX_CONVERT,
            minutes_old=3 * 24 * 60,  # days-stale, like the May commands
            args={"from_ccy": "CHF", "to_ccy": "USD", "amount": 5000},
        ),
        state_dir,
    )

    n = process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert n == 2
    assert broker.submitted == []
    errors = [m for level, m in alerts.messages if level == "error"]
    assert len(errors) == 2
    assert all("expired" in m for m in errors)


def test_ttl_gate_allows_fresh_order_commands(state_dir, risk_manager) -> None:
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    submit(Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 10}), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert len(broker.submitted) == 1


def test_ttl_gate_fails_closed_on_missing_timestamp(state_dir, risk_manager) -> None:
    """An order command whose requested_at is unparseable cannot be proven
    fresh, so it must be refused."""
    import dataclasses

    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    cmd = dataclasses.replace(
        Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 10}),
        requested_at="",
    )
    submit(cmd, state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert broker.submitted == []
    errors = [m for level, m in alerts.messages if level == "error"]
    assert len(errors) == 1 and "expired" in errors[0]


def test_ttl_gate_exempts_non_order_commands(state_dir, risk_manager) -> None:
    """Recovery actions (refresh, reconnect) never expire — same exemption
    as the halt gate."""
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    submit(_aged(CommandType.REFRESH_DATA, minutes_old=10_000), state_dir)
    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    ttl_errors = [m for level, m in alerts.messages if level == "error" and "expired" in m]
    assert ttl_errors == []
