"""Tests for the command processor's safety gates.

The focus here is the halt-gate: order-submitting commands (BUY, SELL,
CLOSE, FLATTEN, FX_CONVERT) MUST be refused while the risk manager is
halted. Otherwise, an operator who types /halt and then /buy still trades.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    Position,
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
        self.connected = True

    def submit_order(self, order: Order) -> None:
        self.submitted.append(order)

    def get_positions(self) -> list[Position]:
        return list(self.positions)


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


def test_halt_gate_blocks_all_order_command_types(state_dir, risk_manager) -> None:
    """BUY, SELL, CLOSE, FLATTEN, FX_CONVERT must all be refused while halted."""
    broker = _FakeBroker()
    alerts = _RecordingAlerts()
    risk_manager.halt("safety test")

    submit(Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 1}), state_dir)
    submit(Command.new(CommandType.SELL, args={"symbol": "AAPL", "qty": 1}), state_dir)
    submit(Command.new(CommandType.CLOSE, args={"symbol": "AAPL"}), state_dir)
    submit(Command.new(CommandType.FLATTEN), state_dir)
    submit(
        Command.new(
            CommandType.FX_CONVERT,
            args={"from_ccy": "CHF", "to_ccy": "USD", "amount": 25000},
        ),
        state_dir,
    )

    process_pending(broker, state_dir, alerts, risk_manager=risk_manager)
    assert broker.submitted == []
    assert len([m for level, m in alerts.messages if level == "error"]) == 5


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
