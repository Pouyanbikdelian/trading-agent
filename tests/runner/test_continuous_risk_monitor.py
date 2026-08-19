"""Runner integration for the minute-level live account risk monitor."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import trading.runner.runner as runner_module
from trading.core.config import settings
from trading.core.types import AccountSnapshot
from trading.risk import RiskLimits, RiskManager
from trading.runner import Runner
from trading.runtime.nyse_session import nyse_session_on


class _Alerts:
    def __init__(self) -> None:
        self.critical_messages: list[str] = []
        self.warning_messages: list[str] = []

    def critical(self, message: str) -> None:
        self.critical_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)


def _snapshot(ts, equity: float) -> AccountSnapshot:
    return AccountSnapshot(
        ts=ts,
        cash=equity,
        equity=equity,
        base_currency="CHF",
    )


def _live_runner(tmp_path: Path, monkeypatch) -> tuple[Runner, RiskManager, _Alerts]:
    live = settings.model_copy(
        update={
            "trading_env": "live",
            "allow_live_trading": True,
            "state_dir": tmp_path,
            "data_dir": tmp_path / "data",
            "log_dir": tmp_path / "logs",
        }
    )
    monkeypatch.setattr(runner_module, "settings", live)
    manager = RiskManager(
        RiskLimits(max_daily_loss_pct=0.02, max_drawdown_pct=0.50),
        halt_state_path=tmp_path / "halt.json",
    )
    alerts = _Alerts()
    runner = Runner.__new__(Runner)
    runner.cycle = type("Cycle", (), {"risk_manager": manager})()
    runner.alerts = alerts
    runner._last_risk_monitor_reject_reason = None
    return runner, manager, alerts


def test_monitor_captures_open_then_halts_once_on_real_loss(tmp_path: Path, monkeypatch) -> None:
    runner, manager, alerts = _live_runner(tmp_path, monkeypatch)
    session = nyse_session_on(date(2026, 3, 9))
    assert session is not None

    opening = session.market_open + timedelta(minutes=1)
    runner._monitor_live_account_risk(
        _snapshot(opening, 100_000.0),
        liveness={"ready": True},
        observed_at=opening,
    )
    assert manager.state.daily_baseline_session == session.label

    loss_time = session.market_open + timedelta(hours=2)
    runner._monitor_live_account_risk(
        _snapshot(loss_time, 97_000.0),
        liveness={"ready": True},
        observed_at=loss_time,
    )
    assert manager.is_halted()
    assert len(alerts.critical_messages) == 1
    assert "continuous account monitor" in alerts.critical_messages[0]

    # A per-minute observer reports the transition, not the same halt forever.
    runner._monitor_live_account_risk(
        _snapshot(loss_time + timedelta(minutes=1), 96_900.0),
        liveness={"ready": True},
        observed_at=loss_time + timedelta(minutes=1),
    )
    assert len(alerts.critical_messages) == 1


def test_monitor_never_uses_a_red_liveness_probe_as_risk_input(tmp_path: Path, monkeypatch) -> None:
    runner, manager, alerts = _live_runner(tmp_path, monkeypatch)
    session = nyse_session_on(date(2026, 3, 9))
    assert session is not None
    opening = session.market_open + timedelta(minutes=1)

    runner._monitor_live_account_risk(
        _snapshot(opening, 100_000.0),
        liveness={"ready": False, "detail": "2FA required"},
        observed_at=opening,
    )

    assert manager.state.daily_baseline_session is None
    assert not manager.is_halted()
    assert not alerts.critical_messages and not alerts.warning_messages


def test_missed_open_blocks_execution_once_without_persisting_halt(
    tmp_path: Path, monkeypatch
) -> None:
    runner, manager, alerts = _live_runner(tmp_path, monkeypatch)
    session = nyse_session_on(date(2026, 3, 9))
    assert session is not None
    noon = session.market_open + timedelta(hours=3)

    runner._monitor_live_account_risk(
        _snapshot(noon, 99_000.0),
        liveness={"ready": True},
        observed_at=noon,
    )
    runner._monitor_live_account_risk(
        _snapshot(noon + timedelta(minutes=1), 99_000.0),
        liveness={"ready": True},
        observed_at=noon + timedelta(minutes=1),
    )

    assert not manager.is_halted()
    assert len(alerts.warning_messages) == 1
    assert "baseline" in alerts.warning_messages[0].lower()
