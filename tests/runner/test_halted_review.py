"""Runner routing tests for review-only off-cycle requests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import trading.runner.runner as runner_module
from trading.core.config import settings
from trading.runner import Runner, RunnerConfig
from trading.runner.alerts import TelegramAlerts


class _Alerts(TelegramAlerts):
    def __init__(self) -> None:
        super().__init__(token=None, chat_id=None, enabled=False)

    def info(self, _msg: str) -> None: ...

    def warning(self, _msg: str) -> None: ...

    def error(self, _msg: str) -> None: ...

    def critical(self, _msg: str) -> None: ...


def _runner(tmp_path: Path) -> Runner:
    runner = Runner.__new__(Runner)
    runner.config = RunnerConfig(universe="sp500", strategies=["top_k_momentum"])
    runner.alerts = _Alerts()
    runner._error_counter_path = tmp_path / "consecutive_errors.json"
    runner._consecutive_errors = 0
    runner._last_success_ts = None
    runner._last_cycle_start_ts = None
    return runner


def test_review_trigger_routes_to_run_review_not_normal_cycle(tmp_path: Path, monkeypatch) -> None:
    """A bot request marked ``mode=review`` must carry through both seams."""
    test_settings = settings.model_copy(update={"state_dir": tmp_path})
    monkeypatch.setattr("trading.core.config.settings", test_settings)
    monkeypatch.setattr(runner_module, "settings", test_settings)
    runner = _runner(tmp_path)
    calls: list[str] = []

    class _Cycle:
        def run_review(self) -> SimpleNamespace:
            calls.append("review")
            return SimpleNamespace(
                status="halted_review", error=None, orders_submitted=0, fills_received=0
            )

        def run_cycle(self) -> SimpleNamespace:
            calls.append("execute")
            raise AssertionError("review trigger reached the executable cycle")

    runner.cycle = _Cycle()
    (tmp_path / "trigger_now.flag").write_text(json.dumps({"mode": "review"}))

    asyncio.run(runner._check_trigger_flag())

    assert calls == ["review"]
    assert not (tmp_path / "trigger_now.flag").exists()


def _execute_runner(tmp_path: Path, monkeypatch, *, pm_age_h: float | None):
    from datetime import datetime, timedelta, timezone

    test_settings = settings.model_copy(update={"state_dir": tmp_path})
    monkeypatch.setattr("trading.core.config.settings", test_settings)
    monkeypatch.setattr(runner_module, "settings", test_settings)
    monkeypatch.setenv("AGENTS_ENABLED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    if pm_age_h is not None:
        (tmp_path / "agent_pm").mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc) - timedelta(hours=pm_age_h)
        (tmp_path / "agent_pm" / "last_run.json").write_text(
            json.dumps({"ok": True, "ts": ts.isoformat(), "weights": {"AAPL": 0.1}})
        )
    runner = _runner(tmp_path)
    calls: list[str] = []

    async def fake_pm() -> None:
        calls.append("pm")

    async def fake_cycle(*, review_only: bool = False) -> None:
        calls.append("review" if review_only else "execute")

    runner._run_agent_pm_async = fake_pm  # type: ignore[method-assign]
    runner._run_cycle_async = fake_cycle  # type: ignore[method-assign]
    return runner, calls


@pytest.mark.parametrize("age", [None, 30.0])
def test_a_manual_cycle_runs_the_pm_first_when_its_decision_is_stale(
    tmp_path: Path, monkeypatch, age
) -> None:
    """Every-second-Friday schedule: most cycles are manual, on days the PM
    has not decided. Without this the bridge refuses a 4-day-old decision
    and the simulation never sees the operator's recycles."""
    runner, calls = _execute_runner(tmp_path, monkeypatch, pm_age_h=age)
    (tmp_path / "trigger_now.flag").write_text(json.dumps({"mode": "execute"}))
    asyncio.run(runner._check_trigger_flag())
    assert calls == ["pm", "execute"]


def test_a_fresh_pm_decision_is_reused_not_repeated(tmp_path: Path, monkeypatch) -> None:
    """The scheduled Friday PM ran 45 min before; a /cycle after it must not
    rebalance the simulation twice."""
    runner, calls = _execute_runner(tmp_path, monkeypatch, pm_age_h=1.0)
    (tmp_path / "trigger_now.flag").write_text(json.dumps({"mode": "execute"}))
    asyncio.run(runner._check_trigger_flag())
    assert calls == ["execute"]


def test_a_review_never_runs_the_pm(tmp_path: Path, monkeypatch) -> None:
    runner, calls = _execute_runner(tmp_path, monkeypatch, pm_age_h=None)
    (tmp_path / "trigger_now.flag").write_text(json.dumps({"mode": "review"}))
    asyncio.run(runner._check_trigger_flag())
    assert calls == ["review"]
