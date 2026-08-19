"""Runner routing tests for review-only off-cycle requests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

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
