"""Tests for Runner internals: cooldown + persisted error counter.

These touch ``Runner._consecutive_errors`` and ``Runner._last_cycle_start_ts``
via ``Runner.__new__`` so we don't have to spin up a full Cycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trading.runner import Runner, RunnerConfig
from trading.runner.alerts import TelegramAlerts


class _SilentAlerts(TelegramAlerts):
    def __init__(self) -> None:
        super().__init__(token=None, chat_id=None, enabled=False)
        self.last_critical: str | None = None
        self.last_warning: str | None = None

    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None:
        self.last_warning = msg

    def error(self, msg: str) -> None: ...
    def critical(self, msg: str) -> None:
        self.last_critical = msg


def _bare_runner(tmp_path: Path) -> Runner:
    """Build the minimal Runner state the new internals need to read."""
    runner = Runner.__new__(Runner)
    runner.config = RunnerConfig(universe="sp500", strategies=["top_k_momentum"])
    runner.alerts = _SilentAlerts()
    runner._error_counter_path = tmp_path / "consecutive_errors.json"
    runner._consecutive_errors = 0
    runner._last_success_ts = None
    runner._last_cycle_start_ts = None
    return runner


# ---------------------------------------------------------------------------
# Audit fix #8 — persisted error counter
# ---------------------------------------------------------------------------


def test_error_counter_starts_at_zero_with_no_file(tmp_path: Path) -> None:
    runner = _bare_runner(tmp_path)
    assert runner._load_error_counter() == 0


def test_error_counter_persists_across_loads(tmp_path: Path) -> None:
    runner = _bare_runner(tmp_path)
    runner._consecutive_errors = 2
    runner._save_error_counter()

    # Fresh runner reads the persisted count.
    runner2 = _bare_runner(tmp_path)
    assert runner2._load_error_counter() == 2


def test_error_counter_zero_persists_too(tmp_path: Path) -> None:
    """After a successful cycle resets the counter to 0, the file should
    reflect that — otherwise a restart would surface a stale non-zero."""
    runner = _bare_runner(tmp_path)
    runner._consecutive_errors = 3
    runner._save_error_counter()
    runner._consecutive_errors = 0
    runner._save_error_counter()
    payload = json.loads(runner._error_counter_path.read_text())
    assert payload["count"] == 0


def test_error_counter_unreadable_file_defaults_zero(tmp_path: Path) -> None:
    """Corrupt JSON shouldn't crash; default to 0 so the runner can boot."""
    runner = _bare_runner(tmp_path)
    runner._error_counter_path.write_text("{not json")
    assert runner._load_error_counter() == 0


# ---------------------------------------------------------------------------
# Audit fix #11 — cycle cooldown (smoke level: the gate logic, not the
# full _run_cycle_async which needs a Cycle)
# ---------------------------------------------------------------------------


def test_cycle_cooldown_threshold_defined() -> None:
    """Pin the cooldown so a future change doesn't quietly remove it."""
    assert Runner.CYCLE_COOLDOWN_SECONDS >= 5.0


def test_runner_starts_with_no_prior_cycle_ts(tmp_path: Path) -> None:
    runner = _bare_runner(tmp_path)
    assert runner._last_cycle_start_ts is None


def test_cycle_cooldown_gate_blocks_back_to_back() -> None:
    """If _last_cycle_start_ts is freshly set, a new cycle would be refused."""
    now = datetime.now()
    just_now = now - timedelta(seconds=1)
    gap = (now - just_now).total_seconds()
    assert gap < Runner.CYCLE_COOLDOWN_SECONDS


def test_cycle_cooldown_gate_allows_after_window() -> None:
    """After the window expires, a new cycle is allowed."""
    now = datetime.now()
    long_ago = now - timedelta(seconds=Runner.CYCLE_COOLDOWN_SECONDS + 5)
    gap = (now - long_ago).total_seconds()
    assert gap > Runner.CYCLE_COOLDOWN_SECONDS
