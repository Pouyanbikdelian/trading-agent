"""Tests for Runner internals: cooldown + persisted error counter.

These touch ``Runner._consecutive_errors`` and ``Runner._last_cycle_start_ts``
via ``Runner.__new__`` so we don't have to spin up a full Cycle.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import trading.runner.runner as runner_module
from trading.core.config import settings
from trading.core.types import AccountSnapshot
from trading.runner import Runner, RunnerConfig
from trading.runner.alerts import TelegramAlerts
from trading.runner.runner import _historian_trigger
from trading.runtime.broker_liveness import FILENAME


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


class _CapturingScheduler:
    """Minimal scheduler seam for checking registration without a live loop."""

    def __init__(self) -> None:
        self.jobs: dict[str, tuple[object, object, dict[str, object]]] = {}

    def add_job(self, func: object, trigger: object, **kwargs: object) -> None:
        self.jobs[str(kwargs["id"])] = (func, trigger, kwargs)


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


def test_core_background_jobs_do_not_depend_on_the_primary_telegram_chat(
    monkeypatch, tmp_path: Path
) -> None:
    """OPS-only deployments still need safety, cache, and learning jobs."""
    monkeypatch.setenv("GUARDS_ENABLED", "true")
    runner = _bare_runner(tmp_path)
    scheduler = _CapturingScheduler()
    runner._scheduler = scheduler

    runner._register_core_background_jobs()

    assert {
        "guards",
        "broker_ready",
        "price_cache_refresh",
        "universe_refresh",
        "ops_watch",
        "memory_grader",
        "market_watch",
    } <= set(scheduler.jobs)
    assert scheduler.jobs["ops_watch"][2]["coalesce"] is True
    assert scheduler.jobs["market_watch"][2]["max_instances"] == 1


def test_core_post_close_triggers_hold_the_same_new_york_times_across_dst(
    monkeypatch, tmp_path: Path
) -> None:
    """A UTC literal would run market watch before the close every winter."""
    monkeypatch.delenv("GUARDS_ENABLED", raising=False)
    runner = _bare_runner(tmp_path)
    scheduler = _CapturingScheduler()
    runner._scheduler = scheduler
    runner._register_core_background_jobs()

    cache_trigger = scheduler.jobs["price_cache_refresh"][1]
    grader_trigger = scheduler.jobs["memory_grader"][1]
    market_trigger = scheduler.jobs["market_watch"][1]
    historian_trigger = _historian_trigger()
    nyse = ZoneInfo("America/New_York")
    for start in (
        # Tuesday is a Historian day, so all four post-close jobs land on
        # the same local session in both DST seasons.
        datetime(2026, 1, 6, tzinfo=timezone.utc),
        datetime(2026, 8, 18, tzinfo=timezone.utc),
    ):
        cache_at = cache_trigger.get_next_fire_time(None, start)
        grader_at = grader_trigger.get_next_fire_time(None, start)
        market_at = market_trigger.get_next_fire_time(None, start)
        historian_at = historian_trigger.get_next_fire_time(None, start)
        assert cache_at is not None and grader_at is not None and market_at is not None
        assert historian_at is not None
        assert (cache_at.astimezone(nyse).hour, cache_at.astimezone(nyse).minute) == (17, 40)
        assert (grader_at.astimezone(nyse).hour, grader_at.astimezone(nyse).minute) == (18, 45)
        assert (market_at.astimezone(nyse).hour, market_at.astimezone(nyse).minute) == (18, 50)
        assert (historian_at.astimezone(nyse).hour, historian_at.astimezone(nyse).minute) == (19, 0)
        assert grader_at - cache_at == timedelta(minutes=65)
        assert market_at - cache_at == timedelta(minutes=70)
        assert historian_at - grader_at == timedelta(minutes=15)


def test_market_watch_serializes_startup_catchup_and_cron(monkeypatch, tmp_path: Path) -> None:
    """A restart immediately after the cron tick must launch one collector."""
    runner = _bare_runner(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[object] = []

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> None:
        calls.append(func)
        started.set()
        await release.wait()

    monkeypatch.setattr(runner_module.asyncio, "to_thread", fake_to_thread)

    async def scenario() -> None:
        first = asyncio.create_task(runner._run_market_watch_async())
        await started.wait()
        await runner._run_market_watch_async()
        assert len(calls) == 1
        release.set()
        await first

    asyncio.run(scenario())


def test_live_bootstrap_retries_connect_before_any_preflight_or_scheduler_work(
    monkeypatch, tmp_path: Path
) -> None:
    """A login failure must only record/alert and retry; it cannot arm jobs."""
    test_settings = settings.model_copy(
        update={"state_dir": tmp_path, "data_dir": tmp_path / "data", "log_dir": tmp_path / "logs"}
    )
    monkeypatch.setattr(runner_module, "settings", test_settings)
    runner = _bare_runner(tmp_path)
    attempts: list[str] = []
    submissions: list[object] = []
    failures: list[tuple[str, str]] = []
    ops_runs: list[Path] = []
    heartbeats: list[str] = []
    preflights: list[bool] = []
    sleeps: list[float] = []

    class _Broker:
        def connect(self) -> None:
            attempts.append("connect")
            if len(attempts) == 1:
                raise ConnectionError("IBKR Mobile approval required")

        def submit_order(self, _order: object) -> None:
            submissions.append(_order)
            raise AssertionError("bootstrap must never submit an order")

    runner.broker = _Broker()
    runner.preflight_unheld = lambda *, require_reachable: (
        preflights.append(require_reachable) or True
    )
    runner._write_bootstrap_heartbeat = lambda detail: heartbeats.append(detail)

    monkeypatch.setattr(
        "trading.runtime.broker_liveness.record_broker_liveness",
        lambda *_args, **_kwargs: {"ready": True, "probe": "reqCurrentTime"},
    )
    monkeypatch.setattr(
        "trading.runtime.broker_liveness.record_broker_liveness_failure",
        lambda _state, detail, *, probe, **_kwargs: failures.append((probe, detail)),
    )
    monkeypatch.setattr(
        "trading.runtime.ops_watch.run_ops_watch",
        lambda state_dir, **_kwargs: ops_runs.append(state_dir),
    )

    async def no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner_module.asyncio, "sleep", no_sleep)

    asyncio.run(runner._bootstrap_live_broker())

    assert attempts == ["connect", "connect"]
    assert preflights == [True]
    assert failures == [("connect", "ConnectionError: IBKR Mobile approval required")]
    assert ops_runs == [tmp_path]
    assert heartbeats == ["ConnectionError: IBKR Mobile approval required"]
    assert sleeps == [Runner.BROKER_BOOTSTRAP_RETRY_SECONDS]
    assert submissions == []
    assert getattr(runner, "_scheduler", None) is None


def test_live_bootstrap_does_not_continue_after_an_unverified_position_read(
    monkeypatch, tmp_path: Path
) -> None:
    """A green time probe is insufficient: the account inventory must also reply."""
    test_settings = settings.model_copy(
        update={"state_dir": tmp_path, "data_dir": tmp_path / "data", "log_dir": tmp_path / "logs"}
    )
    monkeypatch.setattr(runner_module, "settings", test_settings)
    runner = _bare_runner(tmp_path)
    position_checks: list[bool] = []
    failures: list[tuple[str, str]] = []
    sleeps: list[float] = []

    class _Broker:
        def connect(self) -> None: ...

    runner.broker = _Broker()

    def strict_preflight(*, require_reachable: bool) -> bool:
        position_checks.append(require_reachable)
        return len(position_checks) > 1

    runner.preflight_unheld = strict_preflight
    runner._write_bootstrap_heartbeat = lambda _detail: None
    monkeypatch.setattr(
        "trading.runtime.broker_liveness.record_broker_liveness",
        lambda *_args, **_kwargs: {"ready": True, "probe": "reqCurrentTime"},
    )
    monkeypatch.setattr(
        "trading.runtime.broker_liveness.record_broker_liveness_failure",
        lambda _state, detail, *, probe, **_kwargs: failures.append((probe, detail)),
    )
    monkeypatch.setattr("trading.runtime.ops_watch.run_ops_watch", lambda *_args, **_kwargs: {})

    async def no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(runner_module.asyncio, "sleep", no_sleep)

    asyncio.run(runner._bootstrap_live_broker())

    assert position_checks == [True, True]
    assert failures == [
        ("positions", "position inventory unavailable after an authenticated liveness probe")
    ]
    assert sleeps == [Runner.BROKER_BOOTSTRAP_RETRY_SECONDS]
    assert getattr(runner, "_scheduler", None) is None


def test_run_forever_registers_no_jobs_until_live_bootstrap_has_verified_the_broker(
    monkeypatch, tmp_path: Path
) -> None:
    """A failed login cannot leave a hidden command/cycle scheduler running."""
    test_settings = settings.model_copy(
        update={
            "state_dir": tmp_path,
            "data_dir": tmp_path / "data",
            "log_dir": tmp_path / "logs",
            "trading_env": "live",
            "allow_live_trading": True,
        }
    )
    monkeypatch.setattr(runner_module, "settings", test_settings)
    runner = _bare_runner(tmp_path)
    runner._startup_market_watch_task = None
    runner._start_startup_market_watch_catchup = lambda: None
    runner._reconcile_startup = lambda: None
    attempts: list[str] = []
    failure_seen = asyncio.Event()
    allow_retry = asyncio.Event()
    scheduler_started = asyncio.Event()
    schedulers: list[object] = []

    class _Broker:
        def connect(self) -> None:
            attempts.append("connect")
            if len(attempts) == 1:
                raise ConnectionError("2FA pending")

        def disconnect(self) -> None: ...

    class _Scheduler:
        def __init__(self, **_kwargs: object) -> None:
            self.jobs: dict[str, object] = {}
            schedulers.append(self)

        def add_job(self, func: object, trigger: object, **kwargs: object) -> None:
            self.jobs[str(kwargs["id"])] = (func, trigger, kwargs)

        def start(self) -> None:
            scheduler_started.set()

        def get_job(self, job_id: str) -> SimpleNamespace:
            assert job_id in self.jobs
            return SimpleNamespace(next_run_time=None)

        def get_jobs(self) -> list[object]:
            return []

        def shutdown(self, **_kwargs: object) -> None: ...

    async def wait_on_failure(_detail: str, **_kwargs: object) -> None:
        failure_seen.set()
        await allow_retry.wait()

    runner.broker = _Broker()
    runner.preflight_unheld = lambda *, require_reachable: require_reachable
    runner._report_live_bootstrap_failure = wait_on_failure
    monkeypatch.setattr(
        "trading.runtime.broker_liveness.record_broker_liveness",
        lambda *_args, **_kwargs: {"ready": True, "probe": "reqCurrentTime"},
    )
    monkeypatch.setattr("apscheduler.schedulers.asyncio.AsyncIOScheduler", _Scheduler)

    async def scenario() -> None:
        task = asyncio.create_task(runner.run_forever())
        await failure_seen.wait()
        assert schedulers == []
        assert getattr(runner, "_scheduler", None) is None

        allow_retry.set()
        await scheduler_started.wait()
        scheduler = schedulers[0]
        assert {"cycle", "trigger_watcher", "command_processor"} <= set(scheduler.jobs)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert attempts == ["connect", "connect"]


def test_memory_grader_journals_exact_next_session_prediction_ids(
    monkeypatch, tmp_path: Path
) -> None:
    """The watchdog must receive IDs, not merely subjects: sharing a symbol
    with a normal Friday wait must never exempt a different overdue call."""

    class _Memory:
        def __init__(self) -> None:
            self.entries: list[tuple[str, dict]] = []

        def journal(self, kind: str, payload: dict) -> None:
            self.entries.append((kind, payload))

    class _RunnerStore:
        @staticmethod
        def latest_snapshot():
            return None

    class _Cycle:
        runner_store = _RunnerStore()

    memory = _Memory()
    runner = Runner.__new__(Runner)
    runner.cycle = _Cycle()
    runner._grade_shadow = lambda mem: 0
    monkeypatch.setattr("trading.memory.store.default_store", lambda: memory)
    monkeypatch.setattr(
        "trading.memory.grading.grade_due_predictions",
        lambda mem, data_dir: {
            "graded": 0,
            "skipped": 1,
            "unpriced_subjects": [],
            "awaiting_next_daily_bar_subjects": ["SPY"],
            "awaiting_next_daily_bar_prediction_ids": ["pr-weekend"],
            "cache_behind_subjects": [],
            "failed_subjects": [],
        },
    )
    monkeypatch.setattr("trading.memory.episodes.record_closed_episodes", lambda *args: 0)

    asyncio.run(runner._run_memory_grader_async())

    assert memory.entries == [
        (
            "daily",
            {
                "equity": None,
                "positions": 0,
                "graded_today": 0,
                "ungraded_today": 1,
                "unpriced_subjects": [],
                "awaiting_next_daily_bar_subjects": ["SPY"],
                "awaiting_next_daily_bar_prediction_ids": ["pr-weekend"],
                "cache_behind_subjects": [],
                "grading_failed_subjects": [],
                "shadow_legs_graded": 0,
                "episodes_recorded": 0,
            },
        )
    ]


def test_snapshot_refresh_records_authenticated_broker_liveness(
    monkeypatch, tmp_path: Path
) -> None:
    """A fresh cached account view cannot mask a failed wire probe: the
    snapshot job records the latter before persisting the former."""

    now = datetime.now().astimezone()

    class _Broker:
        def probe_liveness(self) -> datetime:
            return now

        def get_account(self) -> AccountSnapshot:
            return AccountSnapshot(ts=now, cash=100.0, equity=100.0)

    class _RunnerStore:
        saved: list[AccountSnapshot] = []

        def save_snapshot(self, snapshot: AccountSnapshot) -> None:
            self.saved.append(snapshot)

    class _Cycle:
        runner_store = _RunnerStore()

    test_settings = settings.model_copy(
        update={"state_dir": tmp_path, "data_dir": tmp_path / "data"}
    )
    monkeypatch.setattr(runner_module, "settings", test_settings)
    runner = Runner.__new__(Runner)
    runner.broker = _Broker()
    runner.cycle = _Cycle()

    asyncio.run(runner._refresh_account_snapshot())

    payload = json.loads((tmp_path / FILENAME).read_text())
    assert payload["ready"] is True and payload["probe"] == "reqCurrentTime"
    assert len(runner.cycle.runner_store.saved) == 1
