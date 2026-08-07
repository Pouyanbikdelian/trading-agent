"""`/refresh` must actually refresh something.

It used to return "refresh queued; runner will pick it up on next cycle"
and drop the request on the floor. Nothing picked it up — the parquet
cache is topped up by an independent scheduled job at 21:40 UTC that
runs whether or not anyone asked. So the command reported success and
changed nothing, and an operator refreshing before a manual `/cycle` got
stale data and a green tick.

Fifth instance of the same shape found on 2026-08-07: a control whose
confirmation message and behaviour had drifted apart.
"""

from __future__ import annotations

import types

from trading.runtime.command_processor import REFRESH_FLAG, _h_refresh_data
from trading.runtime.commands import Command, CommandType


def _cmd() -> Command:
    return Command.new(CommandType.REFRESH_DATA, args={}, requested_by="test")


def _patch_state_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "trading.core.config.settings",
        types.SimpleNamespace(state_dir=tmp_path),
    )


def test_it_leaves_a_flag_the_runner_can_see(monkeypatch, tmp_path) -> None:
    _patch_state_dir(monkeypatch, tmp_path)

    _h_refresh_data(_cmd(), None)

    assert (tmp_path / REFRESH_FLAG).exists()


def test_the_reply_no_longer_claims_the_cycle_will_do_it(monkeypatch, tmp_path) -> None:
    """The old wording pointed at a mechanism that did not exist."""
    _patch_state_dir(monkeypatch, tmp_path)

    out = _h_refresh_data(_cmd(), None)

    assert "next cycle" not in out["note"]
    assert "runner" in out["note"]


def test_it_does_not_fetch_inline(monkeypatch, tmp_path) -> None:
    """This runs inside the 5s command-processor job at max_instances=1.
    A full universe pass takes ~2 minutes; doing it here would block
    /halt and /flatten behind a data refresh."""
    _patch_state_dir(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("fetched inline — this blocks the command loop")

    monkeypatch.setattr("trading.data.cache.ParquetCache.__init__", boom)

    _h_refresh_data(_cmd(), None)


def test_repeated_requests_collapse_to_one_flag(monkeypatch, tmp_path) -> None:
    _patch_state_dir(monkeypatch, tmp_path)

    _h_refresh_data(_cmd(), None)
    _h_refresh_data(_cmd(), None)

    assert len(list(tmp_path.glob("*.flag"))) == 1


def test_the_flag_records_when_it_was_asked_for(monkeypatch, tmp_path) -> None:
    _patch_state_dir(monkeypatch, tmp_path)

    _h_refresh_data(_cmd(), None)

    body = (tmp_path / REFRESH_FLAG).read_text()
    assert body.startswith("20") and "T" in body


def test_the_runner_consumes_the_flag_and_schedules_a_job(monkeypatch, tmp_path) -> None:
    """End-to-end on the runner side: flag present -> one-off job added,
    flag cleared so it fires once rather than every 5 seconds."""
    import asyncio

    from trading.runner.runner import Runner

    scheduled: list[str] = []

    class _Sched:
        def add_job(self, fn, trigger, *, id, replace_existing, max_instances):
            scheduled.append(id)

    runner = object.__new__(Runner)
    runner._scheduler = _Sched()
    runner.broker = None
    runner.alerts = types.SimpleNamespace()
    runner.cycle = types.SimpleNamespace(risk_manager=None)

    monkeypatch.setattr("trading.runner.runner.settings", types.SimpleNamespace(state_dir=tmp_path))
    monkeypatch.setattr(
        "trading.runtime.command_processor.process_pending",
        lambda *a, **k: 0,
    )
    (tmp_path / REFRESH_FLAG).write_text("2026-08-07T21:00:00+00:00")

    asyncio.run(Runner._process_pending_commands(runner))

    assert scheduled == ["price_cache_refresh_now"]
    assert not (tmp_path / REFRESH_FLAG).exists()


def test_no_flag_schedules_nothing(monkeypatch, tmp_path) -> None:
    import asyncio

    from trading.runner.runner import Runner

    scheduled: list[str] = []

    class _Sched:
        def add_job(self, fn, trigger, *, id, replace_existing, max_instances):
            scheduled.append(id)

    runner = object.__new__(Runner)
    runner._scheduler = _Sched()
    runner.broker = None
    runner.alerts = types.SimpleNamespace()
    runner.cycle = types.SimpleNamespace(risk_manager=None)

    monkeypatch.setattr("trading.runner.runner.settings", types.SimpleNamespace(state_dir=tmp_path))
    monkeypatch.setattr(
        "trading.runtime.command_processor.process_pending",
        lambda *a, **k: 0,
    )

    asyncio.run(Runner._process_pending_commands(runner))

    assert scheduled == []
