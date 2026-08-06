"""The single execution lock.

External review 2026-07-15, open until 2026-08-06. Four paths could reach
the broker — the scheduled cycle, the on-demand cycle, the approval flow
and manual commands — sharing nothing but per-job locks, which prevent a
job racing ITSELF and do nothing about two different jobs racing each
other. The 2026-07-14/15 order-stacking incident (two names taken short
on paper) came out of that gap.

The property that matters most here is NOT mutual exclusion — flock gives
that for free. It is that the lock **cannot trap the operator**. A lock
that can deadlock ``/flatten`` is worse than no lock at all, so every
acquisition has a deadline and a crashed holder releases automatically.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from trading.core.exec_lock import (
    ExecutionBusyError,
    current_holder,
    execution_lock,
    lock_path,
)


class TestMutualExclusion:
    @staticmethod
    def _take(tmp_path: Path, holder: str, timeout: float = 0.1) -> None:
        """Acquire and immediately release — raises ExecutionBusyError if the
        lock is held elsewhere."""
        with execution_lock(tmp_path, holder=holder, timeout=timeout):
            pass

    def test_second_holder_is_refused_not_queued_forever(self, tmp_path: Path) -> None:
        with (
            execution_lock(tmp_path, holder="cycle", timeout=0.1),
            pytest.raises(ExecutionBusyError),
        ):
            self._take(tmp_path, "command:/buy")

    def test_the_refusal_names_the_holder(self, tmp_path: Path) -> None:
        """'Try again' is only useful if you know what to wait for."""
        with (
            execution_lock(tmp_path, holder="cycle", timeout=0.1),
            pytest.raises(ExecutionBusyError) as ei,
        ):
            self._take(tmp_path, "command:/close")
        assert "cycle" in ei.value.holder
        assert ei.value.waited_s >= 0.1

    def test_released_on_exit(self, tmp_path: Path) -> None:
        with execution_lock(tmp_path, holder="first", timeout=0.1):
            pass
        with execution_lock(tmp_path, holder="second", timeout=0.1):
            pass  # must not raise

    def test_released_even_when_the_body_raises(self, tmp_path: Path) -> None:
        """An exception mid-submit must not wedge every later order."""
        with pytest.raises(ValueError), execution_lock(tmp_path, holder="boom", timeout=0.1):
            raise ValueError("submit blew up")
        with execution_lock(tmp_path, holder="after", timeout=0.1):
            pass


class TestDiagnostics:
    def test_holder_metadata_is_written(self, tmp_path: Path) -> None:
        with execution_lock(tmp_path, holder="cycle", timeout=0.1):
            raw = json.loads(lock_path(tmp_path).read_text())
        assert raw["holder"] == "cycle"
        assert raw["pid"] == os.getpid()
        assert raw["since"]

    def test_metadata_describes_the_current_holder_not_the_last(self, tmp_path: Path) -> None:
        with execution_lock(tmp_path, holder="cycle", timeout=0.1):
            pass
        with execution_lock(tmp_path, holder="force_flatten", timeout=0.1):
            assert "force_flatten" in current_holder(tmp_path)

    def test_current_holder_never_raises(self, tmp_path: Path) -> None:
        assert current_holder(tmp_path / "nope") == "unknown"
        lock_path(tmp_path).write_text("not json{{{")
        assert current_holder(tmp_path) == "unknown"


def _hold_then_exit(path: str, seconds: float) -> None:
    """Child process: take the lock, then die WITHOUT releasing it."""
    with execution_lock(Path(path), holder="child", timeout=5.0):
        time.sleep(seconds)
        os._exit(1)  # hard kill — no finally, no unlock


class TestCannotTrapTheOperator:
    """The properties that make this lock safe to put in front of
    /flatten. Each of these failing means a human cannot reduce risk."""

    def test_a_crashed_holder_releases_the_lock(self, tmp_path: Path) -> None:
        """flock is released by the kernel when the fd closes, so a
        runner that dies mid-submit cannot lock out the operator. This is
        the whole reason for a file lock over a sentinel file."""
        ctx = mp.get_context("fork")
        p = ctx.Process(target=_hold_then_exit, args=(str(tmp_path), 0.2))
        p.start()
        time.sleep(0.05)  # let the child take it

        with (
            pytest.raises(ExecutionBusyError),
            execution_lock(tmp_path, holder="operator", timeout=0.05),
        ):
            pass

        p.join(timeout=5)
        # The child exited hard; the lock must be free anyway.
        with execution_lock(tmp_path, holder="operator", timeout=1.0):
            pass

    def test_waiting_succeeds_once_the_holder_finishes(self, tmp_path: Path) -> None:
        """Contention should mean 'wait a moment', not 'fail'."""
        ctx = mp.get_context("fork")
        p = ctx.Process(target=_hold_then_exit, args=(str(tmp_path), 0.3))
        p.start()
        time.sleep(0.05)

        started = time.monotonic()
        with execution_lock(tmp_path, holder="operator", timeout=5.0):
            waited = time.monotonic() - started
        p.join(timeout=5)
        assert 0.1 < waited < 5.0

    def test_zero_timeout_still_works_when_free(self, tmp_path: Path) -> None:
        with execution_lock(tmp_path, holder="x", timeout=0.0):
            pass


class TestScope:
    """Only order-submitting commands take the lock. /reconnect must stay
    available while a cycle runs — it is how a wedged broker is
    recovered, and gating it behind the thing that wedged would be
    exactly backwards."""

    def test_order_commands_are_gated_and_recovery_commands_are_not(self) -> None:
        from trading.runtime.command_processor import _ORDER_SUBMITTING_COMMANDS
        from trading.runtime.commands import CommandType

        assert CommandType.BUY in _ORDER_SUBMITTING_COMMANDS
        assert CommandType.FLATTEN in _ORDER_SUBMITTING_COMMANDS
        assert CommandType.RECONNECT_BROKER not in _ORDER_SUBMITTING_COMMANDS
