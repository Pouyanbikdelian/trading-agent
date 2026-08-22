"""Waiting for a human is not work, and it is not a fault.

The runner aborted a cycle at 300s while the approval card promised the
operator ten minutes to decide. On 2026-08-20 that fired for real: the
basket went out at 18:10, nobody pressed anything, and five minutes later
the desk announced

    ⏱️ cycle aborted after 300s (error #1/3) — likely a wedged IBKR Gateway.

about a gateway that was working perfectly. Three unhurried approvals in a
row would have auto-halted the desk for being read slowly.

The budget now has two parts — what the machine needs, plus whatever the
operator is permitted to take — so a timeout again means something is
actually stuck.
"""

from __future__ import annotations

import pytest

from trading.runner.timeouts import (
    CYCLE_WORK_BUDGET_S,
    ZOMBIE_GUARD_MARGIN_S,
    approval_budget_s,
    cycle_timeout_seconds,
    from_settings,
    zombie_guard_ms,
)


class _Settings:
    def __init__(self, *, require: bool, approval_s: float) -> None:
        self.require_cycle_approval = require
        self.cycle_approval_timeout_s = approval_s


class TestTheBudgetIncludesTheOperator:
    def test_approval_off_is_machine_time_only(self) -> None:
        assert (
            cycle_timeout_seconds(require_cycle_approval=False, cycle_approval_timeout_s=600)
            == CYCLE_WORK_BUDGET_S
        )

    def test_approval_on_adds_the_whole_decision_window(self) -> None:
        """The live configuration: 300s of work + the advertised 10 minutes."""
        assert (
            cycle_timeout_seconds(require_cycle_approval=True, cycle_approval_timeout_s=600)
            == 900.0
        )

    def test_the_timeout_can_never_be_shorter_than_the_promise(self) -> None:
        """The bug in one assertion: whatever the card advertises, the
        runner must still be waiting when that clock runs out."""
        for advertised in (60, 300, 600, 1_800):
            budget = cycle_timeout_seconds(
                require_cycle_approval=True, cycle_approval_timeout_s=advertised
            )
            assert budget > advertised

    def test_an_unset_approval_window_is_not_negative_time(self) -> None:
        assert approval_budget_s(require_cycle_approval=True, cycle_approval_timeout_s=-30) == 0.0


class TestTheZombieGuardStaysInside:
    def test_it_sits_one_margin_before_the_abort(self) -> None:
        """The runner cancels the awaiting coroutine but not the worker
        thread, so a cycle already told it aborted could still reach the
        broker. It has to stand down first, and by a known margin."""
        for require in (True, False):
            timeout = cycle_timeout_seconds(
                require_cycle_approval=require, cycle_approval_timeout_s=600
            )
            guard_s = (
                zombie_guard_ms(require_cycle_approval=require, cycle_approval_timeout_s=600)
                / 1000.0
            )
            assert guard_s == pytest.approx(timeout - ZOMBIE_GUARD_MARGIN_S)
            assert guard_s < timeout

    def test_it_never_goes_negative_on_a_tiny_budget(self) -> None:
        guard = zombie_guard_ms(
            require_cycle_approval=False, cycle_approval_timeout_s=0, work_budget_s=1.0
        )
        assert guard == 0.0


class TestFromSettings:
    def test_it_reads_the_live_pair(self) -> None:
        timeout, guard_ms = from_settings(_Settings(require=True, approval_s=600))

        assert timeout == 900.0
        assert guard_ms == 890_000.0

    def test_a_settings_object_missing_the_fields_falls_back_to_work_only(self) -> None:
        timeout, _ = from_settings(object())

        assert timeout == CYCLE_WORK_BUDGET_S

    def test_a_none_approval_timeout_is_treated_as_zero(self) -> None:
        class _Partial:
            require_cycle_approval = True
            cycle_approval_timeout_s = None

        assert from_settings(_Partial())[0] == CYCLE_WORK_BUDGET_S


class TestNothingHardcodesTheOldNumber:
    def test_the_cycle_no_longer_carries_a_literal_zombie_guard(self) -> None:
        """Both guards used to read 290_000 in the source, which is what
        let them drift away from the runner's own timeout."""
        import inspect

        from trading.runner import cycle as cycle_module

        assert "290_000" not in inspect.getsource(cycle_module)

    def test_the_runner_derives_its_timeout_rather_than_naming_one(self) -> None:
        import inspect

        from trading.runner import runner as runner_module

        source = inspect.getsource(runner_module.Runner._run_cycle_async)
        assert "timeouts" in source
        assert "self.CYCLE_TIMEOUT_SECONDS" not in source


class TestTheAbortSaysWhatActuallyHappened:
    def test_a_pending_approval_is_not_blamed_on_the_gateway(self) -> None:
        """The exact sentence the operator saw was false. Whatever the
        wording becomes, it must branch on the pending-approval file."""
        import inspect

        from trading.runner import runner as runner_module

        source = inspect.getsource(runner_module.Runner._run_cycle_async)
        assert "APPROVAL_PENDING_FILE" in source
        gateway_line = next(line for line in source.splitlines() if "wedged IBKR Gateway" in line)
        assert "else" in gateway_line or "if " in gateway_line
