"""One budget for a cycle, derived rather than guessed in two places.

The runner aborts a cycle with ``asyncio.wait_for`` and the cycle refuses
to submit past a zombie guard. Both need the same number, and until
2026-08-20 both hardcoded it at 300s / 290,000ms while the approval
window was allowed to wait 600s for the operator.

The consequence, in production: a basket card said "⏱ Auto-rejects after
10 min", the operator took longer than five, and the runner killed the
cycle mid-wait — reporting *"likely a wedged IBKR Gateway"* about a
gateway that was working perfectly, and counting it as error #1/3 toward
an automatic halt. Three unhurried approvals in a row would have halted
the desk for being read slowly.

Waiting for a human is not work, and it is not a fault. The budget is
the time the machine needs plus however long the operator is permitted
to take, so a timeout again means what it says: something is stuck.
"""

from __future__ import annotations

#: What the machine itself needs: price load, agents, sizing, submit and
#: reconcile. This is the figure that used to be the whole budget.
CYCLE_WORK_BUDGET_S: float = 300.0

#: Margin between the cycle standing down and the runner cancelling it, so
#: the cycle always gets to report its own refusal rather than being
#: killed mid-sentence.
ZOMBIE_GUARD_MARGIN_S: float = 10.0


def approval_budget_s(*, require_cycle_approval: bool, cycle_approval_timeout_s: float) -> float:
    """Time the operator is allowed to spend deciding. Zero when unused."""
    if not require_cycle_approval:
        return 0.0
    return max(0.0, float(cycle_approval_timeout_s))


def cycle_timeout_seconds(
    *,
    require_cycle_approval: bool,
    cycle_approval_timeout_s: float,
    work_budget_s: float = CYCLE_WORK_BUDGET_S,
) -> float:
    """How long the runner waits before calling a cycle stuck."""
    return work_budget_s + approval_budget_s(
        require_cycle_approval=require_cycle_approval,
        cycle_approval_timeout_s=cycle_approval_timeout_s,
    )


def zombie_guard_ms(
    *,
    require_cycle_approval: bool,
    cycle_approval_timeout_s: float,
    work_budget_s: float = CYCLE_WORK_BUDGET_S,
) -> float:
    """When a still-running cycle must refuse to submit.

    Sits ``ZOMBIE_GUARD_MARGIN_S`` inside the runner's timeout. The runner
    cancels the awaiting coroutine but NOT this worker thread, so a cycle
    told it had aborted could otherwise still reach the broker.
    """
    timeout = cycle_timeout_seconds(
        require_cycle_approval=require_cycle_approval,
        cycle_approval_timeout_s=cycle_approval_timeout_s,
        work_budget_s=work_budget_s,
    )
    return max(0.0, timeout - ZOMBIE_GUARD_MARGIN_S) * 1000.0


def from_settings(settings: object) -> tuple[float, float]:
    """``(timeout_seconds, zombie_guard_ms)`` for the live configuration."""
    require = bool(getattr(settings, "require_cycle_approval", False))
    approval = float(getattr(settings, "cycle_approval_timeout_s", 0.0) or 0.0)
    return (
        cycle_timeout_seconds(require_cycle_approval=require, cycle_approval_timeout_s=approval),
        zombie_guard_ms(require_cycle_approval=require, cycle_approval_timeout_s=approval),
    )
