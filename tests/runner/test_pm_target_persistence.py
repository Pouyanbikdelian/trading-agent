"""PM exit history advances only after an acknowledged normal submission."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.agents.pm_signal import load_previous_targets, save_targets
from trading.core.config import settings
from trading.core.types import Signal
from trading.runner.cycle import Cycle


def _candidate_signal(*keys: str) -> Signal:
    return Signal(
        ts=datetime(2026, 8, 18, tzinfo=timezone.utc),
        strategy="agent_pm",
        target_weights={},
        metadata={Cycle._PM_TARGET_KEYS_METADATA: json.dumps(sorted(keys))},
    )


@pytest.fixture
def cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cycle:
    monkeypatch.setattr(
        "trading.core.config.settings", settings.model_copy(update={"state_dir": tmp_path})
    )
    # This lifecycle helper intentionally needs no constructed broker or
    # stores. Avoiding a full Cycle rig keeps the timing contract hermetic.
    return Cycle.__new__(Cycle)


@pytest.mark.parametrize("outcome", ["review", "risk_reject", "approval_timeout", "halt"])
def test_no_submission_keeps_the_previous_pm_exit_history(
    cycle: Cycle, tmp_path: Path, outcome: str
) -> None:
    """Every non-executable path reaches this helper with zero acknowledgements."""
    save_targets(tmp_path, {"etf:GLD"})

    cycle._persist_pm_targets_after_successful_submission(
        _candidate_signal("equity:AMD"),
        orders_submitted=0,
        submission_failed=False,
    )

    assert outcome  # Names the safety paths represented by zero submissions.
    assert load_previous_targets(tmp_path) == {"etf:GLD"}


def test_broker_failure_keeps_the_previous_pm_exit_history(cycle: Cycle, tmp_path: Path) -> None:
    save_targets(tmp_path, {"etf:GLD"})

    cycle._persist_pm_targets_after_successful_submission(
        _candidate_signal("equity:AMD"),
        orders_submitted=1,
        submission_failed=True,
    )

    assert load_previous_targets(tmp_path) == {"etf:GLD"}


def test_acknowledged_clean_normal_submission_advances_pm_exit_history(
    cycle: Cycle, tmp_path: Path
) -> None:
    save_targets(tmp_path, {"etf:GLD"})

    cycle._persist_pm_targets_after_successful_submission(
        _candidate_signal("equity:AMD", "equity:MU"),
        orders_submitted=2,
        submission_failed=False,
    )

    assert load_previous_targets(tmp_path) == {"equity:AMD", "equity:MU"}


def test_malformed_candidate_never_overwrites_previous_history(
    cycle: Cycle, tmp_path: Path
) -> None:
    save_targets(tmp_path, {"etf:GLD"})
    malformed = Signal(
        ts=datetime(2026, 8, 18, tzinfo=timezone.utc),
        strategy="agent_pm",
        target_weights={},
        metadata={Cycle._PM_TARGET_KEYS_METADATA: "not-json"},
    )

    cycle._persist_pm_targets_after_successful_submission(
        malformed,
        orders_submitted=1,
        submission_failed=False,
    )

    assert load_previous_targets(tmp_path) == {"etf:GLD"}
