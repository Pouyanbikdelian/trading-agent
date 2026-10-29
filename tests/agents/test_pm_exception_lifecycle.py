"""Retaining a simulated holding must not strand its live exception review."""

import json
from datetime import timedelta

import pytest

from trading.agents.exceptions import (
    exception_path,
    list_exceptions,
    normalise_proposal,
    set_exception_status,
)
from trading.agents.pm import format_pm_digest, run_agent_pm
from trading.agents.pm_signal import load_pm_signal
from trading.core.clock import UtcClock
from trading.memory import MemoryStore


@pytest.fixture
def research(tmp_path, monkeypatch):
    monkeypatch.setattr("trading.agents.pm._stock_universe", lambda: ("GEN",))
    mem = MemoryStore(tmp_path / "memory")

    def decide(*, on_ladder=False, paperwork=True, thesis=None):
        response = {"target_weights": {"GEN": 0.06}, "rationale": "Retain the simulated holding."}
        if paperwork:
            response["structured_exceptions"] = [
                {
                    "symbol": "GEN",
                    "thesis": thesis
                    or "Recurring revenue supports a bounded earnings revision thesis.",
                    "source_ids": ["committee:street:2026-10-27"],
                    "horizon_days": 14,
                    "invalidation": "The next report reduces recurring revenue guidance.",
                    "max_weight": 0.06,
                    "sector_impact": "Adds a small software allocation.",
                    "correlation_impact": "Review overlap with the existing technology exposure.",
                }
            ]
        return run_agent_pm(
            {
                "positions": [],  # GEN exists in the simulation only, as in the incident.
                "candidate_ladder": {"ranked": [{"symbol": "GEN" if on_ladder else "AAPL"}]},
            },
            mem,
            tmp_path,
            llm=lambda *_: response,
            prices={"GEN": 25.0, "SPY": 500.0},
        )

    yield decide
    mem.conn.close()


def test_retained_name_falling_off_ladder_gets_review_before_live_eligibility(research, tmp_path):
    research(on_ladder=True)
    result = research()
    assert result["opened"] == []
    assert result["opened_off_ladder"] == []
    assert result["off_ladder_targets"] == ["GEN"]
    assert result["off_ladder_pending_approval"] == ["GEN"]
    assert result["off_ladder_invalid_exception"] == []
    assert "/exceptions" in format_pm_digest(result)
    blocked = load_pm_signal(tmp_path, sleeve_pct=1.0)
    assert blocked.signal is None
    assert "GEN" in blocked.reason and "`/exceptions`" in blocked.reason

    proposal_id = result["off_ladder_exception_ids"][0]
    set_exception_status(tmp_path, proposal_id, "approved")
    allowed = load_pm_signal(tmp_path, sleeve_pct=1.0)
    assert allowed.signal.target_weights == {"equity:GEN": 0.06}


def test_retained_name_with_missing_paperwork_has_actionable_refusal(research, tmp_path):
    research(on_ladder=True)
    result = research(paperwork=False)
    assert result["off_ladder_invalid_exception"] == ["GEN"]
    assert result["off_ladder_pending_approval"] == []
    assert list_exceptions(tmp_path) == []
    blocked = load_pm_signal(tmp_path, sleeve_pct=1.0)
    assert blocked.signal is None
    assert "GEN" in blocked.reason and "proposal missing" in blocked.reason
    assert "`/pm run`" in blocked.reason
    assert "`/pm run`" in format_pm_digest(result)


@pytest.mark.parametrize("paperwork", [True, False])
def test_pending_review_survives_repeated_decisions_without_duplicate_records(
    research, tmp_path, paperwork
):
    first = research()
    second = research(paperwork=paperwork)
    assert second["off_ladder_exception_ids"] == first["off_ladder_exception_ids"]
    assert second["off_ladder_pending_approval"] == ["GEN"]
    assert second["off_ladder_invalid_exception"] == []
    assert len(list_exceptions(tmp_path)) == 1


def test_expired_review_requires_a_new_proposal_and_new_approval(research, tmp_path):
    first = research()
    old_id = first["off_ladder_exception_ids"][0]
    set_exception_status(tmp_path, old_id, "approved")
    rows = list_exceptions(tmp_path)
    rows[0]["expires_at"] = (UtcClock().now() - timedelta(seconds=1)).isoformat()
    exception_path(tmp_path).write_text(json.dumps(rows))
    assert load_pm_signal(tmp_path, sleeve_pct=1.0).signal is None
    second = research()
    assert second["off_ladder_exception_ids"] != [old_id]
    assert second["off_ladder_pending_approval"] == ["GEN"]
    assert len(list_exceptions(tmp_path)) == 2
    assert load_pm_signal(tmp_path, sleeve_pct=1.0).signal is None


def test_approved_retained_review_does_not_request_approval_again(research, tmp_path):
    first = research()
    set_exception_status(tmp_path, first["off_ladder_exception_ids"][0], "approved")
    second = research(paperwork=False)
    assert second["off_ladder_pending_approval"] == []
    assert second["off_ladder_invalid_exception"] == []
    assert load_pm_signal(tmp_path, sleeve_pct=1.0).signal is not None


def test_rejected_review_does_not_unlock_a_retained_target(research, tmp_path):
    first = research()
    set_exception_status(tmp_path, first["off_ladder_exception_ids"][0], "rejected")
    result = research(paperwork=False)
    assert result["off_ladder_invalid_exception"] == ["GEN"]
    assert load_pm_signal(tmp_path, sleeve_pct=1.0).signal is None


def test_expired_pending_proposal_cannot_be_approved(research, tmp_path):
    result = research()
    rows = list_exceptions(tmp_path)
    rows[0]["expires_at"] = (UtcClock().now() - timedelta(seconds=1)).isoformat()
    exception_path(tmp_path).write_text(json.dumps(rows))
    assert set_exception_status(tmp_path, result["off_ladder_exception_ids"][0], "approved") is None
    assert load_pm_signal(tmp_path, sleeve_pct=1.0).signal is None


def test_changed_thesis_cannot_reuse_an_older_approval_for_the_same_ticker(research, tmp_path):
    first = research()
    approved_id = first["off_ladder_exception_ids"][0]
    set_exception_status(tmp_path, approved_id, "approved")
    second = research(thesis="A new acquisition thesis replaces the original revenue outlook.")
    assert approved_id not in second["off_ladder_exception_ids"]
    assert second["off_ladder_pending_approval"] == ["GEN"]
    blocked = load_pm_signal(tmp_path, sleeve_pct=1.0)
    assert blocked.signal is None
    assert "`/exceptions`" in blocked.reason
    set_exception_status(tmp_path, second["off_ladder_exception_ids"][0], "rejected")
    third = research(paperwork=False)
    assert third["off_ladder_invalid_exception"] == ["GEN"]
    assert load_pm_signal(tmp_path, sleeve_pct=1.0).signal is None


@pytest.mark.parametrize(
    "horizon, requested", [(float("inf"), 0.06), (14, float("nan")), (14, 0.0)]
)
def test_malformed_model_numbers_cannot_crash_or_create_a_zero_budget_review(
    research, tmp_path, horizon, requested
):
    research()
    raw = list_exceptions(tmp_path)[0]
    raw["horizon_days"] = horizon
    assert normalise_proposal(raw, symbol="GEN", origin="agent", requested_weight=requested) is None
