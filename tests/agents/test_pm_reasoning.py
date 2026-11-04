"""Regression for the Oct 29 two-name PM decision: exclusions, age and book identity."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from trading.agents.pm import format_pm_digest, run_agent_pm
from trading.agents.pm_reasoning import (
    _allocation_facts,
    _candidate_review_plan,
    _date_evidence,
    _fresh_headlines,
    _review_reasoning,
)
from trading.core.clock import UtcClock
from trading.memory import MemoryStore

NOW = datetime(2026, 10, 29, 10, 40, tzinfo=timezone.utc)


def test_dd_exclusion_prompts_next_eligible_alternatives_without_changing_rankings():
    context = {
        "candidate_ladder": {
            "selected_mechanical": ["DD", "MRNA"],
            "ranked": [{"symbol": s} for s in ("DD", "MRNA", "QNT", "FAKE", "DDOG", "RVTY", "CRL")],
        }
    }
    before = json.dumps(context)
    plan = _candidate_review_plan(
        context,
        stocks=("DD", "MRNA", "QNT", "DDOG", "RVTY", "CRL"),
        held=frozenset({"QNT"}),
        excluded=frozenset({"DD"}),
    )
    assert plan["blocked_mechanical"] == {"DD": "operator exclusion"}
    assert plan["required_symbols"] == ["MRNA", "DDOG", "RVTY", "CRL"]
    assert "DD" not in plan["eligible_ranked"]
    assert json.dumps(context) == before
    assert "weights" not in plan  # Replacement review is never an automatic allocation.


def test_evidence_keeps_history_but_flags_old_unknown_and_future_dates():
    rows = [
        {"ts": NOW - timedelta(days=6), "take": "unverified crash rumor"},
        {"ts": NOW - timedelta(hours=72)},
        {"ts": "2026-10-29T10:00:00"},  # timezone unknown
        {"ts": "invalid"},
        {"ts": NOW + timedelta(minutes=1)},
    ]
    dated = _date_evidence(rows, NOW)
    assert [r["requires_revalidation"] for r in dated] == [True, False, True, True, True]
    assert dated[0]["take"] == "unverified crash rumor"
    assert dated[0]["age_hours"] == 144


def test_new_collection_does_not_make_old_headlines_current():
    context = {
        "headlines": [
            {"title": "fresh", "published": "Wed, 28 Oct 2026 18:00:00 GMT"},
            {"title": "old", "published": "Wed, 05 Aug 2026 07:00:00 GMT"},
            {"title": "unknown", "published": ""},
            {"title": "future", "published": "Fri, 30 Oct 2026 07:00:00 GMT"},
        ]
    }
    fresh = _fresh_headlines(context, NOW)
    assert [h["title"] for h in fresh["headlines"]] == ["fresh"]
    assert fresh["headline_freshness"]["omitted_old_or_undated"] == 3
    assert len(context["headlines"]) == 4


def test_simulated_hold_is_a_reduction_and_absent_from_live_snapshot():
    facts = _allocation_facts(
        {"MPC": 0.05, "TMO": 0.04},
        {"MPC": 0.03, "TMO": 0.03},
        {"positions": [], "account": {"equity": 75000}},
    )
    assert facts[0]["sim_change"] == "reduce"
    assert facts[0]["change_pp"] == -2
    assert facts[1]["change_pp"] == -1
    assert all(f["live_position_status"] == "absent_in_snapshot" for f in facts)


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"positions": []},
        {"positions": [{"symbol": "MPC", "qty": "NaN"}], "account": {"equity": 1}},
    ],
)
def test_missing_or_bad_account_is_not_misreported_as_zero_holdings(context):
    assert _allocation_facts({}, {"MPC": 0.03}, context)[0]["live_position_status"] == "unknown"


def test_missing_replacement_review_cannot_silently_pass_completeness_check():
    plan = {"required_symbols": ["MRNA", "DDOG"]}
    response = {
        "candidate_reviews": [
            {
                "symbol": "MRNA",
                "decision": "pass",
                "reason": "event uncertainty",
                "evidence": ["prices:2026-10-28"],
            }
        ]
    }
    result = _review_reasoning(response, plan, {})
    assert not result["complete"]
    assert any("DDOG" in issue for issue in result["issues"])
    assert "cash allocation has no explicit reason" in result["issues"]


def test_cash_is_valid_after_explicit_alternative_review():
    result = _review_reasoning(
        {
            "candidate_reviews": [
                {
                    "symbol": "DDOG",
                    "decision": "pass",
                    "reason": "Sentinel prohibits a replacement sector this cycle",
                    "evidence": ["sentinel_alert:2026-10-28"],
                }
            ],
            "cash_reason": "30% minimum safety cash; remaining cash has no qualifying alternative under the current rotation restriction.",
        },
        {"required_symbols": ["DDOG"]},
        {},
    )
    assert result["complete"]


def test_duplicate_or_contradictory_review_is_flagged():
    row = {"symbol": "MPC", "decision": "pass", "reason": "no", "evidence": ["x"]}
    result = _review_reasoning(
        {"candidate_reviews": [row, row], "cash_reason": "uncertainty"},
        {"required_symbols": []},
        {"MPC": 0.03},
    )
    assert "duplicate review: MPC" in result["issues"]
    assert "MPC: missing or inconsistent allocate review" in result["issues"]


def test_prompt_and_persisted_audit_cover_real_two_name_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trading.agents.pm._stock_universe", lambda: ("DD", "MPC", "TMO", "MRNA", "DDOG")
    )
    (tmp_path / "exclusions.json").write_text(
        json.dumps({"symbols": {"DD": {"reason": "operator"}}})
    )
    pm_dir = tmp_path / "agent_pm"
    pm_dir.mkdir()
    (pm_dir / "portfolio.json").write_text(
        json.dumps({"cash": 91000, "holdings": {"MPC": 50, "TMO": 40}})
    )
    mem = MemoryStore(tmp_path / "memory")
    try:
        old_id = mem.journal(
            "take",
            {
                "agent": "scout",
                "stance": "bullish",
                "take": "old rumor",
                "prediction": {"confidence": 0.9, "subject": "MRNA", "direction": "up"},
            },
        )
        mem.conn.execute(
            "UPDATE journal SET ts=? WHERE id=?",
            ((UtcClock().now() - timedelta(days=6)).timestamp(), old_id),
        )
        context = {
            "positions": [],
            "account": {"equity": 75000},
            "candidate_ladder": {
                "as_of": "2026-10-28",
                "selected_mechanical": ["DD", "MRNA"],
                "ranked": [{"symbol": s} for s in ("MRNA", "DDOG", "MPC", "TMO")],
            },
        }
        captured = {}

        def llm(system, prompt):
            captured.update(json.loads(prompt))
            return {
                "target_weights": {"MPC": 0.03, "TMO": 0.03},
                "rationale": "Keep existing sizes",
            }

        result = run_agent_pm(context, mem, tmp_path, llm=llm, prices={"MPC": 100, "TMO": 100})
        assert captured["sim_portfolio"]["current_weights"] == {"MPC": 0.05, "TMO": 0.04}
        assert not captured["agent_takes"][0]["eligible_for_current_vote"]
        assert captured["candidate_review_plan"]["blocked_mechanical"] == {
            "DD": "operator exclusion"
        }
        assert "DDOG" in result["reasoning_review"]["required_symbols"]
        assert result["weights"] == {"MPC": 0.03, "TMO": 0.03}  # no forced replacement
        assert result["allocation_facts"][0]["change_pp"] == -2
        persisted = json.loads((pm_dir / "last_run.json").read_text())
        assert persisted["reasoning_review"] == result["reasoning_review"]
        journal = mem.journal_tail(1, kind="agent_pm")[0]["payload"]
        assert journal["allocation_facts"] == result["allocation_facts"]
        digest = format_pm_digest(result)
        assert "Reasoning incomplete" in digest
        assert "no current specialist votes" in digest
        assert "5.0% → target 3.0%" in digest
        assert "absent in live snapshot" in digest
    finally:
        mem.conn.close()
