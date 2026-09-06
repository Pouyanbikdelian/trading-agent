"""Learning Curator state stays evidence-led, reversible, and auditable."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading.memory import MemoryStore


def _graded_prediction(mem: MemoryStore, subject: str, move: float = 0.02) -> str:
    prediction_id = mem.add_prediction(
        agent="quant",
        subject=subject,
        direction="up",
        horizon_days=5,
        confidence=0.65,
        statement="measured outcome for curator test",
    )
    mem.grade_prediction(prediction_id, realized_move=move)
    return prediction_id


def test_machine_lesson_ranking_uses_measured_not_review_evidence(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    conditions = {"macro_bucket": "stress"}
    review_only = mem.add_lesson(
        "Review-only lesson should not outrank measured evidence.",
        status="established",
        conditions={"snapshot": conditions},
    )
    measured = mem.add_lesson(
        "Measured lesson should rank ahead of review-only prose.",
        status="established",
        conditions={"snapshot": conditions},
    )
    for index in range(6):
        assert mem.add_evidence(review_only, f"wk-2026-W{index:02d}", supports=True)
    for index in range(3):
        outcome = _graded_prediction(mem, f"M{index}")
        assert mem.add_evidence(measured, outcome, supports=True)

    selected = mem.retrieve_lessons(conditions, max_relevant=1, max_diversifiers=0)

    assert [row["id"] for row in selected] == [measured]
    assert selected[0]["outcome_support"] == 3


def test_curator_archive_requires_measured_challenge_and_restores_to_candidate(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    lesson_id = mem.add_lesson("Thin-breadth breakouts should be treated as a failed setup.")
    supports = [_graded_prediction(mem, f"S{index}") for index in range(3)]
    contradicts = [_graded_prediction(mem, f"C{index}", move=-0.02) for index in range(3)]
    for outcome in supports:
        assert mem.add_evidence(lesson_id, outcome, supports=True)
    for outcome in contradicts:
        assert mem.add_evidence(lesson_id, outcome, supports=False)

    recommendation = next(
        item
        for item in mem.lesson_review({})["archive_recommendations"]
        if item["lesson_id"] == lesson_id
    )
    assert recommendation["outcome_support"] == 3
    assert recommendation["outcome_contradict"] == 3
    assert recommendation["requires_operator_approval"] is True

    assert mem.archive_challenged_lesson(lesson_id, "Approved after measured contradictions")
    assert mem.lessons(status="retired")[0]["id"] == lesson_id
    assert not mem.add_evidence(lesson_id, _graded_prediction(mem, "AFTER"), supports=True)

    assert mem.restore_retired_lesson(lesson_id, "New regime data warrants fresh review")
    row = mem.lessons(status="candidate")[0]
    assert row["id"] == lesson_id
    # Restoration is not a shortcut back into agent context. It can only
    # re-establish after enough *new* measured outcomes offset the old record.
    for index in range(3):
        assert mem.add_evidence(
            lesson_id, _graded_prediction(mem, f"REESTABLISH{index}"), supports=True
        )
    assert mem.lessons(status="established")[0]["id"] == lesson_id
    card = (mem.lessons_dir / f"{lesson_id}.md").read_text()
    assert "Previously archived" in card
    assert any(event["kind"] == "lesson_restored" for event in mem.journal_tail(10))


def test_operator_lessons_never_receive_curator_archive_recommendations(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    lesson_id = mem.add_lesson(
        "Operator rule is managed explicitly, never auto-curated.",
        tags="operator strong",
        status="established",
    )
    for index in range(3):
        outcome = _graded_prediction(mem, f"O{index}", move=-0.02)
        assert mem.add_evidence(lesson_id, outcome, supports=False)

    assert mem.lessons(status="challenged")[0]["id"] == lesson_id
    assert all(
        item["lesson_id"] != lesson_id for item in mem.lesson_review({})["archive_recommendations"]
    )


def test_operator_authorship_requires_an_exact_reserved_tag(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    lookalikes = [
        mem.add_lesson(
            "A machine lesson must not gain operator authority.",
            tags="operator-ish",
            status="established",
        ),
        mem.add_lesson(
            "A cooperative tag is not operator authorship.",
            tags="cooperator signals",
            status="established",
        ),
    ]

    assert mem.operator_lessons("established") == []
    selected = mem.retrieve_lessons({}, max_relevant=4, max_diversifiers=0)
    assert not any(
        row["id"] in lookalikes and row["retrieval_role"] == "operator_stated" for row in selected
    )


def test_lesson_card_keeps_every_archive_restore_cycle(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    lesson_id = mem.add_lesson("Archive history must remain visible across review cycles.")

    assert mem.retire_lesson(lesson_id, "First archival rationale", actor="operator")
    assert mem.restore_retired_lesson(
        lesson_id, "Fresh data warrants a re-review", actor="operator"
    )
    assert mem.retire_lesson(lesson_id, "Second archival rationale", actor="operator")

    card = (mem.lessons_dir / f"{lesson_id}.md").read_text()
    assert "First archival rationale" in card
    assert "Fresh data warrants a re-review" in card
    assert "Second archival rationale" in card
    assert card.count("**archived** by operator") == 2


def test_summary_lifecycle_includes_manual_status_changes_and_actor(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    lesson_id = mem.add_lesson("Manual lesson changes must remain attributable.", actor="operator")
    assert mem.set_lesson_status(lesson_id, "established", actor="operator")

    changes = mem.curator_summary()["changes"]
    transition = next(change for change in changes if change["action"] == "status changed")
    assert transition["lesson_id"] == lesson_id
    assert transition["actor"] == "operator"
    assert transition["reason"] == "candidate → established"


def test_curator_run_is_persisted_for_dashboard_and_markdown_audit(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    lesson_id = mem.add_lesson("Curator audit stays attributable to its review run.")

    run_id = mem.record_curator_run(
        status="completed",
        conditions={"vol_bucket": "elevated"},
        reviewed=1,
        created=0,
        voted=0,
        vote_ok=True,
        archive_recommendations=0,
        actions=[
            {
                "lesson_id": lesson_id,
                "rank": 1,
                "action": "awaiting_evidence",
                "before_status": "candidate",
                "after_status": "candidate",
                "reason": "No completed outcome linked yet.",
                "evidence_ids": [],
            }
        ],
        ts=datetime(2026, 9, 4, 19, tzinfo=timezone.utc),
    )

    summary = mem.curator_summary()
    assert summary["last_run"]["id"] == run_id
    assert summary["last_run"]["ok"] is True
    assert summary["recent_review_actions"][0]["lesson_id"] == lesson_id
    assert (mem.reviews_dir / f"{run_id}.md").exists()


def test_curator_run_rolls_back_when_its_markdown_audit_cannot_be_written(
    tmp_path, monkeypatch
) -> None:
    mem = MemoryStore(tmp_path / "memory")

    def fail_report(*args, **kwargs) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(mem, "_write_curator_report", fail_report)
    with pytest.raises(OSError, match="disk unavailable"):
        mem.record_curator_run(
            status="completed",
            conditions={},
            reviewed=0,
            created=0,
            voted=0,
            vote_ok=None,
            archive_recommendations=0,
            actions=[],
        )

    assert mem.conn.execute("SELECT COUNT(*) FROM curator_runs").fetchone()[0] == 0
    assert mem.conn.execute("SELECT COUNT(*) FROM curator_actions").fetchone()[0] == 0
    assert mem.journal_tail(1, kind="lesson_curation") == []


def test_curator_summary_marks_a_missed_review_stale(tmp_path) -> None:
    mem = MemoryStore(tmp_path / "memory")
    mem.record_curator_run(
        status="completed",
        conditions={},
        reviewed=0,
        created=0,
        voted=0,
        vote_ok=None,
        archive_recommendations=0,
        actions=[],
        ts=datetime.now(tz=timezone.utc) - timedelta(days=6),
    )

    last = mem.curator_summary()["last_run"]
    assert last["fresh"] is False
    assert last["health"] == "stale"
    assert last["ok"] is False
