"""Operator authority and measured market evidence must remain distinguishable."""

import pytest

from trading.agents.context import build_context
from trading.agents.historian import SCOPE_FIELDS, SCOPE_MAX_CHARS, run_historian
from trading.memory import MemoryStore


def test_context_separates_instruction_authority_from_measured_evidence(tmp_path):
    mem = MemoryStore(tmp_path / "memory")
    operator_id = mem.add_lesson(
        "Preserve the owner's instruction without claiming market validation.",
        tags="operator strong",
        status="established",
    )
    machine_id = mem.add_lesson(
        "A machine hypothesis carries measured evidence and applicability.",
        status="established",
        conditions={"scope": {"applies_when": "after breadth confirms"}},
    )
    for index in range(7):
        mem.add_evidence(operator_id, f"wk-{index}", supports=True)
    for index in range(3):
        mem.add_evidence(operator_id, f"old-review-{index}", supports=False)
    prediction = mem.add_prediction(
        agent="quant",
        subject="SPY",
        direction="up",
        horizon_days=5,
        confidence=0.6,
        statement="a measured future observation",
    )
    mem.grade_prediction(prediction, realized_move=0.02)
    mem.add_evidence(machine_id, prediction, supports=True)
    mem.close()

    context = build_context(tmp_path, tmp_path, include_candidate_ladder=False)
    cards = {card["id"]: card for card in context["established_lessons"]}
    operator = cards[operator_id]
    assert operator["provenance"] == "operator_instruction"
    assert "authority does not depend" in operator["authority"]
    assert operator["retrieval_role"] == "operator_stated"
    assert operator["support_vs_contradict"] == "0/0"
    assert operator["measured_outcomes"] == {"support": 0, "contradict": 0}
    assert operator["legacy_and_review_votes"] == {"support": 7, "contradict": 3}
    empirical = cards[machine_id]
    assert empirical["provenance"] == "empirical_lesson"
    assert empirical["measured_outcomes"] == {"support": 1, "contradict": 0}
    assert empirical["promotion_evidence"]["support"] == 1
    assert empirical["scope"]["applies_when"] == "after breadth confirms"


@pytest.mark.parametrize("field", SCOPE_FIELDS)
@pytest.mark.parametrize("invalid", [None, "", "   ", False, [], "x" * (SCOPE_MAX_CHARS + 1)])
def test_new_historian_scope_is_required_and_typed(tmp_path, field, invalid):
    mem = MemoryStore(tmp_path / "memory")
    evidence_id = mem.add_prediction(
        agent="quant",
        subject="SPY",
        direction="up",
        horizon_days=5,
        confidence=0.6,
        statement="verified discovery outcome",
    )
    mem.grade_prediction(evidence_id, realized_move=0.02)
    lesson = {
        "statement": "Breadth confirmation predicts recovery over the next five sessions.",
        "applies_when": "elevated volatility after breadth confirms, over five sessions",
        "fails_when": "fresh macro shock before the fifth session",
        "invalidated_if": "three consecutive breadth signals fail over five sessions",
        "sample": "three separate signals across six weeks",
        "source_ids": [evidence_id],
    }
    if invalid is None:
        lesson.pop(field)
    else:
        lesson[field] = invalid
    digest = run_historian(mem, llm=lambda system, prompt: {"new_lessons": [lesson], "votes": []})
    assert digest["created"] == []
    assert mem.lessons() == []
    run = mem.curator_summary()["last_run"]
    assert run["status"] == "degraded"
    assert field in run["reason"]
    mem.close()


def test_legacy_lesson_is_not_retroactively_rejected_or_demoted(tmp_path):
    mem = MemoryStore(tmp_path / "memory")
    lesson_id = mem.add_lesson(
        "Legacy market lesson has no structured scope yet.", status="established"
    )
    digest = run_historian(mem, llm=lambda system, prompt: {"new_lessons": [], "votes": []})
    assert digest["ok"]
    assert mem.lessons(status="established")[0]["id"] == lesson_id
    mem.close()
