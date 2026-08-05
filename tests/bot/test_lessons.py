"""Operator-authored lessons — tone decides weight.

Before this existed there was no way to write a lesson from Telegram at
all. Asked to record one on 2026-08-05, the copilot filed it as a soft
MANDATE (M5847339): an instruction for the next run, consumed and then
gone, when what was wanted was a durable, gradeable belief. These tests
pin the distinction the whole lesson lifecycle exists to make.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trading.memory.store import MemoryStore


@pytest.fixture
def mem(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


class TestLessonStatus:
    def test_lessons_default_to_candidate(self, mem: MemoryStore) -> None:
        """The historian's proposals must earn their place."""
        lid = mem.add_lesson("Semis trade as one bet, not six.")
        assert mem.lessons(status="candidate")[0]["id"] == lid
        assert mem.lessons(status="established") == []

    def test_operator_can_establish_directly(self, mem: MemoryStore) -> None:
        """It is his desk. A lesson stated as an instruction should not
        wait a month of episodes to be heard."""
        lid = mem.add_lesson("Never average into a falling knife.", status="established")
        assert mem.lessons(status="established")[0]["id"] == lid

    def test_status_must_be_a_real_status(self, mem: MemoryStore) -> None:
        with pytest.raises(ValueError):
            mem.add_lesson("...", status="gospel")

    def test_harden_and_soften_round_trip(self, mem: MemoryStore) -> None:
        lid = mem.add_lesson("Quality mega-caps mean-revert after long drawdowns.")
        assert mem.set_lesson_status(lid, "established") is True
        assert mem.lessons(status="established")[0]["id"] == lid
        assert mem.set_lesson_status(lid, "candidate") is True
        assert mem.lessons(status="candidate")[0]["id"] == lid

    def test_retired_lessons_cannot_be_resurrected_by_status(self, mem: MemoryStore) -> None:
        lid = mem.add_lesson("Something we stopped believing.")
        mem.retire_lesson(lid, why="contradicted twice")
        assert mem.set_lesson_status(lid, "established") is False

    def test_unknown_id_is_false_not_an_exception(self, mem: MemoryStore) -> None:
        assert mem.set_lesson_status("ls-nope", "established") is False

    def test_operator_lessons_are_separable_from_the_historians(self, mem: MemoryStore) -> None:
        mine = mem.add_lesson("Buy quality after 4-5 red days.", tags="operator soft")
        mem.add_lesson("Momentum decays into month end.", tags="historian")
        rows = mem.operator_lessons("candidate")
        assert [r["id"] for r in rows] == [mine]


class TestToneGrading:
    """Tone is read by the same grader the mandate path uses — one
    convention for the operator to learn, not two."""

    @staticmethod
    def _status_for(text: str) -> str:
        from trading.copilot.mandates import STRONG, grade_strength

        return "established" if grade_strength(text) == STRONG else "candidate"

    def test_hard_tone_establishes(self) -> None:
        assert self._status_for("I want you to add this lesson: never chase gaps.") == "established"

    def test_soft_tone_stays_a_candidate(self) -> None:
        assert (
            self._status_for("we could consider buying quality after big down days") == "candidate"
        )

    def test_unrecognised_phrasing_defaults_soft(self) -> None:
        """An unreadable instruction must not silently become binding."""
        assert self._status_for("mega caps rebound eventually") == "candidate"


class TestLessonsReachTheDesk:
    def test_operator_candidates_reach_the_context(self, tmp_path: Path) -> None:
        """Only established lessons used to reach build_context, so a
        tentatively-phrased lesson influenced nothing at all — it waited
        for graded episodes it could never accumulate, because nothing
        trades on a lesson no agent can see."""
        from trading.agents.context import build_context

        mem = MemoryStore(tmp_path / "memory")
        mem.add_lesson("Buy quality after 4-5 red days.", tags="operator soft")

        ctx = build_context(tmp_path, tmp_path / "data")
        under = ctx.get("operator_lessons_under_consideration") or []
        assert any("4-5 red days" in row["lesson"] for row in under)

    def test_copilot_can_see_the_lesson_book(self, tmp_path: Path) -> None:
        """It answered 'I'm read-only' when asked to discuss lessons —
        true about orders, and not what was asked. It could not discuss
        them because it had never been handed any."""
        from trading.copilot import facts

        mem = MemoryStore(tmp_path / "memory")
        mem.add_lesson("Earned over time.", status="established", tags="historian")
        mem.add_lesson("Mine, unproven.", tags="operator soft")

        book = facts.lessons_now(tmp_path)
        assert [r["statement"] for r in book["established"]] == ["Earned over time."]
        assert [r["author"] for r in book["candidates"]] == ["operator"]

    def test_missing_memory_degrades_quietly(self, tmp_path: Path) -> None:
        from trading.copilot import facts

        book = facts.lessons_now(tmp_path / "nothing-here")
        assert book["established"] == [] and book["candidates"] == []
