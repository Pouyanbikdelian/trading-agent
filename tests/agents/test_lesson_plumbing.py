"""Lessons have to reach the decision, not merely be written down.

The 2026-08-19 audit found the learning loop was wired end to end and
still leaked at three points. Nothing here tests whether a lesson is
*good* — only whether it survives the journey from the store to the
model's prompt, which is where they were being lost silently.

  1. An operator-stated lesson could win zero retrieval slots.
  2. The manager prompt was a raw ``[:9000]`` slice with the lesson book
     near the end, so a busy day cut it off mid-string.
  3. The PM's charter never mentioned lessons at all, and its budgeted
     trim listed them among the first things to drop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading.memory.store import MemoryStore

CONDITIONS = {"macro_bucket": "risk_off", "vol_bucket": "high", "style_leader": "value"}


@pytest.fixture
def mem(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory")
    yield store
    store.close()


class TestAnOperatorLessonReachesThePrompt:
    def test_it_is_retrieved_even_with_no_conditions_and_no_evidence(self, mem) -> None:
        """The exact shape `/lesson` writes: no conditions, zero evidence.

        `_condition_match` scores that 0.0, and the ranked loop skips every
        zero-relevance card — so before this fix the operator's own
        conclusion could not occupy a relevant slot at all.
        """
        lid = mem.add_lesson(
            "Do not chase tobacco names on yield alone",
            tags="operator strong",
            status="established",
        )

        picked = mem.retrieve_lessons(CONDITIONS)

        assert lid in {c["id"] for c in picked}
        assert next(c for c in picked if c["id"] == lid)["retrieval_role"] == "operator_stated"

    def test_it_outranks_a_machine_lesson_with_more_evidence(self, mem) -> None:
        """It used to lose to any legacy rule with one supporting episode."""
        strong_machine = mem.add_lesson(
            "Momentum works in low vol", tags="historian", status="established"
        )
        for i in range(5):
            mem.add_evidence(strong_machine, f"E{i}", supports=True)
        operator = mem.add_lesson("Never buy PM", tags="operator strong", status="established")

        picked = mem.retrieve_lessons(CONDITIONS, max_relevant=1, max_diversifiers=0)

        assert [c["id"] for c in picked] == [operator]

    def test_several_operator_lessons_do_not_crowd_out_everything(self, mem) -> None:
        """Guaranteed slots are capped, so the regime picks keep a seat."""
        for i in range(5):
            mem.add_lesson(f"operator rule {i}", tags="operator", status="established")
        matched = mem.add_lesson(
            "regime rule", tags="historian", status="established", conditions=CONDITIONS
        )

        picked = mem.retrieve_lessons(CONDITIONS, max_relevant=3, max_diversifiers=2)

        assert sum(c["retrieval_role"] == "operator_stated" for c in picked) == 3
        assert matched in {c["id"] for c in picked}

    def test_a_tag_that_merely_contains_the_word_is_not_an_operator_lesson(self, mem) -> None:
        """`cooperative` must not smuggle a machine rule into a free slot."""
        sneaky = mem.add_lesson("x", tags="cooperative signals", status="established")

        picked = mem.retrieve_lessons(CONDITIONS)

        roles = {c["id"]: c["retrieval_role"] for c in picked}
        assert roles.get(sneaky) != "operator_stated"


class TestTheManagerPromptStopsSlicingLessonsOff:
    def _payload(self, n_takes: int, take_chars: int) -> dict:
        return {
            "takes": [
                {"agent": f"a{i}", "take": "x" * take_chars, "stance": "bull"}
                for i in range(n_takes)
            ],
            "objections": ["o" * 200],
            "calibration": [{"agent": "a0", "hit_rate": 0.5}],
            "disagreement_index": 0.4,
            "guard_flags": {"ratchet": True},
            "established_lessons": [
                {"id": "L1", "statement": "Never buy PM", "retrieval_role": "operator_stated"}
            ],
            "operator_lessons_under_consideration": [{"id": "L2", "statement": "watch HY OAS"}],
        }

    def test_a_small_payload_is_untouched(self) -> None:
        from trading.agents.committee import _budgeted_manager_prompt

        payload = self._payload(2, 50)

        assert json.loads(_budgeted_manager_prompt(payload)) == payload

    def test_an_oversized_payload_sheds_takes_and_keeps_the_lessons(self) -> None:
        """The whole point: opinions give way, learning does not."""
        from trading.agents.committee import _budgeted_manager_prompt

        rendered = _budgeted_manager_prompt(self._payload(8, 4_000))

        parsed = json.loads(rendered)  # valid JSON, not a mid-string cut
        assert len(parsed["takes"]) < 8
        assert parsed["established_lessons"][0]["statement"] == "Never buy PM"
        assert parsed["operator_lessons_under_consideration"]

    def test_it_never_emits_truncated_json(self) -> None:
        from trading.agents.committee import _budgeted_manager_prompt

        for chars in (1_000, 9_000, 40_000):
            json.loads(_budgeted_manager_prompt(self._payload(8, chars)))


class TestThePMIsToldWhatALessonIs:
    def test_the_charter_names_established_lessons_and_demands_an_answer(self) -> None:
        """It described mandates and holds, and never mentioned lessons.

        They arrived inside ``today_context`` as unlabelled JSON with no
        instruction attached, next to eight other keys.
        """
        from trading.agents.pm import PM_CHARTER

        assert "ESTABLISHED LESSONS" in PM_CHARTER
        assert "established_lessons" in PM_CHARTER
        assert "operator_lessons_under_consideration" in PM_CHARTER

    def test_the_budgeted_trim_no_longer_lists_lessons_as_droppable(self) -> None:
        """A prompt that demands an answer must not silently delete the question."""
        import inspect

        from trading.agents.pm import _budgeted_prompt

        source = inspect.getsource(_budgeted_prompt)
        drop_lists = [
            line
            for line in source.splitlines()
            if "for key in (" in line or line.strip().startswith('"')
        ]
        assert not any(
            '"established_lessons"' in line and "for key in (" in line for line in drop_lists
        )
        # And positively: it survives a squeeze that drops everything else.
        payload = {
            "sim_portfolio": {"equity": 1.0},
            "agent_takes": [{"take": "x" * 30_000}],
            "today_context": {
                "headlines": ["h" * 5_000],
                "established_lessons": [{"id": "L1", "statement": "Never buy PM"}],
            },
        }
        parsed = json.loads(_budgeted_prompt(payload, budget=2_000))
        assert parsed["today_context"]["established_lessons"][0]["statement"] == "Never buy PM"
