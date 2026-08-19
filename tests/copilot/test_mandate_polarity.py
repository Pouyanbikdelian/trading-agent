"""Prohibitions are not instructions to act, and they do not lapse.

Traced on 2026-08-19 from a real failure. The operator wrote "I don't
like PM (Philip Morris), don't buy it in the future". The detector caught
it; the strength grader then matched ``\\bbuy\\b``, ignored the "don't" in
front of it, and filed it as ``strong`` — whereupon STRENGTH_GUIDANCE
told every agent to "act on it unless there is a concrete reason not to".
Fourteen days later it was pruned from the store with no notice, and the
agent PM went on holding PM at 7%.

Three separate defects, three groups of tests:

  * polarity is read before strength, off its own ladder;
  * the phrasings a person actually uses for "no" are recognised at all;
  * a prohibition never expires, and nothing lapses silently.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.copilot.mandates import (
    MEDIUM,
    NEGATIVE,
    POSITIVE,
    STRONG,
    MandateStore,
    grade_polarity,
    grade_strength,
    looks_like_mandate,
)

THE_ORIGINAL = "I don't like PM (Philip Morris), don't buy it in the future"


class TestPolarityIsReadBeforeStrength:
    def test_the_original_message_grades_as_a_prohibition(self) -> None:
        assert grade_polarity(THE_ORIGINAL) == NEGATIVE

    def test_dont_buy_is_not_graded_off_the_word_buy(self) -> None:
        """The exact bug: these two used to be indistinguishable."""
        assert grade_polarity("buy PM next round") == POSITIVE
        assert grade_polarity("don't buy PM next round") == NEGATIVE

    @pytest.mark.parametrize(
        ("text", "strength"),
        [
            ("never buy PM again", STRONG),
            ("don't buy PM going forward", STRONG),
            ("no more PM", STRONG),
            ("stop buying PM", STRONG),
            ("exclude PM from the universe", STRONG),
            ("avoid PM from now on", MEDIUM),
            ("I don't like PM", MEDIUM),
            ("I'd rather not hold PM going forward", MEDIUM),
            ("not keen on PM going forward", "soft"),
        ],
    )
    def test_prohibitions_grade_off_their_own_ladder(self, text: str, strength: str) -> None:
        assert grade_polarity(text) == NEGATIVE
        assert grade_strength(text) == strength

    @pytest.mark.parametrize(
        ("text", "strength"),
        [
            ("I want GS next round", STRONG),
            ("buy more NVDA next cycle", STRONG),
            ("consider XLE going forward", "soft"),
        ],
    )
    def test_positive_grading_is_unchanged(self, text: str, strength: str) -> None:
        assert grade_polarity(text) == POSITIVE
        assert grade_strength(text) == strength

    def test_the_guidance_tells_the_agents_to_read_polarity_first(self) -> None:
        from trading.copilot.mandates import STRENGTH_GUIDANCE

        assert "polarity" in STRENGTH_GUIDANCE
        assert "PROHIBITION" in STRENGTH_GUIDANCE
        # A strong prohibition must not read as a strong reason to act.
        assert "not a strong reason to act" in STRENGTH_GUIDANCE


class TestThePhrasingsPeopleActuallyUse:
    @pytest.mark.parametrize(
        "text",
        [
            THE_ORIGINAL,
            "Avoid PM from now on",
            "Do not hold PM going forward",
            "I do not like PM",
            "exclude PM from the universe",
            "Please stop buying PM",
            "no more PM",
            "PM is a name I never want to own",
            "get rid of PM",
        ],
    )
    def test_a_dislike_is_captured_at_all(self, text: str) -> None:
        """Five of these were journalled as chat and read by nobody."""
        assert looks_like_mandate(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "are you able to consider my conviction on a stock?",
            "why not GS?",
            "SPY is up 3% today",
            "don't worry about it",
            "I didn't understand",
            "thanks, that's not what I meant",
            "can you stop the runner?",
        ],
    )
    def test_ordinary_conversation_is_still_not_an_instruction(self, text: str) -> None:
        """Loosening the detector must not turn chat into standing policy."""
        assert looks_like_mandate(text) is False


class TestNothingLapsesSilently:
    def test_a_prohibition_never_expires(self, tmp_path: Path) -> None:
        """A view goes stale in a fortnight. "Never buy PM" does not."""
        store = MandateStore(tmp_path)

        m = store.add(THE_ORIGINAL, symbols=["PM"])

        assert m.polarity == NEGATIVE
        assert m.expires_at == ""
        assert m.is_expired(datetime.now(tz=timezone.utc) + timedelta(days=365)) is False
        assert [x.id for x in store.active()] == [m.id]

    def test_a_positive_mandate_still_expires_on_schedule(self, tmp_path: Path) -> None:
        store = MandateStore(tmp_path)

        m = store.add("I want GS next round", symbols=["GS"])

        assert m.polarity == POSITIVE
        assert m.expires_at
        assert m.is_expired(datetime.now(tz=timezone.utc) + timedelta(days=15)) is True

    def test_an_expired_mandate_is_retained_and_reportable(self, tmp_path: Path) -> None:
        """It used to be deleted on the next write — no trace, no notice."""
        store = MandateStore(tmp_path)
        store.add("I want GS next round", symbols=["GS"], ttl_days=-1)

        store.add("I want NVDA next round", symbols=["NVDA"])

        assert [m.symbols for m in store.expired()] == [["GS"]]
        assert [m.symbols for m in store.active()] == [["NVDA"]]

    def test_polarity_survives_a_re_grade(self, tmp_path: Path) -> None:
        store = MandateStore(tmp_path)
        m = store.add(THE_ORIGINAL, symbols=["PM"])

        regraded = store.set_strength(m.id, "soft")

        assert regraded is not None
        assert regraded.polarity == NEGATIVE
        assert regraded.expires_at == ""

    def test_a_legacy_row_without_polarity_is_graded_on_read(self, tmp_path: Path) -> None:
        """State files written before 2026-08-19 have no polarity field."""
        import json

        (tmp_path / "operator_mandates.json").write_text(
            json.dumps(
                [
                    {
                        "id": "M1",
                        "text": THE_ORIGINAL,
                        "strength": "strong",
                        "created_at": "2026-08-01T00:00:00+00:00",
                        "expires_at": "",
                        "symbols": ["PM"],
                        "status": "active",
                    }
                ]
            )
        )

        assert MandateStore(tmp_path).active()[0].polarity == NEGATIVE

    def test_the_context_payload_carries_polarity(self, tmp_path: Path) -> None:
        """Without it the agents cannot tell the two apart at all."""
        from trading.copilot.mandates import for_context

        MandateStore(tmp_path).add(THE_ORIGINAL, symbols=["PM"])

        payload = for_context(tmp_path)

        assert payload[0]["polarity"] == NEGATIVE
        assert payload[0]["expires_at"] == "never"
