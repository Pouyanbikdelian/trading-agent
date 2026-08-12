"""The copilot must be able to explain its own interface.

``config_now`` taught it the STATE of the desk. It still could not answer
"how do I make you remember I'm long-term bullish on TSLA" — because the
rules of the interface (three doors, tone grading, the mandate TTL,
/lesson vs /hold) were in no evidence field. It knew what the desk was
doing and not how to be talked to.

The values are read from ``copilot.mandates`` at call time, so a new
strength phrase or a changed TTL cannot leave the copilot coaching the
operator into phrasing that no longer grades the way it claims.
"""

from __future__ import annotations

from pathlib import Path

from trading.copilot import facts

MISSING = Path("/definitely/not/here")


def _iface() -> dict:
    return facts.operator_interface(MISSING)


class TestTheInterfaceDoors:
    def test_all_four_doors_are_described(self) -> None:
        doors = _iface()["four_doors"]

        assert len(doors) == 4
        joined = " ".join(doors).lower()
        assert (
            "command" in joined
            and "proposal" in joined
            and "mandate" in joined
            and "copilot" in joined
        )

    def test_the_door_description_does_not_hide_proposal_persistence(self) -> None:
        """A free-text desk request does create a pending proposal, not a trade."""
        command_door = _iface()["four_doors"][0].lower()

        assert "only door that can change anything" not in command_door
        assert "proposal" in command_door and "approved" in command_door

    def test_it_says_opinions_do_not_persist(self) -> None:
        """The operator's actual question: 'I'm bullish on TSLA' — and the
        answer is that it lands nowhere."""
        assert "PERSISTS NOWHERE" in _iface()["four_doors"][3].upper()

    def test_it_states_that_the_copilot_cannot_act(self) -> None:
        text = _iface()["_the_copilot_cannot_act"]

        assert "cannot" in text
        assert "approval" in text.lower()


class TestMandateGrading:
    def test_the_strength_vocabulary_is_read_from_the_mandates_module(self) -> None:
        """Not a hand-written copy — a new pattern must show up here."""
        from trading.copilot import mandates as m

        graded = _iface()["how_a_mandate_is_graded"]["trigger_phrases_by_strength"]

        assert set(graded) == {s for s, _p in m._STRENGTH_PATTERNS}

    def test_consider_is_flagged_as_soft(self) -> None:
        """'consider X next round' reads as a firm instruction to a human
        and grades soft in code. The gap has to be stated."""
        note = _iface()["how_a_mandate_is_graded"]["_note_consider_is_soft"]

        assert "SOFT" in note.upper()
        assert "I want" in note

    def test_each_strength_says_what_the_desk_will_do(self) -> None:
        does = _iface()["how_a_mandate_is_graded"]["what_each_strength_does"]

        assert set(does) == {"strong", "medium", "soft"}
        assert "may drop it" in does["soft"]

    def test_it_names_the_command_that_fixes_a_misread(self) -> None:
        assert "/harden" in _iface()["how_a_mandate_is_graded"]["_fixing_a_misread"]

    def test_mandates_are_advisory_not_a_limit_override(self) -> None:
        assert "still binds" in _iface()["_mandates_cannot_lift_a_limit"]


class TestTheAlwaysQuestion:
    """The question that prompted this: 'keep this always long term'."""

    def test_the_ttl_is_read_live_not_hardcoded(self) -> None:
        from trading.copilot.mandates import DEFAULT_TTL_DAYS

        assert _iface()["mandates_expire"]["default_ttl_days"] == DEFAULT_TTL_DAYS

    def test_it_says_always_cannot_be_a_mandate(self) -> None:
        meaning = _iface()["mandates_expire"]["_meaning"]

        assert "CANNOT" in meaning
        assert "/lesson" in meaning

    def test_a_durable_belief_routes_to_lesson(self) -> None:
        durable = _iface()["which_tool_for_which_intent"]["a DURABLE belief"]

        assert durable.startswith("/lesson")
        assert "ESTABLISHED" in durable

    def test_hold_is_offered_as_the_mechanical_half(self) -> None:
        """'Keep TSLA long term' wants /lesson AND /hold — the reasoning and
        the mechanical block. Offering only one half half-answers it."""
        hold = _iface()["which_tool_for_which_intent"]["stop a position being sold"]

        assert hold.startswith("/hold")
        assert "/lesson" in hold and "wanted together" in hold

    def test_watchlist_is_explicitly_separate_from_trading(self) -> None:
        tracked = _iface()["which_tool_for_which_intent"]["track a research name"]

        assert "/watchlist add NVDA" in tracked
        assert "does not add a tradable universe member" in tracked


def test_standing_mandates_are_included() -> None:
    """'What have I already told you?' is a question about evidence."""
    assert _iface()["mandates_currently_standing"] == []


def test_it_never_raises_on_a_missing_state_dir() -> None:
    """It runs on every question. It must not be able to break the copilot."""
    out = facts.operator_interface(MISSING)

    assert "four_doors" in out and "mandates_expire" in out


def test_it_is_wired_into_the_evidence_bundle() -> None:
    """A fact provider nothing calls is worth exactly nothing — the shape
    of half the defects this audit found."""
    src = Path("src/trading/copilot/engine.py").read_text()

    assert "NOW_operator_interface" in src
    assert '"interface", facts.operator_interface, state_dir' in src
