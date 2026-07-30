"""Standing instructions, and the tone that grades them.

The property that matters most is at the bottom: a mandate is context for
a decision, never a lever on the risk caps. "I want 40% GS" typed from a
phone must still come out clamped at the single-stock limit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.copilot.mandates import (
    MEDIUM,
    SOFT,
    STRONG,
    MandateStore,
    grade_strength,
    looks_like_mandate,
    mandate_span,
)


class TestToneGrading:
    @pytest.mark.parametrize(
        "text",
        [
            "I want GS in the book next round",
            "buy GS next round",
            "make sure we hold some AI names",
            "we must include GS next time",
            "definitely add GS next round",
        ],
    )
    def test_firm_phrasing_reads_strong(self, text: str) -> None:
        assert grade_strength(text) == STRONG

    @pytest.mark.parametrize(
        "text",
        [
            "high conviction on GS, include it next round",
            "highly consider GS next round",
            "I'd like more AI exposure next round",
            "we should own GS going forward",
        ],
    )
    def test_committed_but_not_absolute_reads_medium(self, text: str) -> None:
        assert grade_strength(text) == MEDIUM

    @pytest.mark.parametrize(
        "text",
        [
            "consider GS next round",
            "I think GS could be a good idea next round",
            "maybe look at GS next round",
            "GS might be worth a look next round",
        ],
    )
    def test_tentative_phrasing_reads_soft(self, text: str) -> None:
        assert grade_strength(text) == SOFT

    def test_unrecognised_phrasing_defaults_soft(self) -> None:
        """Under-weight rather than over-weight: the operator can restate
        more firmly, but cannot un-buy a position taken on a misread."""
        assert grade_strength("GS next round please") == SOFT

    def test_strongest_marker_wins_over_a_later_hedge(self) -> None:
        assert grade_strength("I want GS next round, maybe 5%") == STRONG


class TestMandateDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "high conviction on GS, include it next round",
            "consider GS next round",
            "I want GS in the book next round",
            "going forward I want at least some AI exposure",
            "from now on avoid airlines",
        ],
    )
    def test_forward_looking_instructions_are_mandates(self, text: str) -> None:
        assert looks_like_mandate(text)

    @pytest.mark.parametrize(
        "text",
        [
            "what is GS?",
            "why did we buy MU?",
            "how much cash do we have?",
            "is the MU thesis still valid?",
            "did the committee consider GS?",
            "ok",
        ],
    )
    def test_questions_are_never_mandates(self, text: str) -> None:
        """A question routes to the copilot; capturing it as an
        instruction would put words in the operator's mouth."""
        assert not looks_like_mandate(text)

    @pytest.mark.parametrize(
        "text",
        [
            # Both observed live on 2026-07-30 — captured as mandates and
            # echoed back as accepted instructions. The operator was asking
            # what the desk is capable of; nothing was being instructed.
            "If I tell you I would like to have a stock to buy within the next "
            "rebalancing in the PM agent simulation are you able to do that?",
            "No, I ask you about. Are you able to consider my convictions on a "
            "stock if I recommend you?",
            "can you consider my conviction on a name",
            "is it possible to add GS next round",
            "what happens if I want to buy GS next round",
            "suppose I wanted GS in the book next round",
        ],
    )
    def test_capability_questions_are_never_mandates(self, text: str) -> None:
        assert not looks_like_mandate(text)

    def test_span_is_the_instruction_clause_not_the_whole_bubble(self) -> None:
        """A real instruction wrapped in chatter still lands, and what gets
        stored is the instruction — not the paragraph around it."""
        span = mandate_span(
            "Been reading about banks all morning. I want GS in the book next "
            "round. Does that make sense to you?"
        )
        assert span == "I want GS in the book next round."


class TestStoreLifecycle:
    def test_add_and_list(self, tmp_path: Path) -> None:
        s = MandateStore(tmp_path)
        m = s.add("I want GS next round", symbols=["GS"])
        assert m.strength == STRONG
        active = s.active()
        assert len(active) == 1 and active[0].symbols == ["GS"]

    def test_expired_mandates_drop_out(self, tmp_path: Path) -> None:
        s = MandateStore(tmp_path)
        s.add("consider GS next round", ttl_days=1)
        later = datetime.now(tz=timezone.utc) + timedelta(days=2)
        assert s.active(now=later) == []

    def test_strongest_first(self, tmp_path: Path) -> None:
        """A truncated prompt must keep the firm instruction."""
        s = MandateStore(tmp_path)
        s.add("consider XLE next round")
        s.add("I want GS next round")
        assert s.active()[0].strength == STRONG

    def test_cancel(self, tmp_path: Path) -> None:
        s = MandateStore(tmp_path)
        m = s.add("consider GS next round")
        assert s.cancel(m.id) is True
        assert s.active() == []
        assert s.cancel("nope") is False

    def test_regrade_fixes_a_misread_tone(self, tmp_path: Path) -> None:
        s = MandateStore(tmp_path)
        m = s.add("consider GS next round")
        assert m.strength == SOFT
        again = s.set_strength(m.id, STRONG)
        assert again is not None and again.strength == STRONG
        assert s.active()[0].strength == STRONG

    def test_regrade_unknown_id_returns_none(self, tmp_path: Path) -> None:
        assert MandateStore(tmp_path).set_strength("M999", STRONG) is None

    def test_clear(self, tmp_path: Path) -> None:
        s = MandateStore(tmp_path)
        s.add("consider GS next round")
        assert s.clear() == 1
        assert s.active() == []

    def test_corrupt_file_degrades_quietly(self, tmp_path: Path) -> None:
        (tmp_path / "operator_mandates.json").write_text("{not json")
        assert MandateStore(tmp_path).active() == []

    def test_mandate_is_journaled(self, tmp_path: Path) -> None:
        from trading.memory.store import MemoryStore

        MandateStore(tmp_path).add("I want GS next round", symbols=["GS"])
        mem = MemoryStore(tmp_path / "memory")
        try:
            rows = mem.journal_tail(5, kind="operator_mandate")
        finally:
            mem.close()
        assert rows and rows[0]["payload"]["strength"] == STRONG


class TestMandatesCannotLiftRiskCaps:
    """The load-bearing safety property of this whole feature."""

    def test_guidance_says_caps_still_bind(self) -> None:
        from trading.copilot.mandates import STRENGTH_GUIDANCE

        g = STRENGTH_GUIDANCE.lower()
        assert "no mandate is an order" in g
        assert "none suspends the risk limits" in g
        assert "never allocate past a cap" in g

    def test_a_strong_mandate_is_still_clamped(self) -> None:
        """Even 'I want 40% GS' comes out at the single-stock cap."""
        from trading.agents.pm import MAX_WEIGHT_PER_STOCK, _clamp_weights

        out = _clamp_weights({"GS": 0.40}, stocks=("GS",))
        assert out["GS"] == pytest.approx(MAX_WEIGHT_PER_STOCK)

    def test_a_mandate_cannot_reach_a_pinned_symbol(self) -> None:
        """/hold outranks a mandate — the operator's own long-term pins
        stay off-limits to the PM regardless of what he asked for."""
        from trading.agents.pm import _clamp_weights

        assert _clamp_weights({"GS": 0.1}, stocks=("GS",), blocked=frozenset({"GS"})) == {}

    def test_a_mandate_cannot_conjure_an_off_universe_ticker(self) -> None:
        from trading.agents.pm import _clamp_weights

        assert _clamp_weights({"NOTREAL": 0.2}, stocks=("GS",)) == {}


class TestAgentsSeeMandates:
    def test_context_carries_active_mandates(self, tmp_path: Path) -> None:
        from trading.agents.context import build_context

        MandateStore(tmp_path).add("I want GS next round", symbols=["GS"])
        ctx = build_context(tmp_path, tmp_path / "data")
        assert ctx["operator_mandates"]
        assert ctx["operator_mandates"][0]["strength"] == STRONG

    def test_pm_charter_and_manager_charter_carry_the_guidance(self) -> None:
        from trading.agents.committee import MANAGER_CHARTER
        from trading.agents.pm import PM_CHARTER

        for charter in (MANAGER_CHARTER, PM_CHARTER):
            assert "operator_mandates" in charter
            assert "strong" in charter.lower()

    def test_risk_officer_and_coach_see_mandates(self) -> None:
        from trading.agents.committee import _VIEW_KEYS

        assert "operator_mandates" in _VIEW_KEYS["risk_officer"]
        assert "operator_mandates" in _VIEW_KEYS["position_coach"]
