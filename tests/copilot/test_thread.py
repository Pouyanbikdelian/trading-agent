"""Conversation memory, answer sizing, and objection capture.

The behaviours pinned here are the ones that make the bot conversational
without making it credulous: a follow-up must resolve, a stale thread must
not leak yesterday's topic into this morning's question, and the operator's
pushback must survive the scrollback.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.copilot.engine import answer_budget
from trading.copilot.thread import (
    IDLE_EXPIRY_S,
    MAX_TURNS,
    Thread,
    extract_symbols,
    looks_like_objection,
    recent_objections,
    record_exchange,
)


class TestThreadWindow:
    def test_turns_round_trip(self, tmp_path: Path) -> None:
        t = Thread(tmp_path)
        t.append("operator", "why did we buy MU?")
        t.append("copilot", "momentum rank 3")
        turns, _ = t.load()
        assert [x.role for x in turns] == ["operator", "copilot"]
        assert turns[0].text == "why did we buy MU?"

    def test_window_is_bounded(self, tmp_path: Path) -> None:
        t = Thread(tmp_path)
        for i in range(20):
            t.append("operator", f"q{i}")
        turns, _ = t.load()
        assert len(turns) == MAX_TURNS
        assert turns[-1].text == "q19"  # newest kept, oldest dropped

    def test_missing_file_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert Thread(tmp_path).load() == ([], None)

    def test_corrupt_file_degrades_quietly(self, tmp_path: Path) -> None:
        (tmp_path / "copilot_thread.json").write_text("{not json")
        assert Thread(tmp_path).load() == ([], None)

    def test_stale_thread_is_dropped(self, tmp_path: Path) -> None:
        """Asking 'and XLE?' the next morning must not inherit last
        night's argument about semis."""
        old = (datetime.now(tz=timezone.utc) - timedelta(seconds=IDLE_EXPIRY_S + 60)).isoformat()
        (tmp_path / "copilot_thread.json").write_text(
            json.dumps(
                {"turns": [{"role": "operator", "text": "semis?", "ts": old}], "last_symbol": "MU"}
            )
        )
        turns, sym = Thread(tmp_path).load()
        assert turns == [] and sym is None

    def test_symbol_carries_forward(self, tmp_path: Path) -> None:
        """What makes a bare follow-up answerable at all."""
        t = Thread(tmp_path)
        t.append("operator", "why did we buy MU?", symbol="MU")
        t.append("copilot", "momentum rank 3", symbol="MU")
        t.append("operator", "and the thesis?")  # no symbol named
        _turns, sym = t.load()
        assert sym == "MU"

    def test_clear_resets(self, tmp_path: Path) -> None:
        t = Thread(tmp_path)
        t.append("operator", "hello there")
        t.clear()
        assert t.load() == ([], None)

    def test_long_turns_are_truncated(self, tmp_path: Path) -> None:
        t = Thread(tmp_path)
        t.append("copilot", "x" * 5000)
        turns, _ = t.load()
        assert len(turns[0].text) <= 600


class TestAnswerBudget:
    def test_short_definitional_question_gets_a_short_budget(self) -> None:
        # The 2026-07-29 transcript case: five words, four sentences back.
        assert "1-2 sentences" in answer_budget("What is XLV?")

    def test_why_questions_get_room(self) -> None:
        budget = answer_budget("why did we buy MU last week and what was the dissent?")
        assert "6 sentences" in budget

    def test_thesis_questions_get_room(self) -> None:
        assert "sentences" in answer_budget("is the MU thesis still valid?")

    def test_medium_factual_question_is_mid_sized(self) -> None:
        assert answer_budget("how much cash is in the trading account right now?") == (
            "2-3 sentences"
        )

    def test_empty_question_is_bounded(self) -> None:
        assert answer_budget("") == "1 sentence"

    @pytest.mark.parametrize(
        "q", ["what is XLV?", "how much cash?", "are we halted?", "whats our biggest position?"]
    )
    def test_simple_questions_never_get_the_long_budget(self, q: str) -> None:
        assert "6 sentences" not in answer_budget(q)


class TestObjectionDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "why not XLE instead of SMH?",
            "I disagree, semis are too concentrated",
            "shouldn't we be trimming here?",
            "that's too much tech",
            "I think we should hold more cash",
            "rather than SMH, why not energy",
            "I'm not convinced by that thesis",
        ],
    )
    def test_pushback_is_flagged(self, text: str) -> None:
        assert looks_like_objection(text)

    @pytest.mark.parametrize(
        "text",
        [
            "what is XLV?",
            "how much cash do we have?",
            "show me the positions",
            "is the MU thesis still valid?",
            "what happened after we bought it?",
            "",
        ],
    )
    def test_plain_questions_are_not_objections(self, text: str) -> None:
        """Precision over recall: putting words in the operator's mouth in
        front of the committee is worse than missing one objection."""
        assert not looks_like_objection(text)


class TestSymbolExtraction:
    def test_only_known_symbols_are_extracted(self) -> None:
        got = extract_symbols("why not XLE instead of SMH?", {"XLE", "SMH", "MU"})
        assert got == ["SMH", "XLE"]

    def test_without_a_known_set_nothing_is_guessed(self) -> None:
        # Otherwise "I" and "OK" become tickers.
        assert extract_symbols("I think OK is fine", None) == []

    def test_unknown_uppercase_words_are_dropped(self) -> None:
        assert extract_symbols("WHY NOT XLE", {"XLE"}) == ["XLE"]


class TestRecordExchange:
    def test_objection_is_journaled_as_such(self, tmp_path: Path) -> None:
        kind = record_exchange(
            tmp_path,
            "why not XLE instead of SMH?",
            "SMH ranked 2, XLE ranked 14.",
            known_symbols={"XLE", "SMH"},
        )
        assert kind == "operator_objection"
        found = recent_objections(tmp_path)
        assert len(found) == 1
        assert "why not XLE" in found[0]["said"]
        assert set(found[0]["symbols"]) == {"SMH", "XLE"}

    def test_ordinary_question_is_journaled_as_chat(self, tmp_path: Path) -> None:
        kind = record_exchange(tmp_path, "what is XLV?", "healthcare sector ETF")
        assert kind == "chat"
        assert recent_objections(tmp_path) == []

    def test_exchange_updates_the_thread(self, tmp_path: Path) -> None:
        record_exchange(tmp_path, "why did we buy MU?", "rank 3", symbol="MU")
        turns, sym = Thread(tmp_path).load()
        assert len(turns) == 2 and sym == "MU"

    def test_objections_come_back_newest_first(self, tmp_path: Path) -> None:
        record_exchange(tmp_path, "why not XLE instead?", "a", known_symbols={"XLE"})
        record_exchange(tmp_path, "I disagree with the semis weight", "b")
        found = recent_objections(tmp_path, limit=5)
        assert "disagree" in found[0]["said"]

    def test_limit_is_respected(self, tmp_path: Path) -> None:
        for i in range(8):
            record_exchange(tmp_path, f"why not swap {i}?", "ok")
        assert len(recent_objections(tmp_path, limit=3)) == 3

    def test_thread_survives_an_unwritable_journal(self, tmp_path: Path, monkeypatch) -> None:
        """Conversation continuity must not depend on the memory store."""
        import trading.memory.store as store_mod

        def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(store_mod, "MemoryStore", boom)
        assert record_exchange(tmp_path, "why not XLE?", "reply") is None
        turns, _ = Thread(tmp_path).load()
        assert len(turns) == 2  # thread still updated


class TestEngineIntegration:
    """The thread has to actually reach the prompt, and the exchange has
    to actually be recorded, or none of the above matters."""

    def _answer(self, tmp_path: Path, question: str, captured: dict, symbol=None):
        from trading.copilot.engine import answer

        def fake_llm(system: str, prompt: str) -> str:
            captured["system"] = system
            captured["evidence"] = json.loads(prompt)
            return "an answer"

        return answer(
            question,
            state_dir=tmp_path,
            data_dir=tmp_path / "data",
            symbol=symbol,
            llm=fake_llm,
        )

    def test_budget_and_turns_reach_the_prompt(self, tmp_path: Path) -> None:
        cap: dict = {}
        self._answer(tmp_path, "what is XLV?", cap)
        assert "1-2 sentences" in cap["evidence"]["answer_budget"]
        assert cap["evidence"]["CHAT_recent_turns"] == []

        cap2: dict = {}
        self._answer(tmp_path, "and how big is it?", cap2)
        turns = cap2["evidence"]["CHAT_recent_turns"]
        assert any("what is XLV" in t["text"] for t in turns)

    def test_charter_forbids_treating_turns_as_evidence(self, tmp_path: Path) -> None:
        cap: dict = {}
        self._answer(tmp_path, "what is XLV?", cap)
        charter = cap["system"].lower()
        assert "never as a source of fact" in charter
        assert "the evidence wins" in charter

    def test_answering_records_the_exchange(self, tmp_path: Path) -> None:
        self._answer(tmp_path, "why not XLE instead of SMH?", {})
        found = recent_objections(tmp_path)
        assert len(found) == 1 and "why not XLE" in found[0]["said"]

    def test_carried_symbol_resolves_a_bare_followup(self, tmp_path: Path) -> None:
        """'and the thesis?' names no ticker — it must inherit MU.

        With an empty journal the engine takes its honest no-evidence
        path before calling the LLM, so the proof that the symbol
        resolved is that the refusal names MU rather than being generic."""
        Thread(tmp_path).append("operator", "why did we buy MU?", symbol="MU")
        out = self._answer(tmp_path, "and the thesis?", {})
        assert "MU" in out

    def test_bare_followup_without_a_thread_stays_generic(self, tmp_path: Path) -> None:
        """The counterpart: with no thread there is no symbol to inherit,
        so the same question must NOT invent one."""
        cap: dict = {}
        self._answer(tmp_path, "and the thesis?", cap)
        assert cap["evidence"]["symbol"] is None


class TestCommitteeSeesObjections:
    def test_context_carries_recent_objections(self, tmp_path: Path) -> None:
        from trading.agents.context import build_context

        record_exchange(
            tmp_path, "why not XLE instead of SMH?", "reply", known_symbols={"XLE", "SMH"}
        )
        ctx = build_context(tmp_path, tmp_path / "data")
        assert ctx["operator_objections"]
        assert "why not XLE" in ctx["operator_objections"][0]["said"]

    def test_manager_is_told_they_are_views_not_instructions(self) -> None:
        """Deferring to the operator when the data says otherwise is the
        failure mode; the charter has to say so."""
        from trading.agents.committee import MANAGER_CHARTER

        c = MANAGER_CHARTER.lower()
        assert "standing view" in c
        assert "not as an instruction" in c
        assert "deferring to him when he is wrong" in c

    def test_risk_officer_and_coach_see_them(self) -> None:
        from trading.agents.committee import _VIEW_KEYS

        assert "operator_objections" in _VIEW_KEYS["risk_officer"]
        assert "operator_objections" in _VIEW_KEYS["position_coach"]

    def test_creative_stays_position_blind(self) -> None:
        """Objections name held symbols — leaking them would break the
        deliberate position-blindness of the creative voice."""
        from trading.agents.committee import _VIEW_KEYS

        assert "operator_objections" not in _VIEW_KEYS["creative"]


class TestAccuracyDiscipline:
    """The charter must tell the model what it cannot see, or it fills
    gaps from general knowledge — and an answer from memory looks exactly
    like an answer from evidence."""

    def test_charter_lists_what_it_cannot_access(self) -> None:
        from trading.copilot.engine import CHARTER

        c = CHARTER.lower()
        for missing in ("fundamentals", "analyst ratings", "live or intraday quotes"):
            assert missing in c

    def test_charter_forbids_substituting_general_knowledge(self) -> None:
        from trading.copilot.engine import CHARTER

        c = CHARTER.lower()
        assert "do not substitute general knowledge" in c
        assert "guessing is the one unrecoverable failure" in c

    def test_charter_requires_naming_the_missing_piece(self) -> None:
        from trading.copilot.engine import CHARTER

        assert "say plainly which piece you cannot access" in CHARTER.lower()

    def test_charter_handles_ambiguity_without_announcing_it(self) -> None:
        """Rewritten 2026-07-30. The rule used to say "say what you took it
        to mean in your first clause", and the model complied on every
        reply — "I'm taking your message to mean..." opened every message,
        unambiguous ones included. Showing the assumed reading inside the
        answer conveys the same thing without the tic; a clarifying
        question stays the fallback when a question truly cannot be
        answered."""
        from trading.copilot.engine import CHARTER

        c = CHARTER.lower()
        assert "pick the most likely reading" in c
        assert "never open with a formula" in c
        assert "one short clarifying question" in c

    def test_charter_treats_a_replied_to_alert_as_the_subject(self) -> None:
        from trading.copilot.engine import CHARTER

        c = CHARTER.lower()
        assert "chat_operator_is_replying_to" in c
        assert "rather than inventing a cause" in c


class TestRepliedToEvidence:
    def _evidence(self, tmp_path: Path, question: str, replied_to: str | None) -> dict:
        from trading.copilot.engine import answer

        cap: dict = {}

        def fake_llm(system: str, prompt: str) -> str:
            cap["ev"] = json.loads(prompt)
            return "ok"

        answer(
            question,
            state_dir=tmp_path,
            data_dir=tmp_path / "data",
            replied_to=replied_to,
            llm=fake_llm,
        )
        return cap["ev"]

    def test_replied_to_message_reaches_the_prompt(self, tmp_path: Path) -> None:
        ev = self._evidence(tmp_path, "why this alert?", "Sentinel CAUTION: SPY -1.6%")
        assert "SPY" in ev["CHAT_operator_is_replying_to"]

    def test_absent_when_not_a_reply(self, tmp_path: Path) -> None:
        ev = self._evidence(tmp_path, "how much cash do we have?", None)
        assert ev["CHAT_operator_is_replying_to"] is None
