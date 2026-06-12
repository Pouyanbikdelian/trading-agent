"""Historian — hermetic tests: the LLM proposes, the store's mechanics dispose."""

from __future__ import annotations

from typing import Any

import pytest

from trading.agents.historian import format_historian_digest, run_historian
from trading.memory import MemoryStore


@pytest.fixture
def mem(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


def test_creates_capped_lessons_and_votes(mem: MemoryStore) -> None:
    existing = mem.add_lesson("Sharp corrections inside uptrends resolve upward within 10 days")

    def llm(system: str, prompt: str) -> dict[str, Any]:
        assert "Historian" in system
        assert existing in prompt  # sees the lesson book
        return {
            "new_lessons": [
                {"statement": "Gold breaking down while equities hold flags risk-on rotation"},
                {"statement": "Crowded sector momentum unwinds fastest in the first 2 days"},
                {"statement": "A third lesson beyond the cap should be ignored entirely"},
            ],
            "votes": [
                {"lesson_id": existing, "supports": True, "why": "this week confirmed"},
                {"lesson_id": "ls-hallucinated", "supports": True, "why": "n/a"},
            ],
            "retire": [{"lesson_id": existing, "why": "not established; must be ignored"}],
        }

    digest = run_historian(mem, llm=llm)
    assert digest["ok"] is True
    assert len(digest["created"]) == 2  # cap enforced
    assert digest["voted"] == 1  # hallucinated id ignored
    assert digest["retired"] == 0  # candidates cannot be retired
    rows = mem.lessons()
    assert len(rows) == 3
    assert all(r["status"] == "candidate" for r in rows)  # nothing auto-established


def test_garbage_statements_skipped_and_empty_week_ok(mem: MemoryStore) -> None:
    digest = run_historian(
        mem, llm=lambda s, p: {"new_lessons": [{"statement": "be careful"}], "votes": []}
    )
    assert digest["created"] == []  # too short -> garbage guard
    assert "no new lessons" in format_historian_digest(digest)


def test_promotion_needs_three_weeks_of_support(mem: MemoryStore) -> None:
    lid = mem.add_lesson("In low-IV uptrends, dip buys recover within five sessions")

    def llm_vote(s: str, p: str) -> dict[str, Any]:
        return {"new_lessons": [], "votes": [{"lesson_id": lid, "supports": True}]}

    # NB: evidence is keyed by week tag; same-day reruns dedupe via INSERT
    # OR IGNORE, so simulate weeks by voting directly.
    mem.add_evidence(lid, "wk-1", supports=True)
    mem.add_evidence(lid, "wk-2", supports=True)
    assert mem.lessons(status="candidate")[0]["id"] == lid
    mem.add_evidence(lid, "wk-3", supports=True)
    assert mem.lessons(status="established")[0]["id"] == lid
    run_historian(mem, llm=llm_vote)  # historian voting also counts
    assert mem.lessons(status="established")[0]["support"] == 4


def test_llm_failure_reported(mem: MemoryStore) -> None:
    def boom(s: str, p: str) -> dict[str, Any]:
        raise RuntimeError("LLM API 500")

    digest = run_historian(mem, llm=boom)
    assert digest["ok"] is False
    assert "skipped" in format_historian_digest(digest)


def test_stock_universe_clamp() -> None:
    from trading.agents.pm import _clamp_weights

    w = _clamp_weights({"NVDA": 0.3, "SMH": 0.3, "FAKE": 0.2}, stocks=("NVDA", "AAPL"))
    assert w["NVDA"] == 0.10  # single-stock cap
    assert w["SMH"] == 0.25  # ETF cap
    assert "FAKE" not in w
