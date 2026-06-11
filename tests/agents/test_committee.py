"""Agent committee — hermetic tests with an injected fake LLM."""

from __future__ import annotations

from typing import Any

import pytest

from trading.agents.committee import CHARTERS, format_digest, run_committee
from trading.agents.llm import _extract_json
from trading.memory import MemoryStore


@pytest.fixture
def mem(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


def _take(stance: str, subject: str = "SPY", conf: float = 0.7) -> dict[str, Any]:
    return {
        "stance": stance,
        "take": f"{stance} take on {subject}",
        "prediction": {
            "subject": subject,
            "direction": "up" if stance == "bullish" else "down",
            "horizon_days": 5,
            "confidence": conf,
        },
        "sources": ["reuters"],
        "cited_lessons": [],
    }


def make_fake_llm(broken_agents: set[str] | None = None):
    """Returns (llm_fn, calls list). Routes on charter text."""
    calls: list[str] = []
    broken = broken_agents or set()

    def llm(system: str, prompt: str) -> dict[str, Any]:
        for name in CHARTERS:
            if f"the {name.replace('_', ' ').title()}" in system or name in system.lower():
                calls.append(name)
                if name in broken:
                    return {"nonsense": True}  # missing prediction -> rejected
                stance = {"risk_officer": "bearish", "trader": "bullish"}.get(name, "neutral")
                return _take(stance, conf=0.9 if name == "trader" else 0.6)
        if "Fund Manager" in system:
            calls.append("manager")
            return {
                "posture": "neutral",
                "proposal": "Stay the course; size nothing up until vol confirms.",
                "watch": "5y yield 5d move",
                "dissent_summary": "trader bullish vs risk officer bearish",
            }
        if "Challenger" in system:
            calls.append("challenger")
            return {
                "objections": [
                    {
                        "target_agent": "trader",
                        "objection": "tape strength is 3 days old; base rate says fade it",
                        "falsifier": "two more closes above the 20dma",
                    }
                ]
            }
        raise AssertionError(f"unknown charter: {system[:60]}")

    return llm, calls


def test_full_committee_flow_writes_memory(mem: MemoryStore) -> None:
    llm, calls = make_fake_llm()
    digest = run_committee({"positions": []}, mem, llm=llm)

    assert digest["ok"] is True
    assert set(digest["takes"]) == set(CHARTERS)
    assert calls.count("challenger") == 1 and calls.count("manager") == 1
    # disagreement: bullish(+1) vs bearish(-1) -> (1 - -1)/2 = 1.0
    assert digest["disagreement_index"] == pytest.approx(1.0)

    # Every take became a gradeable prediction + journal entries exist.
    s = mem.stats()
    assert s["predictions"] == len(CHARTERS)
    kinds = {e["kind"] for e in mem.journal_tail(50)}
    assert {"take", "debate", "committee"} <= kinds


def test_broken_agent_is_skipped_not_fatal(mem: MemoryStore) -> None:
    llm, _ = make_fake_llm(broken_agents={"street"})
    digest = run_committee({}, mem, llm=llm)
    assert digest["ok"] is True
    assert "street" not in digest["takes"]
    assert mem.stats()["predictions"] == len(CHARTERS) - 1


def test_all_agents_failing_is_reported(mem: MemoryStore) -> None:
    digest = run_committee({}, mem, llm=lambda s, p: {"junk": 1})
    assert digest["ok"] is False


def test_format_digest_is_telegram_friendly(mem: MemoryStore) -> None:
    llm, _ = make_fake_llm()
    text = format_digest(run_committee({}, mem, llm=llm))
    assert "Daily committee" in text
    assert "Challenger" in text
    assert "Manager" in text
    assert "Disagreement index" in text
    assert len(text) < 4000  # single Telegram message


def test_extract_json_handles_prose_wrapping() -> None:
    assert _extract_json('Sure! Here: {"a": {"b": 1}} hope that helps')["a"]["b"] == 1
    with pytest.raises(ValueError):
        _extract_json("no json here")
