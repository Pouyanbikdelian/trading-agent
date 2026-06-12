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


def test_compact_digest_is_short_and_pointed(mem: MemoryStore) -> None:
    from trading.agents.committee import format_digest_compact

    llm, _ = make_fake_llm()
    digest = run_committee({}, mem, llm=llm)
    text = format_digest_compact(digest)
    assert "Committee" in text and "Conclusion" in text and "/detail" in text
    assert len(text) < 1200  # executive summary, not a transcript
    # Compact is meaningfully shorter than the full rendering.
    assert len(text) < len(format_digest(digest))


def test_specialists_get_sliced_context_challenger_gets_all(mem: MemoryStore) -> None:
    """Anti-echo-chamber: the scout must not see the macro dial, the
    position coach must not see headlines; the challenger sees both."""
    prompts: dict[str, str] = {}
    base_llm, _ = make_fake_llm()

    def spy_llm(system: str, prompt: str):
        for name in CHARTERS:
            if f"the {name.replace('_', ' ').title()}" in system:
                prompts[name] = prompt
        if "professionally disagreeable" in system:
            prompts["challenger"] = prompt
        return base_llm(system, prompt)

    ctx = {"macro_dial": {"btc_confirm_z": -1.5}, "headlines": [{"title": "chips rip"}]}
    run_committee(ctx, mem, llm=spy_llm)
    assert "btc_confirm_z" not in prompts["scout"]
    assert "chips rip" in prompts["scout"]
    assert "chips rip" not in prompts["position_coach"]
    assert "btc_confirm_z" in prompts["quant"]
    assert "btc_confirm_z" in prompts["challenger"]


def test_display_names_escape_markdown(mem: MemoryStore) -> None:
    llm, _ = make_fake_llm()
    text = format_digest(run_committee({}, mem, llm=llm))
    assert "risk officer" in text and "risk_officer" not in text


def test_telegram_splitter_respects_limit_and_lines() -> None:
    from trading.bot.telegram import _split_for_telegram

    text = "\n".join(f"line {i} " + "x" * 80 for i in range(300))
    chunks = _split_for_telegram(text)
    assert all(len(c) <= 3800 + 20 for c in chunks)
    assert len(chunks) <= 4 and chunks[-1].endswith("…(truncated)")
    assert _split_for_telegram("short") == ["short"]


def test_challenger_sees_all_takes_and_market_context(mem: MemoryStore) -> None:
    seen: dict[str, str] = {}

    base_llm, _ = make_fake_llm()

    def spy_llm(system: str, prompt: str):
        if "Challenger" in system and "Fund Manager" not in system:
            seen["prompt"] = prompt
            return {
                "objections": [
                    {
                        "target_agent": "committee",
                        "objection": "consensus is crowded",
                        "falsifier": "breadth expansion",
                    }
                ],
                "market_phase_caveat": "late-stage rallies punish chasing",
            }
        return base_llm(system, prompt)

    digest = run_committee({"macro_dial": {"composite": 1.2}}, mem, llm=spy_llm)
    # Challenger received market context AND every agent's take.
    assert "Market context" in seen["prompt"]
    for name in CHARTERS:
        assert name in seen["prompt"] or name in str(digest["takes"])
    assert digest["market_caveat"].startswith("late-stage")
    assert digest["objections"][0]["target_agent"] == "committee"
