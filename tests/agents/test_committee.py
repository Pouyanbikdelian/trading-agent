"""Agent committee — hermetic tests with an injected fake LLM."""

from __future__ import annotations

import copy
import json
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
        if "Challenger" in system and "Fund Manager" not in system:
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
    parsed = json.loads(seen["prompt"])
    assert parsed["market_context"]["macro_dial"] == {"composite": 1.2}
    for name in CHARTERS:
        assert name in parsed["takes"]
    assert digest["market_caveat"].startswith("late-stage")
    assert digest["objections"][0]["target_agent"] == "committee"


def test_the_challenger_keeps_up_to_ten_objections_and_a_full_caveat(mem: MemoryStore) -> None:
    """2026-09-26: capped at five objections and a 300-char raw slice, while
    the charter asks it for every material weakness."""
    from trading.agents.committee import MAX_OBJECTIONS

    base_llm, _ = make_fake_llm()
    caveat = "Late-cycle fragility: " + "breadth is narrowing while rates rise. " * 20

    def spy_llm(system: str, prompt: str):
        if "Challenger" in system and "Fund Manager" not in system:
            return {
                "objections": [
                    {"target_agent": "committee", "objection": f"o{i}", "falsifier": "f"}
                    for i in range(14)
                ],
                "market_phase_caveat": caveat,
            }
        return base_llm(system, prompt)

    digest = run_committee({"macro_dial": {"composite": 1.2}}, mem, llm=spy_llm)
    assert MAX_OBJECTIONS == 10 and len(digest["objections"]) == 10
    assert len(digest["market_caveat"]) > 300


def test_manager_bounds_mapping_takes_without_erasing_dissent() -> None:
    from trading.agents.committee import _budgeted_manager_prompt

    takes = {
        name: {**_take("bullish", name.upper()), "take": "full rationale " * 800}
        for name in CHARTERS
    }
    objections = [
        {
            "target_agent": "committee",
            "objection": "Crowded trade",
            "falsifier": "Broad earnings acceleration",
        }
    ]
    payload = {
        "takes": takes,
        "objections": objections,
        "guard_flags": ["Concentration elevated"],
        "established_lessons": [{"id": "L1", "lesson": "Protect downside"}],
        "calibration": [{"agent": "quant", "n": 50, "hit_rate": 0.52}],
    }
    original = copy.deepcopy(payload)
    rendered = _budgeted_manager_prompt(payload, budget=6000)
    parsed = json.loads(rendered)

    assert len(rendered) <= 6000
    assert set(parsed["takes"]) == set(CHARTERS)
    assert parsed["objections"] == objections
    for key in ("guard_flags", "calibration", "established_lessons"):
        assert parsed[key] == payload[key]
    for name, take in takes.items():
        assert parsed["takes"][name]["prediction"] == take["prediction"]
        assert parsed["takes"][name]["stance"] == take["stance"]
    assert payload == original


def test_protected_prompt_overflow_is_explicit() -> None:
    from trading.agents.committee import _budgeted_context, _budgeted_manager_prompt

    lessons = [{"id": "L1", "lesson": "Never discard conditions " * 1000}]
    with pytest.raises(ValueError, match="protected manager evidence"):
        _budgeted_manager_prompt(
            {"takes": {"quant": _take("bullish")}, "established_lessons": lessons}, budget=1000
        )
    with pytest.raises(ValueError, match="protected specialist context"):
        _budgeted_context({"established_lessons": lessons}, budget=1000)


def test_busy_committee_prompts_are_valid_bounded_and_keep_all_voices(mem: MemoryStore) -> None:
    from trading.agents.committee import (
        CHALLENGER_PROMPT_BUDGET,
        MANAGER_PROMPT_BUDGET,
        SPECIALIST_PROMPT_BUDGET,
    )

    context = {
        "account": {"equity": 100000, "base_currency": "CHF"},
        "positions": [{"symbol": "SPY", "qty": 10}],
        "macro_dial": {"composite": -1.3},
        "headlines": [{"title": "News " * 800, "source": "reuters"} for _ in range(48)],
        "candidate_ladder": {
            "source": "runner",
            "ranked": [{"symbol": f"T{i}", "rank": i, "score": 0.2} for i in range(25)],
        },
        "_data_gaps": ["vol_surface stale: treat as unknown"],
        "established_lessons": [{"id": "L1", "lesson": "Respect evidence"}],
        "operator_objections": [{"text": "Question crowded momentum"}],
    }
    original = copy.deepcopy(context)
    prompts: dict[str, dict[str, Any]] = {}

    def spy_llm(system: str, prompt: str) -> dict[str, Any]:
        if "Challenger" in system and "Fund Manager" not in system:
            assert len(prompt) <= CHALLENGER_PROMPT_BUDGET
            prompts["challenger"] = json.loads(prompt)
            return {
                "objections": [
                    {
                        "target_agent": "committee",
                        "objection": "Crowding",
                        "falsifier": "Earnings breadth",
                    }
                ],
                "market_phase_caveat": "Late-cycle fragility",
            }
        if "Fund Manager" in system:
            assert len(prompt) <= MANAGER_PROMPT_BUDGET
            prompts["manager"] = json.loads(prompt)
            return {"posture": "neutral"}
        name = next(name for name, charter in CHARTERS.items() if charter == system)
        rendered = prompt.split("\n", 1)[1]
        assert len(rendered) <= SPECIALIST_PROMPT_BUDGET
        prompts[name] = json.loads(rendered)
        return {**_take("bullish", name.upper()), "take": "Long rationale " * 1000}

    result = run_committee(context, mem, llm=spy_llm)

    assert result["ok"]
    assert set(result["takes"]) == set(CHARTERS)
    assert set(prompts["challenger"]["takes"]) == set(CHARTERS)
    assert set(prompts["manager"]["takes"]) == set(CHARTERS)
    assert prompts["manager"]["objections"][0]["objection"] == "Crowding"
    assert prompts["manager"]["market_phase_caveat"] == "Late-cycle fragility"
    assert prompts["manager"]["operator_objections"] == context["operator_objections"]
    for name in CHARTERS:
        assert prompts[name]["_data_gaps"] == context["_data_gaps"]
    market = prompts["challenger"]["market_context"]
    for key in ("account", "positions", "macro_dial", "_data_gaps", "established_lessons"):
        assert market[key] == context[key]
    assert len(market["candidate_ladder"]["ranked"]) >= 8
    assert market["_prompt_omissions"]["headlines"] > 0
    assert context == original


def test_missing_challenger_is_visible_to_manager_operator_and_journal(mem: MemoryStore) -> None:
    from trading.agents.committee import format_digest_compact

    base_llm, _ = make_fake_llm()
    manager_payload = {}

    def broken_challenger(system: str, prompt: str) -> dict[str, Any]:
        if "Fund Manager" in system:
            manager_payload.update(json.loads(prompt))
        elif "Challenger" in system:
            raise RuntimeError("provider unavailable")
        return base_llm(system, prompt)

    digest = run_committee({}, mem, llm=broken_challenger)

    assert digest["unavailable_agents"] == ["challenger"]
    assert manager_payload["unavailable_agents"] == ["challenger"]
    assert "Incomplete review — unavailable: challenger" in format_digest_compact(digest)
    assert "Incomplete review — unavailable: challenger" in format_digest(digest)
    committee_row = next(row for row in mem.journal_tail(20) if row["kind"] == "committee")
    assert committee_row["payload"]["unavailable_agents"] == ["challenger"]
