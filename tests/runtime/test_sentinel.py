"""Sentinel — hermetic tests: injected moves, injected LLM, tmp state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.runtime.sentinel import check_triggers, format_sentinel_alert, run_sentinel


def _seed_book(tmp_path: Path, holdings: dict[str, float]) -> None:
    pm = tmp_path / "agent_pm"
    pm.mkdir(parents=True, exist_ok=True)
    (pm / "portfolio.json").write_text(json.dumps({"holdings": holdings, "cash": 0}))


def test_quiet_market_no_triggers_no_llm(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10})

    def boom(s: str, p: str) -> dict[str, Any]:
        raise AssertionError("LLM must not be called on a quiet market")

    moves = {"SPY": -0.3, "^VIX": 2.0, "SMH": -1.0}
    assert check_triggers(tmp_path, moves=moves) == []
    assert run_sentinel(tmp_path, llm=boom, moves=moves) == {"quiet": True}


def test_triggers_fire_and_escalate(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10, "MU": 5})
    moves = {"SPY": -2.1, "^VIX": 28.0, "SMH": -6.2, "MU": -3.0}
    trig = check_triggers(tmp_path, moves=moves)
    assert any("SPY" in t for t in trig)
    assert any("VIX" in t for t in trig)
    assert any("SMH" in t for t in trig)
    assert not any("MU" in t for t in trig)  # -3% is below the wire

    def llm(system: str, prompt: str) -> dict[str, Any]:
        assert "Sentinel" in system
        return {
            "severity": "alarm",
            "assessment": "Correlated selling with vol spike — systemic, not idiosyncratic.",
            "suggested_action": "consider /mode defense",
            "convene_committee": True,
        }

    res = run_sentinel(tmp_path, llm=llm, moves=moves)
    assert res["quiet"] is False and res["severity"] == "alarm"
    assert res["convene_committee"] is True
    text = format_sentinel_alert(res)
    assert "🚨" in text and "SPY" in text and "Convening" in text


def test_debounce_suppresses_repeat_alerts(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10})
    moves = {"SPY": -2.5, "^VIX": 25.0, "SMH": -7.0}
    llm = lambda s, p: {"severity": "caution", "assessment": "x", "suggested_action": "y"}  # noqa: E731

    t0 = datetime.now(tz=timezone.utc)
    first = run_sentinel(tmp_path, llm=llm, moves=moves, now=t0)
    assert first["quiet"] is False
    again = run_sentinel(tmp_path, llm=llm, moves=moves, now=t0 + timedelta(minutes=30))
    assert again.get("debounced") is True
    later = run_sentinel(tmp_path, llm=llm, moves=moves, now=t0 + timedelta(hours=3))
    assert later["quiet"] is False  # debounce window expired


def test_llm_failure_still_alerts(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10})

    def boom(s: str, p: str) -> dict[str, Any]:
        raise RuntimeError("LLM API 500")

    res = run_sentinel(tmp_path, llm=boom, moves={"SPY": -3.0, "^VIX": 30.0, "SMH": -8.0})
    assert res["quiet"] is False
    assert res["severity"] == "caution"  # blind escalation beats silence
