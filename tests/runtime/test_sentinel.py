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


def _alarm(_s: str, _p: str) -> dict[str, Any]:
    return {
        "severity": "alarm",
        "assessment": "correlated selling",
        "suggested_action": "consider /mode defense",
        "convene_committee": True,
    }


def test_persistent_same_names_reping_but_no_recommittee(tmp_path: Path) -> None:
    """A drawdown that just stays put should keep pinging (operator wants the
    heads-up) but NOT re-run the committee — the core fix."""
    _seed_book(tmp_path, {"MU": 5, "SNDK": 5})
    moves = {"SPY": -0.2, "^VIX": 5.0, "MU": -10.6, "SNDK": -10.6}  # only held names trip

    t0 = datetime.now(tz=timezone.utc)
    first = run_sentinel(tmp_path, llm=_alarm, moves=moves, now=t0)
    assert first["convene_committee"] is True  # initial escalation convenes

    # 3h later, identical names down the same amount: past the ping debounce
    # so the caution still fires, but the committee must not re-convene.
    later = run_sentinel(tmp_path, llm=_alarm, moves=moves, now=t0 + timedelta(hours=3))
    assert later["quiet"] is False
    assert later["convene_committee"] is False
    assert later["committee_suppressed"] is True
    assert "Committee held" in format_sentinel_alert(later)


def test_new_name_escalates_immediately(tmp_path: Path) -> None:
    """A fresh name cracking is real news: it bypasses the ping debounce and
    re-convenes even inside the quiet window."""
    _seed_book(tmp_path, {"MU": 5, "SNDK": 5, "CIEN": 5})
    t0 = datetime.now(tz=timezone.utc)

    moves1 = {"SPY": -0.2, "^VIX": 5.0, "MU": -10.6, "SNDK": -10.6, "CIEN": -1.0}
    first = run_sentinel(tmp_path, llm=_alarm, moves=moves1, now=t0)
    assert first["convene_committee"] is True

    # 30 min later — inside the debounce — but CIEN newly breaches its wire.
    moves2 = {"SPY": -0.2, "^VIX": 5.0, "MU": -10.6, "SNDK": -10.6, "CIEN": -6.0}
    nxt = run_sentinel(tmp_path, llm=_alarm, moves=moves2, now=t0 + timedelta(minutes=30))
    assert nxt["quiet"] is False  # bypassed the debounce
    assert nxt["convene_committee"] is True  # and re-convened (still under the cap)
    assert any("CIEN" in t for t in nxt["triggers"])


def test_daily_committee_cap(tmp_path: Path) -> None:
    """Even with successive genuine legs down, the sentinel stops convening the
    committee after MAX_COMMITTEE_PER_DAY (default 2)."""
    _seed_book(tmp_path, {"MU": 5})
    t0 = datetime.now(tz=timezone.utc)

    a = run_sentinel(tmp_path, llm=_alarm, moves={"MU": -6.0}, now=t0)
    b = run_sentinel(tmp_path, llm=_alarm, moves={"MU": -10.0}, now=t0 + timedelta(hours=1))
    c = run_sentinel(tmp_path, llm=_alarm, moves={"MU": -14.0}, now=t0 + timedelta(hours=2))

    assert a["convene_committee"] is True
    assert b["convene_committee"] is True  # -6 -> -10 is another material leg
    assert c["convene_committee"] is False  # cap reached
    assert "cap" in c["committee_suppressed_reason"]
