"""Sentinel — hermetic tests: injected moves, injected LLM, tmp state.

Two behaviours under test:
* run_sentinel — informational tripwire alerts, debounced, NEVER convenes.
* run_late_day_derisk — the one price-driven committee trigger: a holding
  down >= threshold, once per day, mechanical (no LLM).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.runtime.sentinel import (
    check_triggers,
    format_derisk_alert,
    format_sentinel_alert,
    run_late_day_derisk,
    run_sentinel,
)


def _seed_book(tmp_path: Path, holdings: dict[str, float]) -> None:
    pm = tmp_path / "agent_pm"
    pm.mkdir(parents=True, exist_ok=True)
    (pm / "portfolio.json").write_text(json.dumps({"holdings": holdings, "cash": 0}))


def _info(_s: str, _p: str) -> dict[str, Any]:
    return {
        "severity": "alarm",
        "assessment": "correlated selling with a vol spike",
        "suggested_action": "consider /mode defense",
    }


def test_quiet_market_no_triggers_no_llm(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10})

    def boom(s: str, p: str) -> dict[str, Any]:
        raise AssertionError("LLM must not be called on a quiet market")

    moves = {"SPY": -0.3, "^VIX": 2.0, "SMH": -1.0}
    assert check_triggers(tmp_path, moves=moves) == []
    assert run_sentinel(tmp_path, llm=boom, moves=moves) == {"quiet": True}


def test_sentinel_alerts_but_never_convenes(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10, "MU": 5})
    moves = {"SPY": -2.1, "^VIX": 28.0, "SMH": -6.2, "MU": -3.0}
    trig = check_triggers(tmp_path, moves=moves)
    assert any("SPY" in t for t in trig)
    assert any("SMH" in t for t in trig)
    assert not any("MU" in t for t in trig)  # -3% is below the wire

    res = run_sentinel(tmp_path, llm=_info, moves=moves)
    assert res["quiet"] is False and res["severity"] == "alarm"
    # information only: no committee decision leaves this path, ever
    assert "convene_committee" not in res and "convene" not in res
    text = format_sentinel_alert(res)
    assert "🚨" in text and "SPY" in text
    assert "Convening" not in text and "Info only" in text


def test_debounce_suppresses_repeat_alerts(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10})
    moves = {"SPY": -2.5, "^VIX": 25.0, "SMH": -7.0}

    t0 = datetime.now(tz=timezone.utc)
    first = run_sentinel(tmp_path, llm=_info, moves=moves, now=t0)
    assert first["quiet"] is False
    again = run_sentinel(tmp_path, llm=_info, moves=moves, now=t0 + timedelta(minutes=30))
    assert again.get("debounced") is True
    later = run_sentinel(tmp_path, llm=_info, moves=moves, now=t0 + timedelta(hours=3))
    assert later["quiet"] is False  # debounce window expired


def test_new_or_worse_name_bypasses_debounce(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10, "CIEN": 5})
    t0 = datetime.now(tz=timezone.utc)
    m1 = {"SPY": -0.2, "^VIX": 5.0, "SMH": -6.0, "CIEN": -1.0}
    first = run_sentinel(tmp_path, llm=_info, moves=m1, now=t0)
    assert first["quiet"] is False

    # 20 min later CIEN newly breaches its wire — inside the debounce, re-pings
    m2 = {"SPY": -0.2, "^VIX": 5.0, "SMH": -6.0, "CIEN": -6.5}
    nxt = run_sentinel(tmp_path, llm=_info, moves=m2, now=t0 + timedelta(minutes=20))
    assert nxt["quiet"] is False
    assert any("CIEN" in t for t in nxt["triggers"])


def test_llm_failure_still_alerts(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"SMH": 10})

    def boom(s: str, p: str) -> dict[str, Any]:
        raise RuntimeError("LLM API 500")

    res = run_sentinel(tmp_path, llm=boom, moves={"SPY": -3.0, "^VIX": 30.0, "SMH": -8.0})
    assert res["quiet"] is False
    assert res["severity"] == "caution"  # blind alert beats silence


# --- late-day de-risk: the only automatic, price-driven committee trigger ---


def test_lateday_derisk_convenes_on_big_drop(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"MU": 5, "SNDK": 5})
    res = run_late_day_derisk(tmp_path, moves={"MU": -12.4, "SNDK": -3.0})
    assert res["quiet"] is False and res["convene"] is True
    assert "MU" in res["symbols"] and "SNDK" not in res["symbols"]
    assert "MU" in format_derisk_alert(res)


def test_lateday_derisk_quiet_below_threshold(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"MU": 5})
    # down 8% is a Sentinel info alert, but below the 10% committee bar
    assert run_late_day_derisk(tmp_path, moves={"MU": -8.0}) == {"quiet": True}


def test_lateday_derisk_runs_once_per_day(tmp_path: Path) -> None:
    _seed_book(tmp_path, {"MU": 5})
    t0 = datetime(2026, 7, 7, 19, 10, tzinfo=timezone.utc)  # ~15:10 ET
    first = run_late_day_derisk(tmp_path, moves={"MU": -14.0}, now=t0)
    assert first["convene"] is True
    second = run_late_day_derisk(tmp_path, moves={"MU": -16.0}, now=t0 + timedelta(minutes=5))
    assert second["quiet"] is True and second.get("already_convened") is True
