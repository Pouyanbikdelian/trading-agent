"""The historical PM simulation must never masquerade as a usable decision."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("state", ["failed", "missing", "stale", "fresh", "disabled"])
def test_pm_card_uses_execution_eligibility(tmp_path, monkeypatch, state):
    from trading.agents import pm
    from trading.bot import telegram as tg

    monkeypatch.setattr(
        tg,
        "settings",
        SimpleNamespace(state_dir=tmp_path, agent_pm_sleeve_pct=0 if state == "disabled" else 1),
    )
    monkeypatch.setattr(pm, "performance", lambda *_: {"return_pct": 1.0})
    monkeypatch.setattr(pm, "_marks_age_note", lambda *_: " · marks 22h old")
    monkeypatch.setattr(tg, "_live_equity_for_pm", lambda: (70000, "CHF", True))
    p = tmp_path / "agent_pm"
    p.mkdir()
    (p / "portfolio.json").write_text(json.dumps({"holdings": {}, "cash": 1000000}))
    decision = {
        "ok": state != "failed",
        "ts": (
            datetime.now(timezone.utc) - timedelta(hours=8 if state == "stale" else 0)
        ).isoformat(),
        "weights": {"MSFT": 0.1},
        "reason": "candidate preparation failed: stale prices",
    }
    if state != "missing":
        (p / "last_run.json").write_text(json.dumps(decision))
    out = tg._cmd_pm([])
    if state == "fresh":
        assert "fresh decision" in out
        assert "Historical weights" not in out
    else:
        assert "execution unavailable" in out
        assert "Historical weights" in out
        assert "live targets" not in out
        assert "deployed ≈" not in out
