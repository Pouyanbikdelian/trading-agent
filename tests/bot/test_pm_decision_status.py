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


def test_pm_review_exposes_exclusion_alternatives_and_missing_reasoning(tmp_path, monkeypatch):
    from trading.bot import telegram as tg

    monkeypatch.setattr(tg, "settings", SimpleNamespace(state_dir=tmp_path))
    p = tmp_path / "agent_pm"
    p.mkdir()
    (p / "last_run.json").write_text(
        json.dumps(
            {
                "ts": "2026-10-29T10:41:00+00:00",
                "candidate_review_plan": {
                    "blocked_mechanical": {"DD": "operator exclusion"},
                    "alternatives_to_review": ["DDOG"],
                },
                "reasoning_review": {
                    "complete": False,
                    "issues": ["DDOG: missing reason"],
                    "candidate_reviews": [],
                    "cash_reason": "not yet justified",
                },
            }
        )
    )
    output = tg._cmd_pm(["review"])
    assert "`DD` blocked: operator exclusion" in output
    assert "Alternatives requested: DDOG" in output
    assert "INCOMPLETE" in output and "DDOG: missing reason" in output
    assert not (tmp_path / "agent_pm_now.flag").exists()


def test_pm_review_supports_legacy_decisions(tmp_path, monkeypatch):
    from trading.bot import telegram as tg

    monkeypatch.setattr(tg, "settings", SimpleNamespace(state_dir=tmp_path))
    p = tmp_path / "agent_pm"
    p.mkdir()
    (p / "last_run.json").write_text(json.dumps({"weights": {"MPC": 0.03}}))
    assert "No structured candidate review" in tg._cmd_pm(["review"])


def test_pm_shadow_reads_saved_status_without_requesting_a_run(tmp_path, monkeypatch):
    from trading.bot import telegram as tg

    monkeypatch.setattr(tg, "settings", SimpleNamespace(state_dir=tmp_path))
    assert "No PM shadow grading status" in tg._cmd_pm(["shadow"])
    p = tmp_path / "agent_pm"
    p.mkdir()
    (p / "shadow_status.json").write_text(
        json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(), "counts": {"pending": 3}})
    )
    assert "pending: 3" in tg._cmd_pm(["shadow"])
    assert not (tmp_path / "agent_pm_now.flag").exists()


def test_review_exposes_claim_uncertainty_sizing_and_cash_deadline():
    from trading.agents.pm_reasoning import _format_candidate_review

    output = _format_candidate_review(
        {
            "reasoning_review": {
                "claims": [
                    {
                        "id": "c1",
                        "statement": "event rumor",
                        "assessment": "uncertain",
                        "evidence_ids": ["news:1"],
                        "verification_needed": "issuer filing",
                    }
                ],
                "sizing_reviews": [
                    {"symbol": "AAA", "target_weight": "bad", "invalidation": "thesis fails"}
                ],
                "cash_plan": {"review_at": "2026-10-31T12:00:00+00:00", "best_alternative": "BBB"},
            }
        }
    )
    assert "issuer filing" in output and "not verified" in output
    assert "target=bad" in output and "thesis fails" in output
    assert "2026-10-31T12:00:00+00:00" in output and "best alternative: BBB" in output


def test_pm_accept_forwards_exact_id_and_full_reason_without_requesting_a_run(
    tmp_path, monkeypatch
):
    from trading.agents import pm_audit
    from trading.bot import telegram as tg

    monkeypatch.setattr(tg, "settings", SimpleNamespace(state_dir=tmp_path))
    calls = []

    def accept(state, decision, reason):
        calls.append((state, decision, reason))
        return "acknowledged"

    monkeypatch.setattr(pm_audit, "_accept_review", accept)
    assert "exact decision ID" in tg._cmd_pm(["accept"])
    assert not calls
    assert tg._cmd_pm(["accept", "pm-exact", "Reviewed", "event", "uncertainty"]) == "acknowledged"
    assert calls == [(tmp_path, "pm-exact", "Reviewed event uncertainty")]
    assert not (tmp_path / "agent_pm_now.flag").exists()
