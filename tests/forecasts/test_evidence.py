"""Forecasts as PM evidence: an input the operator switched on, with limits."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from tests.agents.test_pm_audit_contract import valid_response
from trading.agents.pm import PM_CHARTER, format_pm_digest, run_agent_pm
from trading.core.clock import UtcClock
from trading.forecasts import store
from trading.forecasts.evidence import pm_evidence
from trading.memory import MemoryStore


def _switch(monkeypatch, on: bool) -> None:
    from trading.core import config

    monkeypatch.setattr(
        config, "settings", config.settings.model_copy(update={"forecast_pm_evidence": on})
    )


def _seed(state_dir, asof: str) -> None:
    rows = [
        {"asof": asof, "symbol": "AAA", "horizon": h, "source": src, "prob": p}
        for src, probs in (
            ("baseline", (0.52, 0.49)),
            ("llm:claude-sonnet-5-5:numbers", (0.5, 0.51)),
            ("llm:claude-sonnet-5-5:news", (0.58, 0.61)),
        )
        for h, p in zip((5, 21), probs, strict=True)
    ]
    store.save(state_dir, rows)
    store.write_status(
        state_dir,
        {
            "summary": {
                "sources": [{"source": "baseline", "horizon": 5, "graded": 0}],
                "paired": [],
            }
        },
    )


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr("trading.agents.pm._stock_universe", lambda: ("AAA",))
    mem = MemoryStore(tmp_path / "memory")
    now = UtcClock().now()
    context = {
        "account": {"equity": 100000, "base_currency": "USD"},
        "positions": [],
        "candidate_ladder": {
            "as_of": now.date().isoformat(),
            "ranked": [{"symbol": "AAA", "rank": 1, "score": 0.5}],
            "selected_mechanical": ["AAA"],
        },
    }
    prompts: list[dict] = []

    def run(response):
        def llm(system, prompt):
            prompts.append(json.loads(prompt))
            return response

        return run_agent_pm(context, mem, tmp_path, prices={"AAA": 100}, llm=llm)

    yield run, now, prompts
    mem.close()


def test_off_by_default_the_pm_never_sees_a_forecast(desk, tmp_path, monkeypatch) -> None:
    run, now, prompts = desk
    _seed(tmp_path, now.date().isoformat())
    _switch(monkeypatch, False)
    assert pm_evidence(tmp_path, {"AAA"}, now=now) is None
    result = run(valid_response(now))
    assert "shadow_forecasts" not in prompts[0]
    assert "forecast:AAA" not in prompts[0]["evidence_catalog"]
    assert result["target_forecasts"] is None


def test_switched_on_the_pm_gets_the_numbers_and_their_track_record(
    desk, tmp_path, monkeypatch
) -> None:
    run, now, prompts = desk
    _seed(tmp_path, now.date().isoformat())
    _switch(monkeypatch, True)
    result = run(valid_response(now))

    entry = prompts[0]["evidence_catalog"]["forecast:AAA"]
    assert entry["kind"] == "model_estimate"
    assert entry["facts"] == {
        "baseline_p5": 0.52,
        "baseline_p21": 0.49,
        "llm_numbers_p5": 0.5,
        "llm_numbers_p21": 0.51,
        "llm_news_p5": 0.58,
        "llm_news_p21": 0.61,
    }
    shown = prompts[0]["shadow_forecasts"]
    assert "forecasts" not in shown  # sent once, in the catalog
    assert shown["track_record"][0]["verdict"] == "no outcomes yet"
    assert "UNPROVEN" in shown["how_to_read"]

    # ...and the operator sees what the forecasters said about each pick.
    assert result["target_forecasts"]["targets"]["AAA"]["llm_news_p5"] == 0.58
    card = format_pm_digest(result, None)
    assert "Forecasts, unproven" in card
    assert "`AAA` model 52%/49% · LLM 50%/51% · LLM+news 58%/61%" in card
    assert not result["requires_operator_review"], result["reasoning_review"]["issues"]


def test_a_forecast_can_be_cited_only_as_uncertain(desk, tmp_path, monkeypatch) -> None:
    run, now, _ = desk
    _seed(tmp_path, now.date().isoformat())
    _switch(monkeypatch, True)

    honest = valid_response(now)
    honest["claims"].append(
        {
            "id": "c2",
            "statement": "the news forecaster leans positive",
            "assessment": "uncertain",
            "evidence_ids": ["forecast:AAA"],
            "verification_needed": "its graded track record",
        }
    )
    assert not run(honest)["requires_operator_review"]

    overclaimed = valid_response(now)
    overclaimed["claims"].append(
        {
            "id": "c2",
            "statement": "AAA will outperform",
            "assessment": "measured",
            "evidence_ids": ["forecast:AAA"],
            "numeric_check": {
                "evidence_id": "forecast:AAA",
                "field": "llm_news_p5",
                "operator": "ge",
                "value": 0.5,
            },
        }
    )
    issues = run(overclaimed)["reasoning_review"]["issues"]
    assert any("c2: claimed measurement is unsupported" in i for i in issues)


def test_stale_or_missing_forecasts_are_not_evidence(tmp_path, monkeypatch) -> None:
    now = UtcClock().now()
    _switch(monkeypatch, True)
    assert pm_evidence(tmp_path, {"AAA"}, now=now) is None  # no ledger at all
    _seed(tmp_path, (now - timedelta(days=9)).date().isoformat())
    assert pm_evidence(tmp_path, {"AAA"}, now=now) is None
    assert pm_evidence(tmp_path, set(), now=now) is None


def test_the_charter_limits_what_a_forecast_may_do() -> None:
    text = PM_CHARTER.replace("\n", " ")
    assert "never opens a position without other dated evidence" in text
    assert "never sets or raises a size" in text
    assert "must use assessment=uncertain" in text
