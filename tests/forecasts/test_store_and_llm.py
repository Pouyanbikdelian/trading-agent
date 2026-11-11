"""The ledger grades honestly, and the LLM's answers are taken literally."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.forecasts import llm, store


def _row(asof, symbol, horizon, source, prob):
    return {"asof": asof, "symbol": symbol, "horizon": horizon, "source": source, "prob": prob}


def test_a_forecast_once_written_cannot_be_replaced(tmp_path) -> None:
    assert store.save(tmp_path, [_row("2026-11-10", "S00", 5, "baseline", 0.55)]) == 1
    assert store.save(tmp_path, [_row("2026-11-10", "S00", 5, "baseline", 0.95)]) == 0
    assert store.has_forecasts(tmp_path, "2026-11-10", "baseline")


@pytest.mark.parametrize("prob", [0.0, 1.0, 1.2, float("nan"), float("inf")])
def test_impossible_probabilities_are_not_stored(tmp_path, prob) -> None:
    assert store.save(tmp_path, [_row("2026-11-10", "S00", 5, "baseline", prob)]) == 0


def test_grading_waits_for_a_finished_bar_then_scores_excess(tmp_path, panel) -> None:
    close, bench = panel
    start = len(close.index) - 6
    asof = str(close.index[start].date())
    store.save(
        tmp_path, [_row(asof, "S00", 5, "baseline", 0.6), _row(asof, "S00", 21, "baseline", 0.6)]
    )

    # The outcome bar is the newest one; "now" is still that day.
    same_day = close.index[-1].to_pydatetime().replace(hour=23)
    assert store.grade_due(tmp_path, close, bench, now=same_day)["graded"] == 0

    next_day = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
    counts = store.grade_due(tmp_path, close, bench, now=next_day)
    assert counts == {"graded": 1, "waiting": 1, "unpriced": 0}
    expected = close["S00"].iloc[-1] / close["S00"].iloc[start] - bench.iloc[-1] / bench.iloc[start]
    summary = store.summary(tmp_path)
    five = next(s for s in summary["sources"] if s["horizon"] == 5)
    assert five["graded"] == 1 and summary["open"] == 1
    assert five["hit_rate"] == float((0.6 > 0.5) == (expected > 0))
    assert "too early" in five["verdict"]


def test_the_llm_is_compared_with_the_baseline_on_the_same_forecasts(tmp_path, panel) -> None:
    close, bench = panel
    rows = []
    for back in range(40, 60):
        asof = str(close.index[len(close.index) - back].date())
        for symbol in ("S00", "S29"):
            realised_up = symbol == "S29"  # the drift makes S29 the winner
            rows.append(_row(asof, symbol, 5, "baseline", 0.5))
            rows.append(_row(asof, symbol, 5, "llm:test", 0.9 if realised_up else 0.1))
    rows.append(_row(str(close.index[-45].date()), "S15", 5, "llm:test", 0.5))  # no baseline twin
    store.save(tmp_path, rows)
    store.grade_due(tmp_path, close, bench, now=datetime(2026, 11, 20, tzinfo=timezone.utc))
    paired = store.summary(tmp_path)["paired"]
    assert len(paired) == 1
    assert paired[0]["common_forecasts"] == 40  # the unpaired forecast is left out
    assert paired[0]["brier_baseline"] == 0.25
    assert paired[0]["independent_periods"] == 4  # 20 days, one per 5 sessions
    assert "too early" in paired[0]["verdict"]


def test_only_well_formed_answers_for_stocks_that_were_asked_survive() -> None:
    response = {
        "forecasts": [
            {"symbol": "aapl", "p5": 0.52, "p21": 0.58, "why": "x"},
            {"symbol": "MSFT", "p5": 1.4, "p21": 0.5},
            {"symbol": "NVDA", "p5": "high", "p21": True},
            {"symbol": "TSLA", "p5": 0.6, "p21": 0.6},
            {"symbol": "AAPL", "p5": 0.9, "p21": 0.9},
            "nonsense",
        ]
    }
    parsed = llm.parse(response, ["AAPL", "MSFT", "NVDA"])
    assert parsed == {
        "AAPL": {"probs": {5: 0.52, 21: 0.58}, "why": "x"},
        "MSFT": {"probs": {21: 0.5}, "why": ""},
    }
    assert llm.parse({"forecasts": None}, ["AAPL"]) == {}


def test_one_failed_batch_loses_ten_stocks_not_the_run_and_calls_are_capped() -> None:
    cards = [{"symbol": f"X{i:03d}"} for i in range(140)]
    calls = []

    def call(system, prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise TimeoutError("slow")
        import json

        asked = [c["symbol"] for c in json.loads(prompt)["stocks"]]
        return {"forecasts": [{"symbol": s, "p5": 0.5, "p21": 0.5} for s in asked]}

    answers, errors = llm.forecast("2026-11-10", cards, call=call)
    assert len(calls) == llm.MAX_CALLS
    assert len(answers) == 90 and len(errors) == 1 and "TimeoutError" in errors[0]
    assert "between 0.45 and 0.55" in llm.SYSTEM
