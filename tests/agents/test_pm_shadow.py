"""Matched cohorts cannot earn returns before entry or hide absent FX/data."""

import copy
from datetime import datetime, timezone

import pandas as pd
import pytest

from trading.reporting.pm_shadow import _evaluate_shadow, _format_shadow_status, _shadow_plan
from trading.risk.limits import RiskLimits

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


@pytest.fixture
def cohort():
    limits = RiskLimits(max_position_pct=1, max_sector_exposure=1).model_dump(mode="json")
    payload = {
        "today_context": {
            "operator_live_account_equity_NOT_YOUR_BOOK": {"base_currency": "CHF"},
            "candidate_ladder": {
                "mechanical_weights": {"AAA": 0.5, "BBB": 0.5},
                "shadow_risk_limits": limits,
                "shadow_sector_map": {"AAA": "health", "BBB": "energy"},
                "quote_currencies": {"AAA": "USD", "BBB": "USD"},
            },
        },
        "operator_excluded_never_buy": [],
        "operator_held_do_not_trade": [],
    }
    result = {
        "weights": {"AAA": 0.1},
        "ts": "2026-10-01T20:00:00+00:00",
        "reasoning_review": {"candidate_reviews": [{"symbol": "BBB", "decision": "pass"}]},
        "sentinel_scope_review": {"live_cap_unchanged": 0.7, "experimental_market_only_cap": 1},
    }
    plan = _shadow_plan(payload, result, 10)
    index = pd.date_range("2026-10-01", periods=7, freq="B", tz="UTC")
    # The enormous pre-entry move must not be earned by any cohort.
    prices = pd.DataFrame(
        {"AAA": [1, 100, 101, 102, 103, 104, 105], "BBB": [1, 100, 100, 100, 100, 100, 100]},
        index=index,
    )
    fx = pd.DataFrame({"USDCHF": [0.8] * 7}, index=index)
    return plan, prices, fx, payload, result


def test_no_lookahead_matched_dates_costs_cash_and_attribution(cohort):
    plan, prices, fx, _, _ = cohort
    report = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    assert report["status"] == "graded"
    assert report["entry_at"].startswith("2026-10-02")
    assert report["arms"]["cash"]["net_return"] == 0
    assert 0 < report["arms"]["pm"]["net_return"] < 0.005
    assert report["arms"]["pm"]["cost_fraction"] > 0
    assert report["arms"]["pm"]["turnover"] > 0.1
    effects = report["attribution"]
    assert effects["selection_and_sizing_at_matched_requested_gross"] + effects[
        "cash_allocation_effect"
    ] == pytest.approx(effects["pm_minus_mechanical"])
    assert report["arms"]["mechanical"]["entry_gross"] <= 0.7
    assert report["arms"]["experimental_scoped_sentinel_mechanical"]["entry_gross"] > 0.7


def test_fx_changes_affect_identical_base_currency_cohorts(cohort):
    plan, prices, fx, _, _ = cohort
    original = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    fx.iloc[-1] = 1
    changed = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    assert changed["arms"]["pm"]["net_return"] > original["arms"]["pm"]["net_return"]
    assert changed["arms"]["cash"]["net_return"] == 0


@pytest.mark.parametrize("gap", ["fx_column", "fx_date", "price_date", "price_symbol"])
def test_missing_data_is_unavailable_not_zero_or_fallback(cohort, gap):
    plan, prices, fx, _, _ = cohort
    if gap == "fx_column":
        fx = pd.DataFrame()
    elif gap == "fx_date":
        fx = fx.drop(fx.index[3])
    elif gap == "price_date":
        prices.loc[prices.index[3], "AAA"] = float("nan")
    else:
        prices = prices.drop(columns="AAA")
    report = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    assert report["status"] == "unavailable"
    assert "arms" not in report


def test_future_prices_are_not_graded_early(cohort):
    plan, prices, fx, _, _ = cohort
    report = _evaluate_shadow(plan, prices, fx, horizon=5, now=prices.index[3].to_pydatetime())
    assert report["status"] == "pending"


def test_same_sector_limits_bind_all_arms(cohort):
    plan, prices, fx, _, _ = cohort
    plan["sectors"] = {"AAA": "one", "BBB": "one"}
    plan["risk_limits"]["max_sector_exposure"] = 0.2
    report = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    assert all(a["entry_gross"] <= 0.2 for a in report["arms"].values())


def test_daily_loss_guard_exits_and_stays_in_cash(cohort):
    plan, prices, fx, _, _ = cohort
    prices["AAA"] = [1, 100, 50, 50, 100, 120, 130]
    plan["arms"]["pm"] = {"AAA": 0.5}
    report = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    assert report["arms"]["pm"]["risk_exit"]
    assert report["arms"]["pm"]["net_return"] < -0.2


def test_exclusions_hold_in_every_counterfactual(cohort):
    _, _, _, payload, result = cohort
    payload["operator_excluded_never_buy"] = ["AAA"]
    plan = _shadow_plan(payload, result, 10)
    assert all("AAA" not in weights for weights in plan["arms"].values())


def test_missing_mechanical_or_sector_metadata_is_explicit(cohort):
    _, _, _, payload, result = cohort
    missing = copy.deepcopy(payload)
    del missing["today_context"]["candidate_ladder"]["mechanical_weights"]
    assert _shadow_plan(missing, result, 10)["status"] == "unavailable"
    del payload["today_context"]["candidate_ladder"]["shadow_sector_map"]["AAA"]
    assert _shadow_plan(payload, result, 10)["status"] == "unavailable"


def test_grading_failures_and_sample_size_are_visible(cohort):
    plan, prices, fx, _, _ = cohort
    grade = _evaluate_shadow(plan, prices, fx, horizon=5, now=NOW)
    text = _format_shadow_status(
        {"checked_at": "2026-01-01T00:00:00+00:00", "counts": {"graded": 1}, "records": [grade]}
    )
    assert "overdue" in text and "1 overlapping cohorts" in text
    assert "not independent" in text
    assert "Grading failed" in _format_shadow_status({"status": "failed", "reason": "bad FX"})


def test_grader_records_exact_forward_data_once_and_reports_unmatured_legs(cohort, tmp_path):
    from trading.agents.pm_audit import _archive_decision, _audit_dir, _start_audit
    from trading.reporting.pm_shadow import _grade_saved_shadows

    plan, prices, fx, _, _ = cohort
    aid, rhash = _start_audit(tmp_path, {"prompt": "frozen test inputs"})
    raw = {"audit_id": aid, "audit_request_sha256": rhash, "shadow_plan": plan}
    _archive_decision(tmp_path, raw)
    data = tmp_path / "market_data"
    for symbol in ["AAA", "BBB", "SPY"]:
        path = data / "equity" / symbol
        path.mkdir(parents=True)
        source = prices["AAA"] if symbol == "SPY" else prices[symbol]
        source.to_frame("close").to_parquet(path / "1D.parquet")
    path = data / "fx" / "USDCHF"
    path.mkdir(parents=True)
    fx["USDCHF"].to_frame("close").to_parquet(path / "1D.parquet")
    first = _grade_saved_shadows(tmp_path, data, NOW)
    assert first["counts"] == {"graded": 1, "pending": 2, "unavailable": 0, "invalid_audit": 0}
    grade_path = _audit_dir(tmp_path, aid) / "shadow-5.json"
    original = grade_path.read_bytes()
    second = _grade_saved_shadows(tmp_path, data, NOW)
    assert second["counts"] == first["counts"]
    assert grade_path.read_bytes() == original
    assert b"price_observations" in original and b"fx_observations" in original

    # A valid envelope copied from another decision must still be rejected.
    from trading.agents.pm_audit import _read_record, _write_record

    saved, _ = _read_record(grade_path.parent, grade_path.name)
    saved["decision_sha256"] = "another-decision"
    grade_path.unlink()
    _write_record(grade_path.parent, grade_path.name, saved)
    assert _grade_saved_shadows(tmp_path, data, NOW)["counts"]["invalid_audit"] == 1


def test_todays_partial_candle_cannot_complete_a_grade(cohort):
    plan, prices, fx, _, _ = cohort
    during_last_bar = (prices.index[-1] + pd.Timedelta(hours=18)).to_pydatetime()
    assert _evaluate_shadow(plan, prices, fx, horizon=5, now=during_last_bar)["status"] == "pending"
