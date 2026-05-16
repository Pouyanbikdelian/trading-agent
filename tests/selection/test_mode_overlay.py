r"""Tests for the mode-based weight reshape overlay."""

from __future__ import annotations

import pandas as pd
import pytest

from trading.runtime.mode import Mode
from trading.selection.mode_overlay import (
    ModePolicy,
    apply_mode,
    estimate_mode_impact,
)


@pytest.fixture
def weights() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"AAPL": 0.5, "MSFT": 0.5},  # k=2 strategy at gross 1.0
        index=idx,
        dtype=float,
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"AAPL": 150, "MSFT": 350, "XLP": 70, "XLU": 60, "GLD": 180, "QUAL": 130},
        index=idx,
        dtype=float,
    )


def test_bull_is_pass_through(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    out = apply_mode(weights, prices, Mode.BULL)
    pd.testing.assert_frame_equal(out, weights)


def test_neutral_is_pass_through(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    out = apply_mode(weights, prices, Mode.NEUTRAL)
    pd.testing.assert_frame_equal(out, weights)


def test_flatten_zeros_everything(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    out = apply_mode(weights, prices, Mode.FLATTEN)
    assert (out.abs() < 1e-12).all().all()


def test_defense_scales_strategy_and_adds_defensive_sleeve(
    weights: pd.DataFrame, prices: pd.DataFrame
) -> None:
    out = apply_mode(weights, prices, Mode.DEFENSE)
    last = out.iloc[-1]
    # Strategy names scaled to 70%
    assert last["AAPL"] == pytest.approx(0.5 * 0.70)
    assert last["MSFT"] == pytest.approx(0.5 * 0.70)
    # 30% spread equally across 4 defensive ETFs = 7.5% each
    for tkr in ("XLP", "XLU", "GLD", "QUAL"):
        assert last[tkr] == pytest.approx(0.30 / 4)
    # Total gross ≈ 1.0
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-6)


def test_bear_holds_cash(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    out = apply_mode(weights, prices, Mode.BEAR)
    last = out.iloc[-1]
    # Strategy at 20%
    assert last["AAPL"] == pytest.approx(0.5 * 0.20)
    assert last["MSFT"] == pytest.approx(0.5 * 0.20)
    # Defensive at 30% total
    sleeve_total = sum(last[t] for t in ("XLP", "XLU", "GLD", "QUAL"))
    assert sleeve_total == pytest.approx(0.30, abs=1e-6)
    # The remaining 50% is cash — not on any weight column
    assert last.abs().sum() == pytest.approx(0.50, abs=1e-6)


def test_defense_skips_unavailable_defensive_etfs(weights: pd.DataFrame) -> None:
    # Drop GLD from prices — sleeve falls back to 3 names.
    idx = weights.index
    prices = pd.DataFrame(
        {"AAPL": 150, "MSFT": 350, "XLP": 70, "XLU": 60, "QUAL": 130},
        index=idx,
        dtype=float,
    )
    out = apply_mode(weights, prices, Mode.DEFENSE)
    last = out.iloc[-1]
    assert "GLD" not in out.columns
    # 30% spread across 3 names = 10% each
    for tkr in ("XLP", "XLU", "QUAL"):
        assert last[tkr] == pytest.approx(0.10)
    assert last.abs().sum() == pytest.approx(1.0, abs=1e-6)


def test_defense_with_no_defensive_etfs_falls_back_to_cash(weights: pd.DataFrame) -> None:
    # No defensive ETFs in the price frame at all → mode just scales gross down.
    idx = weights.index
    prices = pd.DataFrame({"AAPL": 150, "MSFT": 350}, index=idx, dtype=float)
    out = apply_mode(weights, prices, Mode.DEFENSE)
    last = out.iloc[-1]
    assert last.abs().sum() == pytest.approx(0.70, abs=1e-6)


def test_custom_policy_overrides_defaults(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    policy = ModePolicy(
        defense_strategy_gross=0.80,
        defense_defensive_gross=0.20,
    )
    out = apply_mode(weights, prices, Mode.DEFENSE, policy=policy)
    last = out.iloc[-1]
    assert last["AAPL"] == pytest.approx(0.5 * 0.80)
    assert sum(last[t] for t in ("XLP", "XLU", "GLD", "QUAL")) == pytest.approx(0.20, abs=1e-6)


def test_estimate_mode_impact_basic() -> None:
    cur = pd.Series({"AAPL": 0.50, "MSFT": 0.50}, dtype=float)
    tgt = pd.Series({"AAPL": 0.35, "MSFT": 0.35, "XLP": 0.30}, dtype=float)
    impact = estimate_mode_impact(cur, tgt, equity=100_000.0, cost_bps=10.0)
    # Two sells of $15k each, one buy of $30k → turnover 60% = $60k
    assert impact["turnover_dollar"] == pytest.approx(60_000.0, abs=1.0)
    assert impact["trading_cost_dollar"] == pytest.approx(60.0, abs=0.1)
    assert impact["n_changes"] == 3
    syms = {r["symbol"] for r in impact["sells"]}
    assert syms == {"AAPL", "MSFT"}
    assert impact["buys"][0]["symbol"] == "XLP"
