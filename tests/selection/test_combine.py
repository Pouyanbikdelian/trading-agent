"""Tests for the strategy combiners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.selection import equal_weight, inverse_vol, min_variance


@pytest.fixture
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=200, freq="1D", tz="UTC")


@pytest.fixture
def three_strategies(idx: pd.DatetimeIndex) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    rng = np.random.default_rng(0)
    weights = {
        "alpha": pd.DataFrame({"A": np.ones(200), "B": np.zeros(200)}, index=idx),
        "beta": pd.DataFrame({"A": np.full(200, 0.5), "B": np.full(200, 0.5)}, index=idx),
        "gamma": pd.DataFrame({"A": np.zeros(200), "B": np.ones(200)}, index=idx),
    }
    returns = {
        "alpha": pd.Series(rng.normal(0, 0.005, 200), index=idx),
        "beta": pd.Series(rng.normal(0, 0.010, 200), index=idx),
        "gamma": pd.Series(rng.normal(0, 0.020, 200), index=idx),
    }
    return weights, returns


# ----------------------------------------------------------- equal_weight ----


def test_equal_weight_averages_inputs(three_strategies) -> None:
    weights, _ = three_strategies
    out = equal_weight(weights)
    # alpha + beta + gamma = (1+0.5+0, 0+0.5+1) = (1.5, 1.5), /3 = (0.5, 0.5).
    np.testing.assert_allclose(out["A"].values, 0.5)
    np.testing.assert_allclose(out["B"].values, 0.5)


def test_equal_weight_rejects_empty() -> None:
    with pytest.raises(ValueError):
        equal_weight({})


def test_equal_weight_aligns_on_intersection() -> None:
    idx = pd.date_range("2022-01-01", periods=5, freq="1D", tz="UTC")
    a = pd.DataFrame({"X": [1.0] * 5, "Y": [0.0] * 5}, index=idx)
    b = pd.DataFrame({"Y": [1.0] * 5, "Z": [1.0] * 5}, index=idx)
    out = equal_weight({"a": a, "b": b})
    # Common column = Y only.
    assert list(out.columns) == ["Y"]


# ------------------------------------------------------------ inverse_vol ----


def test_inverse_vol_assigns_more_to_lowest_vol(three_strategies) -> None:
    weights, returns = three_strategies
    out = inverse_vol(weights, returns)
    # alpha has the lowest vol → its weights frame dominates the blend.
    # alpha weights are (1, 0) so blended frame's A > B.
    assert out["A"].iloc[-1] > out["B"].iloc[-1]


def test_inverse_vol_respects_lookback(three_strategies) -> None:
    weights, returns = three_strategies
    out_full = inverse_vol(weights, returns)
    out_short = inverse_vol(weights, returns, lookback=20)
    # Different lookbacks → in general, different blends.
    assert not np.allclose(out_full.values, out_short.values)


def test_inverse_vol_rejects_all_zero_vol() -> None:
    idx = pd.date_range("2022-01-01", periods=10, freq="1D", tz="UTC")
    weights = {
        "a": pd.DataFrame({"X": [1.0] * 10}, index=idx),
        "b": pd.DataFrame({"X": [1.0] * 10}, index=idx),
    }
    returns = {"a": pd.Series([0.0] * 10, index=idx), "b": pd.Series([0.0] * 10, index=idx)}
    with pytest.raises(ValueError, match="zero realized vol"):
        inverse_vol(weights, returns)


# ------------------------------------------------------------ min_variance ----


def test_min_variance_loads_lowest_vol_strategy(three_strategies) -> None:
    weights, returns = three_strategies
    out = min_variance(weights, returns)
    # Same intuition as inverse_vol: alpha (lowest vol) dominates.
    assert out["A"].iloc[-1] > out["B"].iloc[-1]


def test_min_variance_outputs_sum_to_blended_weights() -> None:
    idx = pd.date_range("2022-01-01", periods=100, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    weights = {
        "a": pd.DataFrame({"X": np.ones(100)}, index=idx),
        "b": pd.DataFrame({"X": np.ones(100)}, index=idx),
    }
    returns = {
        "a": pd.Series(rng.normal(0, 0.01, 100), index=idx),
        "b": pd.Series(rng.normal(0, 0.01, 100), index=idx),
    }
    out = min_variance(weights, returns)
    # Both strategies have identical weight frames (all ones) and i.i.d. but
    # different sample vols, so the combiner's strategy-level scalars sum to 1
    # → the X column should also be ~1.0.
    assert out["X"].iloc[-1] == pytest.approx(1.0, rel=1e-9)


def test_min_variance_short_series_rejected() -> None:
    idx = pd.date_range("2022-01-01", periods=1, freq="1D", tz="UTC")
    weights = {
        "a": pd.DataFrame({"X": [1.0]}, index=idx),
        "b": pd.DataFrame({"X": [1.0]}, index=idx),
    }
    returns = {"a": pd.Series([0.01], index=idx), "b": pd.Series([0.02], index=idx)}
    with pytest.raises(ValueError, match="at least 2"):
        min_variance(weights, returns)
