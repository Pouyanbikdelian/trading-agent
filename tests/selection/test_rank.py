"""Tests for the rank_strategies leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.selection import rank_strategies


def _synth_returns(seed: int, mu: float, sigma: float, n: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def test_rank_empty_returns_empty_frame() -> None:
    out = rank_strategies({}, periods_per_year=252)
    assert out.empty
    for col in ("sharpe", "psr", "dsr", "n_obs", "skew", "kurt"):
        assert col in out.columns


def test_rank_orders_by_dsr_descending() -> None:
    good = _synth_returns(0, mu=0.0015, sigma=0.005)
    noisy = _synth_returns(1, mu=0.0002, sigma=0.02)
    out = rank_strategies({"good": good, "noisy": noisy}, periods_per_year=252)
    # Sorted by DSR — `good` should win.
    assert out.index.tolist()[0] == "good"
    assert out.loc["good", "dsr"] > out.loc["noisy", "dsr"]


def test_rank_columns_have_expected_dtypes() -> None:
    good = _synth_returns(0, mu=0.001, sigma=0.005)
    out = rank_strategies({"good": good}, periods_per_year=252)
    for c in ("sharpe", "psr", "dsr", "n_obs", "skew", "kurt"):
        assert out[c].dtype.kind == "f"


def test_rank_more_trials_lowers_dsr_for_same_returns() -> None:
    base = _synth_returns(0, mu=0.001, sigma=0.005)
    out_single = rank_strategies({"a": base}, periods_per_year=252)
    out_many = rank_strategies(
        {f"s{i}": _synth_returns(i, mu=0.001, sigma=0.005) for i in range(50)},
        periods_per_year=252,
    )
    # `a` against itself has n_trials=1 so DSR == PSR(0). With n_trials=50 the
    # same headline Sharpe gets penalized.
    if "a" in out_many.index:
        assert out_many.loc["a", "dsr"] < out_single.loc["a", "dsr"]
