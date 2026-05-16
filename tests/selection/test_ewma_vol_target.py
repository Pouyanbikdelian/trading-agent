r"""Tests for the EWMA volatility-target overlay.

EWMA is the recursive vol estimator from RiskMetrics — same call
signature as the rolling-std vol_target but more responsive to recent
regime changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.selection import ewma_vol_target, vol_target


@pytest.fixture
def daily_idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")


def _gbm(seed: int, mu: float, sigma: float, idx: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(mu, sigma, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(log_ret)), index=idx)


def test_ewma_scales_down_high_vol(daily_idx: pd.DatetimeIndex) -> None:
    # 4% daily vol → ~63% annualised. Target 10% → big haircut expected.
    p = _gbm(0, 0, 0.04, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.ones(len(daily_idx))}, index=daily_idx)
    out = ewma_vol_target(w, p, target_vol=0.10, lookback=60, periods_per_year=252)
    avg = out["A"].iloc[100:].abs().mean()
    assert 0.1 < avg < 0.5


def test_ewma_never_levers_up_by_default(daily_idx: pd.DatetimeIndex) -> None:
    # Calm regime — without an explicit max_leverage>1 the overlay must
    # NEVER add gross. Default max_leverage=1.0 enforces this.
    p = _gbm(0, 0, 0.002, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.ones(len(daily_idx))}, index=daily_idx)
    out = ewma_vol_target(w, p, target_vol=0.50, lookback=60, periods_per_year=252)
    assert out["A"].abs().max() <= 1.0001


def test_ewma_reacts_faster_than_rolling_std() -> None:
    r"""Construct a calm regime followed by a sudden vol regime change.
    EWMA's response to the shock should peak earlier than rolling-std.
    """
    idx = pd.date_range("2022-01-01", periods=600, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    # Phase 1: calm 1.5% daily vol for 300 bars
    calm = rng.normal(0, 0.015, 300)
    # Phase 2: hot 5% daily vol for 300 bars
    hot = rng.normal(0, 0.05, 300)
    returns = np.concatenate([calm, hot])
    p = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(returns))}, index=idx)
    w = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)

    ewma_out = ewma_vol_target(w, p, target_vol=0.10, lookback=60, periods_per_year=252)
    std_out = vol_target(w, p, target_vol=0.10, lookback=60, periods_per_year=252, max_leverage=1.0)

    # Inspect the first 60 bars of the high-vol phase. EWMA should
    # have downsized substantially; rolling-std lags.
    transition_zone = slice(305, 360)
    ewma_mean = ewma_out["A"].iloc[transition_zone].mean()
    std_mean = std_out["A"].iloc[transition_zone].mean()
    assert ewma_mean < std_mean


def test_ewma_rejects_bad_params(daily_idx: pd.DatetimeIndex) -> None:
    p = _gbm(0, 0, 0.01, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.ones(len(daily_idx))}, index=daily_idx)

    with pytest.raises(ValueError):
        ewma_vol_target(w, p, target_vol=-0.1, lookback=60, periods_per_year=252)
    with pytest.raises(ValueError):
        ewma_vol_target(w, p, target_vol=0.10, lookback=60, lam=1.5, periods_per_year=252)
    with pytest.raises(ValueError):
        ewma_vol_target(w, p, target_vol=0.10, lookback=60, lam=0.0, periods_per_year=252)


def test_ewma_no_lookahead() -> None:
    r"""Today's scale factor uses information available up to yesterday.

    Verify: if I replace tomorrow's return with anything, today's scale
    factor doesn't change."""
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    p_base = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 300)))}, index=idx)
    w = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)

    out_base = ewma_vol_target(w, p_base, target_vol=0.10, lookback=60, periods_per_year=252)

    # Tamper with the last bar — today's row should be unaffected because
    # today uses yesterday's forecast.
    p_mod = p_base.copy()
    p_mod.iloc[-1] = p_mod.iloc[-1] * 5.0
    out_mod = ewma_vol_target(w, p_mod, target_vol=0.10, lookback=60, periods_per_year=252)
    assert out_base["A"].iloc[-2] == pytest.approx(out_mod["A"].iloc[-2])
