"""Vol-target overlay tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading.selection import vol_target


@pytest.fixture
def daily_idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")


def _gbm(seed: int, mu: float, sigma: float, n: int, idx: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(mu, sigma, n)
    return pd.Series(100 * np.exp(np.cumsum(log_ret)), index=idx)


def test_vol_target_scales_down_high_vol_portfolio(daily_idx: pd.DatetimeIndex) -> None:
    # Single asset with 4% daily vol → ~63% annualized. Target 10% → big haircut.
    p = _gbm(0, 0, 0.04, 400, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.ones(400)}, index=daily_idx)
    out = vol_target(w, p, target_vol=0.10, lookback=60, periods_per_year=252)
    # Average post-warm-up gross exposure should be << 1.0.
    avg = out["A"].iloc[100:].abs().mean()
    assert 0.1 < avg < 0.6


def test_vol_target_scales_up_low_vol_portfolio(daily_idx: pd.DatetimeIndex) -> None:
    p = _gbm(0, 0, 0.002, 400, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.ones(400)}, index=daily_idx)
    out = vol_target(w, p, target_vol=0.10, lookback=60, periods_per_year=252, max_leverage=10.0)
    # Annualized realized vol is ~3% — overlay should lever up.
    avg = out["A"].iloc[100:].abs().mean()
    assert avg > 1.5


def test_vol_target_respects_max_leverage(daily_idx: pd.DatetimeIndex) -> None:
    p = _gbm(0, 0, 0.001, 400, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.ones(400)}, index=daily_idx)
    out = vol_target(w, p, target_vol=1.0, lookback=60, periods_per_year=252, max_leverage=2.0)
    # All scale factors are clipped to 2.0.
    assert out["A"].iloc[100:].abs().max() == pytest.approx(2.0, rel=1e-9)


def test_vol_target_warmup_leaves_weights_unchanged(daily_idx: pd.DatetimeIndex) -> None:
    p = _gbm(0, 0, 0.01, 400, daily_idx).to_frame("A")
    w = pd.DataFrame({"A": np.full(400, 0.5)}, index=daily_idx)
    out = vol_target(w, p, target_vol=0.10, lookback=60, periods_per_year=252)
    # Inside the rolling-vol warm-up there is no scale → weights pass through unchanged.
    np.testing.assert_allclose(out["A"].iloc[:60].values, 0.5)


def test_vol_target_no_lookahead(daily_idx: pd.DatetimeIndex) -> None:
    """Spike the price on the last bar and confirm earlier output is
    *identical* to the run without the spike. If the overlay peeked
    ahead, the historical scale factors would shift."""
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    no_spike = pd.DataFrame({"A": base.copy()}, index=daily_idx)
    spiked = no_spike.copy()
    spiked.iloc[-1, 0] *= 5.0

    w = pd.DataFrame({"A": np.ones(400)}, index=daily_idx)
    out_clean = vol_target(
        w, no_spike, target_vol=0.10, lookback=20, periods_per_year=252, max_leverage=5.0
    )
    out_spiked = vol_target(
        w, spiked, target_vol=0.10, lookback=20, periods_per_year=252, max_leverage=5.0
    )
    # Every bar before the last must be unaffected by the future spike.
    pd.testing.assert_series_equal(out_clean["A"].iloc[:-1], out_spiked["A"].iloc[:-1])


def test_vol_target_rejects_bad_parameters(daily_idx: pd.DatetimeIndex) -> None:
    p = pd.DataFrame({"A": np.linspace(100.0, 200.0, 400)}, index=daily_idx)
    w = pd.DataFrame({"A": np.ones(400)}, index=daily_idx)
    with pytest.raises(ValueError, match="target_vol"):
        vol_target(w, p, target_vol=0.0, lookback=20, periods_per_year=252)
    with pytest.raises(ValueError, match="max_leverage"):
        vol_target(w, p, target_vol=0.1, lookback=20, periods_per_year=252, max_leverage=0)


def test_vol_target_annualization_consistent() -> None:
    """If realized daily vol is exactly 1% and lookback covers the whole
    series, the overlay's scale = target / (0.01 * sqrt(252))."""
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    # Construct prices so per-bar returns are alternating ±1% — deterministic vol.
    ret = np.tile([0.01, -0.01], 150)
    prices = pd.DataFrame(
        {"A": np.concatenate([[100.0], 100.0 * np.cumprod(1 + ret)[:-1]])}, index=idx
    )
    w = pd.DataFrame({"A": np.ones(300)}, index=idx)
    # Per-bar return alternates exactly ±0.01, so std with ddof=1 ≈ 0.01.
    out = vol_target(
        w, prices, target_vol=0.10, lookback=200, periods_per_year=252, max_leverage=10.0
    )
    expected_scale = 0.10 / (0.01 * math.sqrt(252))
    assert out["A"].iloc[-1] == pytest.approx(expected_scale, rel=5e-2)
