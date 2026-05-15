r"""Tests for the proportional beta-hedge overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.selection import beta_hedge


@pytest.fixture
def daily_idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")


def _make_market_and_high_beta_name(idx: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(idx)
    mkt_ret = rng.normal(0.0004, 0.01, n)
    # High-beta name: 1.5x market + small idiosyncratic noise
    name_ret = 1.5 * mkt_ret + rng.normal(0.0, 0.005, n)
    spy = 100.0 * np.exp(np.cumsum(mkt_ret))
    aaa = 100.0 * np.exp(np.cumsum(name_ret))
    return pd.DataFrame({"SPY": spy, "AAA": aaa}, index=idx)


def test_beta_hedge_adds_negative_spy_weight(daily_idx: pd.DatetimeIndex) -> None:
    prices = _make_market_and_high_beta_name(daily_idx)
    weights = pd.DataFrame({"AAA": np.ones(len(daily_idx))}, index=daily_idx)
    out = beta_hedge(
        weights, prices, benchmark="SPY", beta_lookback=63, base_intensity=1.0, max_hedge=2.0
    )
    # After warm-up, SPY weight should be negative on average.
    assert out["SPY"].iloc[200:].mean() < -0.1


def test_beta_hedge_respects_cap(daily_idx: pd.DatetimeIndex) -> None:
    prices = _make_market_and_high_beta_name(daily_idx)
    weights = pd.DataFrame({"AAA": np.ones(len(daily_idx))}, index=daily_idx)
    out = beta_hedge(
        weights, prices, benchmark="SPY", beta_lookback=63, base_intensity=1.0, max_hedge=0.3
    )
    assert out["SPY"].min() >= -0.3 - 1e-9


def test_beta_hedge_noop_when_benchmark_missing(daily_idx: pd.DatetimeIndex) -> None:
    prices = pd.DataFrame(
        {
            "AAA": 100.0
            * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, len(daily_idx))))
        },
        index=daily_idx,
    )
    weights = pd.DataFrame({"AAA": np.ones(len(daily_idx))}, index=daily_idx)
    out = beta_hedge(weights, prices, benchmark="SPY")
    # No SPY column means no hedge — output identical to input.
    assert out.equals(weights)


def test_beta_hedge_only_shorts_never_longs(daily_idx: pd.DatetimeIndex) -> None:
    # Portfolio with negative beta (short the high-beta name).
    prices = _make_market_and_high_beta_name(daily_idx)
    weights = pd.DataFrame({"AAA": -np.ones(len(daily_idx))}, index=daily_idx)
    out = beta_hedge(
        weights, prices, benchmark="SPY", beta_lookback=63, base_intensity=1.0, max_hedge=2.0
    )
    # The hedge weight must never be positive — overlay only shorts the market.
    assert (out["SPY"] <= 0.0 + 1e-9).all()


def test_beta_hedge_vix_signal_increases_hedge_when_high(daily_idx: pd.DatetimeIndex) -> None:
    prices = _make_market_and_high_beta_name(daily_idx)
    weights = pd.DataFrame({"AAA": np.ones(len(daily_idx))}, index=daily_idx)
    # Construct a VIX series that ramps up over time.
    vix = pd.Series(np.linspace(12.0, 40.0, len(daily_idx)), index=daily_idx)
    out = beta_hedge(
        weights,
        prices,
        benchmark="SPY",
        beta_lookback=63,
        vix_series=vix,
        vix_floor=0.0,
        vix_scale=1.0,
        max_hedge=2.0,
    )
    early = out["SPY"].iloc[100:150].mean()
    late = out["SPY"].iloc[-50:].mean()
    # Late should be more negative than early (more hedge as VIX rose).
    assert late < early
