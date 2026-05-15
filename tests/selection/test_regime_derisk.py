r"""Tests for the regime_derisk overlay (cash-conversion in broken markets)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.selection import regime_derisk


@pytest.fixture
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=600, freq="1D", tz="UTC")


def test_regime_derisk_passes_through_in_uptrend(idx: pd.DatetimeIndex) -> None:
    # Pure uptrend — SPY always above its SMA(200). Overlay should be a no-op.
    spy = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx, name="SPY")
    weights = pd.DataFrame({"AAA": np.ones(len(idx)) * 0.5}, index=idx)
    out = regime_derisk(
        weights, pd.DataFrame({"SPY": spy}, index=idx), benchmark="SPY", trend_window=200
    )
    # After warm-up, weights should equal the input (no scaling).
    assert np.allclose(out["AAA"].iloc[250:].to_numpy(), 0.5)


def test_regime_derisk_cuts_exposure_in_sustained_downtrend(idx: pd.DatetimeIndex) -> None:
    # SPY runs up for 300 bars, then collapses below its SMA(200) and stays.
    up = np.linspace(100.0, 300.0, 300)
    down = np.linspace(300.0, 150.0, 300)
    spy = pd.Series(np.concatenate([up, down]), index=idx, name="SPY")
    weights = pd.DataFrame({"AAA": np.ones(len(idx))}, index=idx)
    out = regime_derisk(
        weights,
        pd.DataFrame({"SPY": spy}, index=idx),
        benchmark="SPY",
        trend_window=200,
        confirm_days=5,
        derisk_scale=0.3,
        deep_derisk_scale=0.1,
    )
    # Final bars should be in the deep-derisk regime (price way below SMA
    # AND death-cross fired).
    tail = out["AAA"].iloc[-50:]
    assert (tail <= 0.31).all()  # at least the derisk scale
    assert tail.iloc[-1] <= 0.11  # near the deep derisk scale


def test_regime_derisk_ignores_brief_dip(idx: pd.DatetimeIndex) -> None:
    # SPY in clear uptrend with a single 3-day dip below its SMA. The
    # 5-day confirmation rule should prevent the overlay from firing.
    spy_vals = np.linspace(100.0, 200.0, len(idx))
    # Plant a 3-day artificial collapse around bar 400 (below the SMA).
    spy_vals[400:403] = 100.0
    spy = pd.Series(spy_vals, index=idx, name="SPY")
    weights = pd.DataFrame({"AAA": np.ones(len(idx))}, index=idx)
    out = regime_derisk(
        weights,
        pd.DataFrame({"SPY": spy}, index=idx),
        benchmark="SPY",
        trend_window=200,
        confirm_days=5,
    )
    # The dip is only 3 bars, less than confirm_days=5 → never persistent.
    assert (out["AAA"].iloc[210:].to_numpy() > 0.99).all()


def test_regime_derisk_noop_when_benchmark_missing(idx: pd.DatetimeIndex) -> None:
    # No SPY in price frame and no explicit benchmark_prices → no-op.
    prices = pd.DataFrame(
        {"AAA": 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, len(idx))))},
        index=idx,
    )
    weights = pd.DataFrame({"AAA": np.ones(len(idx))}, index=idx)
    out = regime_derisk(weights, prices, benchmark="SPY")
    assert out.equals(weights)


def test_regime_derisk_rejects_inverted_scales(idx: pd.DatetimeIndex) -> None:
    spy = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    weights = pd.DataFrame({"AAA": np.ones(len(idx))}, index=idx)
    with pytest.raises(ValueError):
        regime_derisk(
            weights,
            pd.DataFrame({"SPY": spy}, index=idx),
            derisk_scale=0.1,
            deep_derisk_scale=0.3,  # inverted on purpose
        )
