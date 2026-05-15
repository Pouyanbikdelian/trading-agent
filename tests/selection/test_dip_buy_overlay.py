r"""Tests for the buy-the-dip overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.selection import dip_buy


def test_dip_buy_boosts_on_pullback() -> None:
    # Construct a price that rises, then dips ~6% from peak. The dip
    # should trigger a boost since the long SMA is still below price.
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    p_series = np.concatenate(
        [
            np.linspace(100.0, 200.0, 250),  # long uptrend
            np.linspace(200.0, 188.0, 50),  # ~6% drawdown from peak 200
        ]
    )
    prices = pd.DataFrame({"AAA": p_series}, index=idx)
    weights = pd.DataFrame({"AAA": 0.10 * np.ones(len(idx))}, index=idx)
    out = dip_buy(
        weights,
        prices,
        trigger=0.05,
        boost=0.20,
        max_per_position=0.20,
        trend_filter=0,  # disabled — trend-filter behaviour is tested separately
    )
    # At least one bar during the drawdown should be boosted above base.
    assert (out["AAA"] > 0.10 + 1e-9).any()


def test_dip_buy_respects_max_per_position() -> None:
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    p_series = np.concatenate([np.linspace(100.0, 200.0, 250), np.linspace(200.0, 160.0, 50)])
    prices = pd.DataFrame({"AAA": p_series}, index=idx)
    weights = pd.DataFrame({"AAA": 0.18 * np.ones(len(idx))}, index=idx)
    out = dip_buy(
        weights,
        prices,
        trigger=0.05,
        boost=1.0,
        max_per_position=0.20,
        trend_filter=0,
    )
    assert out["AAA"].max() <= 0.20 + 1e-9


def test_dip_buy_does_not_trigger_on_unheld_names() -> None:
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    p_series = np.concatenate([np.linspace(100.0, 200.0, 250), np.linspace(200.0, 180.0, 50)])
    prices = pd.DataFrame({"AAA": p_series}, index=idx)
    # Never held — overlay should not magically open a position.
    weights = pd.DataFrame({"AAA": np.zeros(len(idx))}, index=idx)
    out = dip_buy(weights, prices, trigger=0.05, boost=0.50, max_per_position=0.20)
    assert (out["AAA"].abs() < 1e-12).all()


def test_dip_buy_skips_when_below_long_trend() -> None:
    # Falling price the entire time — trend filter should block boosts.
    idx = pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")
    p_series = np.linspace(200.0, 100.0, 400)
    prices = pd.DataFrame({"AAA": p_series}, index=idx)
    weights = pd.DataFrame({"AAA": 0.10 * np.ones(len(idx))}, index=idx)
    out = dip_buy(weights, prices, trigger=0.05, boost=0.50, trend_filter=100)
    # With trend filter active and price perpetually below its SMA, no boost.
    assert np.allclose(out["AAA"].iloc[200:].to_numpy(), 0.10)


def test_dip_buy_resets_peak_after_firing() -> None:
    # After a boost, the peak resets. The overlay should not double-fire
    # on the same drawdown.
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    # Long uptrend, then a flat plateau at the dip level.
    p_series = np.concatenate(
        [
            np.linspace(100.0, 200.0, 250),
            188.0 * np.ones(50),  # 6% below peak, stays there
        ]
    )
    prices = pd.DataFrame({"AAA": p_series}, index=idx)
    weights = pd.DataFrame({"AAA": 0.10 * np.ones(len(idx))}, index=idx)
    out = dip_buy(
        weights,
        prices,
        trigger=0.05,
        boost=0.20,
        max_per_position=0.20,
        trend_filter=0,
    )
    # The boost fires once, then the peak resets to ~188 and the
    # drawdown becomes 0 — subsequent bars should drop back to the base
    # weight (no compound boosting).
    boosted_bars = (out["AAA"] > 0.10 + 1e-9).sum()
    assert boosted_bars >= 1
    assert boosted_bars < 50  # didn't fire on every plateau bar
