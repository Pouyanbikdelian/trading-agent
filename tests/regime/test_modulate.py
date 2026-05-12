"""Tests for the Strategy.modulate hook and the regime_scale helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.regime import regime_scale
from trading.strategies import Donchian


def _weights_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"A": [1.0, 1.0, 1.0, 1.0, 1.0], "B": [-0.5] * 5},
        index=idx,
    )


def test_regime_scale_applies_per_row() -> None:
    w = _weights_df()
    regime = pd.Series([0, 1, 2, 0, 2], index=w.index)
    out = regime_scale(w, regime, {0: 1.0, 1: 0.5, 2: 0.0})
    # Row 0,3 → full size; row 1 → half; rows 2,4 → zero.
    np.testing.assert_allclose(out["A"].values, [1.0, 0.5, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(out["B"].values, [-0.5, -0.25, 0.0, -0.5, 0.0])


def test_regime_scale_unknown_state_uses_default() -> None:
    w = _weights_df()
    regime = pd.Series([0, -1, 1, -1, 1], index=w.index)
    out = regime_scale(w, regime, {0: 1.0, 1: 1.0}, unknown_scale=0.25)
    # The -1 (warm-up) bars get 0.25.
    np.testing.assert_allclose(out["A"].values, [1.0, 0.25, 1.0, 0.25, 1.0])


def test_regime_scale_reindexes_when_misaligned() -> None:
    w = _weights_df()
    short_regime = pd.Series([0, 0], index=w.index[:2])  # missing the last 3 bars
    out = regime_scale(w, short_regime, {0: 1.0}, unknown_scale=0.0)
    np.testing.assert_allclose(out["A"].values, [1.0, 1.0, 0.0, 0.0, 0.0])


def test_default_modulate_is_identity() -> None:
    """A strategy without a regime_scale_map should leave weights unchanged."""
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100.0, 150.0, 300)}, index=idx)
    s = Donchian(lookback=20)
    w = s.generate(prices)
    regime = pd.Series(np.random.default_rng(0).integers(0, 3, len(idx)), index=idx)
    out = s.modulate(w, regime)
    pd.testing.assert_frame_equal(out, w)


def test_subclass_modulate_uses_scale_map() -> None:
    class _NoExposureInHighVol(Donchian):
        name = "_no_exposure_in_high_vol_test"  # unused — we don't @register
        regime_scale_map = {0: 1.0, 1: 1.0, 2: 0.0}

    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100.0, 150.0, 300)}, index=idx)
    s = _NoExposureInHighVol(lookback=20)
    w = s.generate(prices)
    regime = pd.Series([2] * len(idx), index=idx)  # always "high vol"
    out = s.modulate(w, regime)
    # Every bar should be zeroed.
    assert (out.values == 0.0).all()
