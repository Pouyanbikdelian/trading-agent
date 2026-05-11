"""RealizedVolRegime tests — concatenate distinct-vol blocks and assert
that the classifier separates them correctly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.regime import RealizedVolRegime


def _three_vol_blocks(seed: int = 0) -> pd.Series:
    """Returns made of three blocks: low-vol, mid-vol, high-vol — each 200 bars."""
    rng = np.random.default_rng(seed)
    low = rng.normal(0, 0.003, 200)
    mid = rng.normal(0, 0.010, 200)
    high = rng.normal(0, 0.030, 200)
    idx = pd.date_range("2022-01-01", periods=600, freq="1D", tz="UTC")
    return pd.Series(np.concatenate([low, mid, high]), index=idx, name="r")


def test_separates_three_vol_buckets() -> None:
    rets = _three_vol_blocks()
    rv = RealizedVolRegime(window=20, n_states=3).fit(rets)
    labels = rv.predict(rets)
    # Each block should be dominated by its corresponding bucket. We allow
    # the warm-up at boundaries to leak — check the *middle* of each block.
    low_block = labels.iloc[50:180]
    mid_block = labels.iloc[250:380]
    high_block = labels.iloc[450:580]
    assert (low_block == 0).mean() > 0.85
    assert (mid_block == 1).mean() > 0.85
    assert (high_block == 2).mean() > 0.85


def test_warmup_returns_minus_one() -> None:
    rets = _three_vol_blocks()
    rv = RealizedVolRegime(window=20, n_states=3).fit(rets)
    labels = rv.predict(rets)
    # First window-1 bars can't have a rolling std → label -1.
    assert (labels.iloc[:18] == -1).all()


def test_predict_before_fit_raises() -> None:
    rv = RealizedVolRegime(window=20, n_states=3)
    with pytest.raises(RuntimeError, match="fit"):
        rv.predict(pd.Series([0.01] * 30))


def test_fit_rejects_too_few_observations() -> None:
    short = pd.Series([0.0] * 25, name="r")
    rv = RealizedVolRegime(window=20, n_states=3)
    with pytest.raises(ValueError, match="at least"):
        rv.fit(short)


def test_two_state_classifier() -> None:
    rets = _three_vol_blocks()
    rv = RealizedVolRegime(window=20, n_states=2).fit(rets)
    labels = rv.predict(rets)
    # Two-bucket classifier: low/mid in 0, high in 1 (roughly).
    high_block = labels.iloc[450:580]
    assert (high_block == 1).mean() > 0.85


def test_n_states_clamped_in_params() -> None:
    with pytest.raises(Exception):
        RealizedVolRegime(n_states=1)
    with pytest.raises(Exception):
        RealizedVolRegime(n_states=99)
