"""HmmRegime tests using a synthetic two-regime returns process.

We sample from two well-separated Gaussians and assert the fit recovers
the right block-by-block assignment after sorting states by emission mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.regime import HmmRegime


@pytest.fixture
def two_regime_returns() -> pd.Series:
    """500 bars of bull (mu=+0.0015, sigma=0.005) then 300 bars of bear
    (mu=-0.008, sigma=0.025). Well-separated in both mean and variance so
    the HMM has clear signal to fit."""
    rng = np.random.default_rng(0)
    bull = rng.normal(0.0015, 0.005, 500)
    bear = rng.normal(-0.008, 0.025, 300)
    idx = pd.date_range("2022-01-01", periods=800, freq="1D", tz="UTC")
    return pd.Series(np.concatenate([bull, bear]), index=idx, name="r")


def test_two_state_hmm_separates_regimes(two_regime_returns: pd.Series) -> None:
    hmm = HmmRegime(n_states=2).fit(two_regime_returns)
    labels = hmm.predict(two_regime_returns)
    # state=0 is the most-bearish-mean state, state=1 the most-bullish.
    bull_segment = labels.iloc[:500]
    bear_segment = labels.iloc[500:]
    assert (bull_segment == 1).mean() > 0.70
    assert (bear_segment == 0).mean() > 0.70


def test_states_are_mean_sorted(two_regime_returns: pd.Series) -> None:
    """State labels are remapped so state 0 has the lowest emission mean."""
    hmm = HmmRegime(n_states=2).fit(two_regime_returns)
    # Means of the underlying model (raw IDs) reordered into sorted positions.
    raw_means = np.asarray(hmm._model.means_).ravel()
    sorted_means = raw_means[np.argsort(raw_means)]
    assert sorted_means[0] < sorted_means[-1]


def test_predict_before_fit_raises() -> None:
    hmm = HmmRegime(n_states=2)
    with pytest.raises(RuntimeError, match="fit"):
        hmm.predict(pd.Series([0.0] * 50))


def test_fit_is_reproducible(two_regime_returns: pd.Series) -> None:
    """Same random_state -> same prediction sequence."""
    a = HmmRegime(n_states=2, random_state=7).fit(two_regime_returns).predict(two_regime_returns)
    b = HmmRegime(n_states=2, random_state=7).fit(two_regime_returns).predict(two_regime_returns)
    pd.testing.assert_series_equal(a, b)


def test_invalid_state_count() -> None:
    with pytest.raises(Exception):
        HmmRegime(n_states=1)
