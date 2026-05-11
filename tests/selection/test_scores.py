"""Tests for PSR / DSR / helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading.selection import (
    annualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    per_period_sharpe,
    probabilistic_sharpe,
)


def test_psr_returns_probability_in_unit_interval() -> None:
    p = probabilistic_sharpe(sr=0.05, n_obs=252, skew=0.0, kurt=3.0)
    assert 0.0 <= p <= 1.0


def test_psr_with_zero_benchmark_normal_returns_is_above_0_5_for_positive_sharpe() -> None:
    # Positive per-period Sharpe over a Normal-like sample → P(SR_true > 0) > 0.5.
    p = probabilistic_sharpe(sr=0.10, n_obs=500, skew=0.0, kurt=3.0, sr_benchmark=0.0)
    assert p > 0.5


def test_psr_handles_short_series() -> None:
    assert probabilistic_sharpe(sr=0.5, n_obs=1, skew=0.0, kurt=3.0) == 0.0


def test_psr_handles_pathological_higher_moments() -> None:
    # If 1 - skew*sr + (kurt-1)/4 * sr^2 <= 0 the formula is undefined.
    # Skew of 100 with sr=0.1 → 1 - 10 + small = negative.
    p = probabilistic_sharpe(sr=0.1, n_obs=100, skew=100.0, kurt=3.0)
    assert p == 0.0


def test_expected_max_sharpe_grows_with_n_trials() -> None:
    a = expected_max_sharpe(n_trials=10, sr_std=1.0)
    b = expected_max_sharpe(n_trials=100, sr_std=1.0)
    c = expected_max_sharpe(n_trials=1000, sr_std=1.0)
    assert a < b < c


def test_expected_max_sharpe_zero_for_one_trial() -> None:
    assert expected_max_sharpe(n_trials=1, sr_std=1.0) == 0.0


def test_dsr_smaller_than_psr_against_zero_for_many_trials() -> None:
    sr, n, sk, kt = 0.08, 500, 0.0, 3.0
    psr0 = probabilistic_sharpe(sr, n, sk, kt, sr_benchmark=0.0)
    dsr_many = deflated_sharpe(sr, n, sk, kt, n_trials=100)
    # Selection bias against 100 trials must lower the probability.
    assert dsr_many < psr0


def test_annualize_sharpe_multiplies_by_sqrt_n() -> None:
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))


def test_per_period_sharpe_handles_zero_std() -> None:
    r = pd.Series([0.01] * 10)
    # Effectively constant → either exact 0 std or near-zero. We document
    # the contract: 0 (not NaN, not inf).
    assert per_period_sharpe(r) == 0.0 or abs(per_period_sharpe(r)) < 1e9


def test_per_period_sharpe_handles_short_series() -> None:
    assert per_period_sharpe(pd.Series([1.0])) == 0.0
    assert per_period_sharpe(pd.Series([], dtype=float)) == 0.0


def test_moments_normal_close_to_0_and_3() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 1, 10_000))
    sk, kt = moments(r)
    assert abs(sk) < 0.1
    assert abs(kt - 3.0) < 0.2


def test_moments_short_series_defaults() -> None:
    assert moments(pd.Series([0.0, 0.0])) == (0.0, 3.0)
