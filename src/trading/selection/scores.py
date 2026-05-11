"""Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

Both are due to Bailey & López de Prado (2014, "The Sharpe Ratio Efficient
Frontier"). They answer the question that a raw Sharpe number can't:

    Given how much data we have, the non-normality of the returns, and
    how many strategies we tested before picking this one, what is the
    *probability* that the true Sharpe exceeds some benchmark?

PSR
---
PSR(SR_b) = Φ( (SR - SR_b) * sqrt(T - 1) / sqrt(1 - γ3 * SR + (γ4 - 1)/4 * SR^2) )

where SR is the *observed* (in-sample) Sharpe, SR_b is the benchmark Sharpe
we want to beat, T is the number of return observations, γ3 is skewness and
γ4 is the kurtosis (not excess — see the original paper).

DSR
---
DSR = PSR(SR_0) where SR_0 is the expected maximum Sharpe of N independent
zero-Sharpe trials, derived from extreme value theory:

    SR_0 ≈ sqrt(V) * ( (1 - γ_e) Φ^-1(1 - 1/N) + γ_e Φ^-1(1 - 1/(N*e)) )

with V the variance of the trial Sharpes (we use the sample variance if
``sr_std`` is given, else fall back to 1/sqrt(T) which assumes Gaussian
returns under the null) and γ_e ≈ 0.5772 (Euler-Mascheroni).

References
----------
* Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012)
* López de Prado, "Backtesting" (in *Advances in Financial ML*, 2018)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    """Probability that the *true* Sharpe exceeds ``sr_benchmark``.

    Inputs are per-period: pass the Sharpe and moments computed on the same
    frequency as ``n_obs``. ``kurt`` is full (Pearson) kurtosis, not excess —
    a Normal has kurt = 3.
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        # Pathological higher-moments — PSR is undefined; punt to 0.
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum of ``n_trials`` independent draws from a zero-mean
    Sharpe distribution with standard deviation ``sr_std``."""
    if n_trials < 2:
        return 0.0
    inv_phi_a = norm.ppf(1.0 - 1.0 / n_trials)
    inv_phi_b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_std * ((1.0 - _EULER_GAMMA) * inv_phi_a + _EULER_GAMMA * inv_phi_b))


def deflated_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    sr_std: float | None = None,
) -> float:
    """Deflated Sharpe Ratio — PSR vs. the expected best-of-N benchmark.

    If ``sr_std`` is None we approximate it as ``1 / sqrt(n_obs)`` (the
    Gaussian-null variance of an estimated Sharpe). Pass the empirical
    sample std of the ``n_trials`` Sharpes if you have it — strictly better.
    """
    if sr_std is None:
        sr_std = 1.0 / math.sqrt(max(n_obs, 1))
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe(sr, n_obs, skew, kurt, sr_benchmark=sr0)


def annualize_sharpe(sr_per_period: float, periods_per_year: int) -> float:
    """Convert a per-bar Sharpe to an annual figure (multiplies by sqrt(N))."""
    return float(sr_per_period * math.sqrt(periods_per_year))


_STD_EPS = 1e-12
"""Below this threshold the sample std is treated as zero. Float roundoff
on a constant series gives ~1e-19 rather than exactly 0; without a
tolerance we'd return a nonsense Sharpe of ~1e+16."""


def per_period_sharpe(returns: pd.Series) -> float:
    """Sharpe on the *return frequency* of the series (no annualization).

    Returns 0 for degenerate inputs (too few points, constant returns).
    """
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std < _STD_EPS or np.isnan(std):
        return 0.0
    return float(returns.mean() / std)


def moments(returns: pd.Series) -> tuple[float, float]:
    """(skew, kurt) — Pearson kurtosis (not excess). Returns (0, 3) for
    series too short to estimate moments meaningfully."""
    if len(returns) < 4:
        return 0.0, 3.0
    r = returns.dropna()
    # pandas skew/kurt are bias-adjusted and excess respectively; convert.
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0
    return skew, kurt
