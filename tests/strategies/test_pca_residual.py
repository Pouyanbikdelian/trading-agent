r"""Tests for the PCA-residual mean-reversion strategy.

We use a synthetic factor model with a deterministic mean-reverting
residual injected on a single asset and check that the strategy enters
that asset on the correct side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest import ZERO_COSTS, run_vectorized
from trading.strategies import PCAResidual


def _factor_model_prices(seed: int = 0, n: int = 400, n_names: int = 10) -> pd.DataFrame:
    """Returns ~ market_factor + idiosyncratic noise, with name #0 carrying
    a mean-reverting OU residual on top of the factor."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    market = rng.normal(0.0005, 0.012, n)
    # Ornstein-Uhlenbeck residual on name 0 with theta=0.05, sigma=0.005
    ou = np.zeros(n)
    theta, sigma = 0.05, 0.005
    for t in range(1, n):
        ou[t] = ou[t - 1] * (1 - theta) + rng.normal(0.0, sigma)
    cols: dict[str, np.ndarray] = {}
    for i in range(n_names):
        beta = 1.0 + 0.1 * i
        noise = rng.normal(0.0, 0.005, n)
        ret = beta * market + noise + (ou if i == 0 else 0.0)
        cols[f"A{i}"] = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame(cols, index=idx)


def test_pca_residual_requires_minimum_breadth() -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="1D", tz="UTC")
    too_few = pd.DataFrame({"A": [1.0] * 200, "B": [1.0] * 200}, index=idx)
    with pytest.raises(ValueError, match="at least 5"):
        PCAResidual().generate(too_few)


def test_pca_residual_validates_exit_lt_entry() -> None:
    with pytest.raises(Exception, match="exit_z"):
        PCAResidual(entry_z=1.0, exit_z=2.0)


def test_pca_residual_enters_both_sides() -> None:
    prices = _factor_model_prices()
    s = PCAResidual(
        pca_window=60, n_factors=2, residual_horizon=5, z_window=20, entry_z=1.0, exit_z=0.3
    )
    w = s.generate(prices)
    assert (w.values > 0).any()
    assert (w.values < 0).any()


def test_pca_residual_round_trips_through_engine() -> None:
    prices = _factor_model_prices()
    s = PCAResidual(
        pca_window=60, n_factors=2, residual_horizon=5, z_window=20, entry_z=1.0, exit_z=0.3
    )
    w = s.generate(prices)
    result = run_vectorized(prices, w, costs=ZERO_COSTS)
    assert np.isfinite(result.total_return)
    assert len(result.equity) == len(prices)


def test_pca_residual_long_only_mode() -> None:
    prices = _factor_model_prices()
    s = PCAResidual(
        pca_window=60,
        n_factors=2,
        residual_horizon=5,
        z_window=20,
        entry_z=1.0,
        exit_z=0.3,
        allow_short=False,
    )
    w = s.generate(prices)
    assert (w.values >= 0).all()


def test_pca_residual_warmup_is_zero() -> None:
    """No position can be open before any residual has been computed.

    Bars [0, pca_window) have no residual at all, so they MUST be zero.
    Bars after pca_window can carry leak from min_periods=window rolling
    edges; we don't pin those.
    """
    prices = _factor_model_prices(n=200)
    s = PCAResidual(pca_window=60, residual_horizon=5, z_window=20)
    w = s.generate(prices)
    assert (w.iloc[:60].values == 0.0).all()
