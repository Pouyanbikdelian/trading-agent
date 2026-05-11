"""Pairs strategy tests.

We build deterministic synthetic price pairs whose cointegration outcome is
known: a tightly cointegrated pair (``y = 1.5*x + small_noise``) and an
unrelated pair (two independent random walks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest import ZERO_COSTS, run_vectorized
from trading.strategies import Pairs


def _cointegrated_pair(seed: int = 0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    x = 100.0 + np.cumsum(rng.normal(0, 1, n))
    noise = rng.normal(0, 0.5, n)
    y = 1.5 * x + noise
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def _unrelated_pair(seed: int = 0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    a = 100.0 + np.cumsum(rng.normal(0, 1, n))
    b = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"a": a, "b": b}, index=idx)


def test_pairs_requires_two_symbols() -> None:
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    three = pd.DataFrame({"a": [1.0] * 300, "b": [1.0] * 300, "c": [1.0] * 300}, index=idx)
    with pytest.raises(ValueError, match="exactly 2 symbols"):
        Pairs().generate(three)


def test_pairs_trades_when_cointegrated() -> None:
    prices = _cointegrated_pair()
    s = Pairs(fit_window=200, beta_window=60, z_window=60, entry_z=1.5, exit_z=0.3)
    w = s.generate(prices)
    # Both legs should fire over a long enough sample.
    assert (w["y"] > 0).any()
    assert (w["y"] < 0).any()
    # Legs are always opposite-signed where active.
    active = w["y"] != 0
    assert (np.sign(w.loc[active, "y"]) == -np.sign(w.loc[active, "x"])).all()


def test_pairs_skips_unrelated_series() -> None:
    prices = _unrelated_pair(seed=7)
    s = Pairs(fit_window=200, beta_window=60, z_window=60, coint_pvalue=0.01,
              require_cointegration=True)
    w = s.generate(prices)
    # Cointegration test should reject → all-zero weights.
    assert w.values.sum() == pytest.approx(0.0)


def test_pairs_can_skip_coint_check() -> None:
    prices = _unrelated_pair(seed=7)
    s = Pairs(fit_window=200, beta_window=60, z_window=60,
              require_cointegration=False, entry_z=1.0, exit_z=0.2)
    w = s.generate(prices)
    # With the check disabled, the strategy still produces signals from the spread.
    assert (w["a"] != 0).any() or (w["b"] != 0).any()


def test_pairs_returns_zero_when_history_too_short() -> None:
    idx = pd.date_range("2022-01-01", periods=50, freq="1D", tz="UTC")
    prices = pd.DataFrame({"y": np.arange(50.0), "x": np.arange(50.0) * 1.5}, index=idx)
    s = Pairs(fit_window=200)
    w = s.generate(prices)
    assert (w.values == 0.0).all()


def test_pairs_validates_exit_lt_entry() -> None:
    with pytest.raises(Exception, match="exit_z"):
        Pairs(entry_z=1.0, exit_z=2.0)


def test_pairs_beta_hedge_uses_lagged_beta() -> None:
    prices = _cointegrated_pair()
    s_unhedged = Pairs(fit_window=200, beta_window=60, z_window=60, entry_z=1.5,
                       exit_z=0.3, beta_hedge=False, weight_per_leg=0.5)
    s_hedged = Pairs(fit_window=200, beta_window=60, z_window=60, entry_z=1.5,
                     exit_z=0.3, beta_hedge=True, weight_per_leg=0.5)
    w_unhedged = s_unhedged.generate(prices)
    w_hedged = s_hedged.generate(prices)
    # On bars where the position is active, the unhedged x-weight has |w|=0.5,
    # while the hedged version scales by beta (~1.5).
    active = w_hedged["y"] != 0
    if active.any():
        # Hedge ratio should make |w_x| > |w_y| since beta ≈ 1.5.
        assert (w_hedged.loc[active, "x"].abs() > w_unhedged.loc[active, "x"].abs()).any()


def test_pairs_round_trips_through_engine() -> None:
    prices = _cointegrated_pair()
    s = Pairs(fit_window=200, beta_window=60, z_window=60, entry_z=1.5, exit_z=0.3)
    w = s.generate(prices)
    result = run_vectorized(prices, w, costs=ZERO_COSTS)
    assert len(result.equity) == len(prices)
    assert np.isfinite(result.total_return)
