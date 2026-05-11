"""Shared fixtures for strategy tests.

Deterministic price paths chosen to make state-machine logic checkable
without statistical fudge factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def idx_300d() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC", name="ts")


@pytest.fixture
def trending_up(idx_300d: pd.DatetimeIndex) -> pd.DataFrame:
    """One symbol with a clean monotonic uptrend."""
    return pd.DataFrame({"A": np.linspace(100.0, 200.0, 300)}, index=idx_300d)


@pytest.fixture
def trending_down(idx_300d: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"A": np.linspace(200.0, 100.0, 300)}, index=idx_300d)


@pytest.fixture
def mean_reverting(idx_300d: pd.DatetimeIndex) -> pd.DataFrame:
    """Sine wave around 100 — perfect for mean-reversion tests."""
    t = np.arange(300)
    wave = 100 + 10 * np.sin(2 * np.pi * t / 40)
    return pd.DataFrame({"A": wave}, index=idx_300d)


@pytest.fixture
def three_asset_random_walk(idx_300d: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.010, 300))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 300))),
            "C": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.009, 300))),
        },
        index=idx_300d,
    )
