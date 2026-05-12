"""Shared fixtures for backtest tests.

Hand-built deterministic price paths so we can write tests with exact
expected numbers — no statistical tolerances on toy data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def idx_30d() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC", name="ts")


@pytest.fixture
def linear_prices(idx_30d: pd.DatetimeIndex) -> pd.DataFrame:
    """Two symbols, both rising linearly. Useful for exact-math tests."""
    return pd.DataFrame(
        {
            "A": np.linspace(100.0, 129.0, 30),  # +29 over 30 bars
            "B": np.linspace(50.0, 79.0, 30),  # +29 over 30 bars
        },
        index=idx_30d,
    )


@pytest.fixture
def flat_prices(idx_30d: pd.DatetimeIndex) -> pd.DataFrame:
    """Constant prices — pure cost-drag test bed."""
    return pd.DataFrame({"A": [100.0] * 30}, index=idx_30d)


@pytest.fixture
def long_only_weights(idx_30d: pd.DatetimeIndex) -> pd.DataFrame:
    """50/50 long both symbols for the full period."""
    return pd.DataFrame(
        {"A": [0.5] * 30, "B": [0.5] * 30},
        index=idx_30d,
    )
