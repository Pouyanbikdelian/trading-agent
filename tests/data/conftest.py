"""Shared fixtures for data-layer tests.

Generates synthetic OHLCV frames in the canonical schema so cache/adapter
tests don't need network or vendor libs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument


def make_bars(
    start: datetime,
    periods: int,
    freq: str = "1D",
    seed: int = 0,
    with_adj_close: bool = True,
) -> pd.DataFrame:
    """Build a synthetic OHLCV frame matching the canonical schema."""
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC", name="ts")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=periods))
    open_ = close + rng.normal(0, 0.2, size=periods)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, size=periods))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, size=periods))
    volume = rng.integers(1_000, 100_000, size=periods).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_close": close if with_adj_close else np.nan,
        },
        index=idx,
    )


@pytest.fixture
def aapl() -> Instrument:
    return Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def btc() -> Instrument:
    return Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, exchange="binance")


@pytest.fixture
def eurusd() -> Instrument:
    return Instrument(symbol="EURUSD", asset_class=AssetClass.FX)


@pytest.fixture
def synthetic_daily() -> pd.DataFrame:
    return make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=30, freq="1D")
