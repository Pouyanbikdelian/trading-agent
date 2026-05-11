"""Larry Connors' RSI(2) — short-horizon mean reversion.

Rules
-----
* Compute RSI with a 2-bar lookback.
* Enter long when ``RSI(2) < entry_threshold`` AND the price is above its
  long-term moving average (regime filter — don't catch falling knives).
* Exit when the close crosses above its ``exit_sma`` SMA.

This is one of the most-cited retail mean-reversion systems. We keep the
default parameters from Connors' book; users can override via params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from trading.strategies.base import Strategy, StrategyParams, register


class Rsi2Params(StrategyParams):
    rsi_period: int = Field(default=2, ge=2)
    regime_sma: int = Field(default=200, ge=2)
    """Long-only filter: skip entries when price is below this SMA."""
    entry_threshold: float = Field(default=10.0, gt=0.0, lt=50.0)
    exit_sma: int = Field(default=5, ge=2)
    weight_per_asset: float = Field(default=1.0, gt=0.0)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI with EMA smoothing (the standard, not the SMA variant)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0, RSI is conventionally 100 (no losses).
    rsi = rsi.where(avg_loss > 0, 100.0)
    return rsi


@register
class Rsi2(Strategy):
    name = "rsi2"
    Params = Rsi2Params

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rsi = prices.apply(lambda s: _rsi(s, p.rsi_period))
        regime = prices > prices.rolling(p.regime_sma, min_periods=p.regime_sma).mean()
        exit_band = prices > prices.rolling(p.exit_sma, min_periods=p.exit_sma).mean()

        entry = (rsi < p.entry_threshold) & regime  # True on bars where we want to *be* long
        exit_ = exit_band                            # True on bars where we want to be flat

        # State machine: enter on True, exit on True-exit. Implement vectorized with ffill.
        # Encode: +1 on entry, 0 on exit, NaN otherwise, then ffill.
        weights = np.zeros(prices.shape, dtype=float)
        for j, col in enumerate(prices.columns):
            event = pd.Series(np.nan, index=prices.index)
            event[entry[col]] = 1.0
            event[exit_[col]] = 0.0
            # If a bar has both entry and exit signals, treat exit as winning
            # (the recipe is to exit on a close above the exit SMA — exits beat re-entries).
            both = entry[col] & exit_[col]
            event[both] = 0.0
            event = event.ffill().fillna(0.0)
            weights[:, j] = event.values

        return pd.DataFrame(weights, index=prices.index, columns=prices.columns) * p.weight_per_asset
