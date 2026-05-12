"""Z-score mean-reversion on rolling residuals.

Rules
-----
* For each symbol, compute the z-score of the price (or log-price) against
  its rolling mean over a window:
      z_t = (p_t - rolling_mean_t) / rolling_std_t
* Enter long when ``z < -entry_z`` (price unusually cheap).
* Enter short when ``z > +entry_z`` (price unusually expensive, optional).
* Exit when ``|z| < exit_z`` (mean has been recaptured).

Why log price? Reduces heteroscedasticity — a $1 std on a $10 stock is
worlds away from $1 on a $1000 stock. Toggleable for tests that pin against
exact linear-price calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from trading.strategies.base import Strategy, StrategyParams, register


class ZScoreMeanRevParams(StrategyParams):
    window: int = Field(default=20, ge=3)
    entry_z: float = Field(default=2.0, gt=0.0)
    exit_z: float = Field(default=0.5, ge=0.0)
    use_log_price: bool = True
    allow_short: bool = True
    weight_per_asset: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _exit_lt_entry(self) -> ZScoreMeanRevParams:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        return self


@register
class ZScoreMeanRev(Strategy):
    name = "zscore_meanrev"
    Params = ZScoreMeanRevParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        x = np.log(prices) if p.use_log_price else prices
        mu = x.rolling(p.window, min_periods=p.window).mean()
        sigma = x.rolling(p.window, min_periods=p.window).std(ddof=1)
        # Use yesterday's z (shift 1) so we don't peek at today's close while
        # deciding today's weight. The backtester adds a second shift when
        # it holds w_t during bar t+1, so total lag is 2 bars — conservative.
        z = ((x - mu) / sigma).shift(1)

        long_entry = z < -p.entry_z
        short_entry = z > p.entry_z
        exit_signal = z.abs() < p.exit_z

        weights = np.zeros(prices.shape, dtype=float)
        for j, col in enumerate(prices.columns):
            event = pd.Series(np.nan, index=prices.index)
            event[long_entry[col]] = 1.0
            if p.allow_short:
                event[short_entry[col]] = -1.0
            event[exit_signal[col]] = 0.0
            event = event.ffill().fillna(0.0)
            weights[:, j] = event.values

        return (
            pd.DataFrame(weights, index=prices.index, columns=prices.columns) * p.weight_per_asset
        )
