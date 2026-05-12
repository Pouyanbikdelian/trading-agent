"""EMA crossover — the simplest possible trend filter.

Rules
-----
* Long when ``ema_fast > ema_slow``.
* Short (or flat) when ``ema_fast <= ema_slow``.

Pandas' ``ewm(span=N).mean()`` is computed with ``adjust=True`` by default,
which biases the first few values toward the original observation. We use
``adjust=False`` to get the textbook EMA recurrence
``ema_t = a * x_t + (1-a) * ema_{t-1}`` with ``a = 2/(span+1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from trading.strategies.base import Strategy, StrategyParams, register


class EmaCrossParams(StrategyParams):
    fast_span: int = Field(default=20, ge=2)
    slow_span: int = Field(default=100, ge=3)
    allow_short: bool = False
    weight_per_asset: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _fast_lt_slow(self) -> EmaCrossParams:
        if self.fast_span >= self.slow_span:
            raise ValueError("fast_span must be < slow_span")
        return self


@register
class EmaCross(Strategy):
    name = "ema_cross"
    Params = EmaCrossParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        fast = prices.ewm(span=p.fast_span, adjust=False).mean()
        slow = prices.ewm(span=p.slow_span, adjust=False).mean()

        # The signal is decided at the close of bar t. The backtester then
        # holds that weight during bar t+1, which is the standard no-lookahead
        # convention; the EMAs themselves do not peek ahead.
        long_mask = fast > slow
        weights = np.where(long_mask, 1.0, -1.0 if p.allow_short else 0.0).astype(float)
        df = pd.DataFrame(weights, index=prices.index, columns=prices.columns) * p.weight_per_asset

        # Warm-up: zero out the first ``slow_span`` bars where the slow EMA
        # hasn't yet stabilized. Without this, a freshly initialized slow EMA
        # equals the first price, which makes the cross trigger spuriously.
        warmup = max(p.slow_span, p.fast_span)
        df.iloc[:warmup] = 0.0
        return df
