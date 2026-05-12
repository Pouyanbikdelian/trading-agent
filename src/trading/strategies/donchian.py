"""Donchian channel breakout — classic trend-following.

Rules
-----
* Go long when the close breaks above the rolling-N high (excluding today).
* Go flat (or short, if ``allow_short``) when it breaks below the rolling-N low.
* Hold the position until the *opposite* breakout fires — a position-flipping
  state machine, not a one-bar signal.

The "excluding today" detail matters: if we used the rolling max *including*
today, the close that triggers the breakout would be the same value as the
max, so the comparison would never strictly clear. Shifting the rolling
window by one bar is the standard fix and removes lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from trading.strategies.base import Strategy, StrategyParams, register


class DonchianParams(StrategyParams):
    lookback: int = Field(default=55, ge=2)
    """Bars used to compute the rolling high/low. Classic Turtle = 20 (short)
    or 55 (long); we default to the longer system."""

    allow_short: bool = False
    """If True, downside breakout flips to -1.0; otherwise it flattens to 0."""

    weight_per_asset: float = Field(default=1.0, gt=0.0)
    """Target weight when in a position. The combiner / risk manager rescales."""


@register
class Donchian(Strategy):
    name = "donchian"
    Params = DonchianParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        # Shift by 1 so today's bar can't see today's own value in the rolling window.
        upper = prices.shift(1).rolling(p.lookback, min_periods=p.lookback).max()
        lower = prices.shift(1).rolling(p.lookback, min_periods=p.lookback).min()

        # State: +1 when long, 0 when flat, -1 when short (if allowed).
        long_trigger = prices > upper
        short_trigger = prices < lower

        state = np.zeros(prices.shape, dtype=float)
        # Vectorize the position state per column using ffill of the trigger.
        for j, col in enumerate(prices.columns):
            # Encode events: +1 on long break, -1 on short break (or 0 if not allowed),
            # NaN elsewhere; forward-fill to carry the position.
            event = pd.Series(np.nan, index=prices.index)
            event[long_trigger[col]] = 1.0
            if p.allow_short:
                event[short_trigger[col]] = -1.0
            else:
                event[short_trigger[col]] = 0.0
            event = event.ffill().fillna(0.0)
            state[:, j] = event.values

        weights = (
            pd.DataFrame(state, index=prices.index, columns=prices.columns) * p.weight_per_asset
        )
        return weights
