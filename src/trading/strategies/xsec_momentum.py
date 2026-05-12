"""Cross-sectional momentum — long winners, short losers (optional).

Construction
------------
* On each rebalance bar, compute each symbol's return over a lookback window
  ending ``skip`` bars ago (the classic "12-1" formulation: 12-month total
  return ending 1 month before the formation date, so the most recent month
  is excluded — that recent month is the well-documented short-term reversal
  effect).
* Rank symbols by formation-period return. Long the top ``top_frac``, short
  the bottom ``bottom_frac`` (or skip shorts if ``long_only``).
* Hold equal-weighted within each leg. Total gross exposure is
  ``top_frac + bottom_frac`` if long/short, else just ``top_frac``.

Why monthly? The recent-month-skip insight is sample-size sensitive — daily
"21-1 day momentum" is a different beast. We rebalance every ``rebalance``
bars (default 21 = monthly on daily data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from trading.strategies.base import Strategy, StrategyParams, register


class XSecMomentumParams(StrategyParams):
    lookback: int = Field(default=252, ge=2)
    """Formation window in bars. 252 ≈ 12 months on daily data."""

    skip: int = Field(default=21, ge=0)
    """Bars to exclude at the end of the lookback (short-term reversal blackout)."""

    rebalance: int = Field(default=21, ge=1)
    """Rebalance every N bars. 21 ≈ monthly on daily data."""

    top_frac: float = Field(default=0.2, gt=0.0, le=1.0)
    bottom_frac: float = Field(default=0.2, ge=0.0, le=1.0)
    long_only: bool = False

    @model_validator(mode="after")
    def _fracs_valid(self) -> XSecMomentumParams:
        if self.top_frac + self.bottom_frac > 1.0:
            raise ValueError("top_frac + bottom_frac must be <= 1.0")
        return self


@register
class XSecMomentum(Strategy):
    name = "xsec_momentum"
    Params = XSecMomentumParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n_symbols = prices.shape[1]
        if n_symbols < 2:
            raise ValueError("cross-sectional momentum needs >= 2 symbols")

        # Formation return per (ts, symbol): price ratio over [t - lookback, t - skip].
        # Shift by ``skip`` to drop the recent window, then look back ``lookback - skip`` bars.
        eff_lookback = p.lookback - p.skip
        if eff_lookback <= 0:
            raise ValueError("lookback must exceed skip")
        formation = prices.shift(p.skip).pct_change(eff_lookback)

        # Compute the target weight cross-section at every bar, then apply rebalance gating.
        n_top = max(1, round(n_symbols * p.top_frac))
        n_bot = max(0 if p.long_only else 1, round(n_symbols * p.bottom_frac))

        weights = np.zeros(prices.shape, dtype=float)
        for i in range(len(prices)):
            row = formation.iloc[i]
            if row.isna().any():
                continue
            ranks = row.rank(method="first", ascending=False)
            longs = ranks <= n_top
            shorts = ranks > n_symbols - n_bot
            if longs.any():
                weights[i, longs.values] = p.top_frac / longs.sum()
            if shorts.any() and not p.long_only:
                weights[i, shorts.values] = -p.bottom_frac / shorts.sum()

        # Rebalance gating: only update the weight on rebalance bars; carry forward in between.
        w = pd.DataFrame(weights, index=prices.index, columns=prices.columns)
        rebalance_mask = np.arange(len(prices)) % p.rebalance == 0
        # Set non-rebalance rows to NaN so ffill carries the most recent rebalance values.
        w[~rebalance_mask] = np.nan
        w = w.ffill().fillna(0.0)
        return w
