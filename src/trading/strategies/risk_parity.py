"""Risk parity — inverse-vol weighting.

Each symbol gets a weight proportional to ``1 / realized_vol``. Symbols
whose returns are noisier get a smaller slice of the portfolio. This is the
crude (and well-studied) version of risk parity; the more sophisticated
"equal risk contribution" variant requires solving a small optimization
problem per rebalance and isn't worth the dependency for v1.

Rebalance every ``rebalance`` bars and ffill in between.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from trading.strategies.base import Strategy, StrategyParams, register


class RiskParityParams(StrategyParams):
    vol_lookback: int = Field(default=60, ge=5)
    rebalance: int = Field(default=21, ge=1)
    target_gross: float = Field(default=1.0, gt=0.0)
    """Sum of |weights| at each rebalance. Risk manager may scale further."""


@register
class RiskParity(Strategy):
    name = "risk_parity"
    Params = RiskParityParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        # Use simple returns; for crypto with extreme moves, log-returns might
        # be more honest. Daily equities don't care.
        ret = prices.pct_change()
        vol = ret.rolling(p.vol_lookback, min_periods=p.vol_lookback).std(ddof=1)
        # Avoid div-by-zero for synthetic flat series.
        inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        row_sum = inv_vol.sum(axis=1)
        # Where every symbol's vol is undefined, row_sum is 0/NaN → leave the row blank.
        w = inv_vol.div(row_sum, axis=0).fillna(0.0) * p.target_gross

        # Rebalance gating identical to xsec_momentum.
        rebal_mask = np.arange(len(prices)) % p.rebalance == 0
        w[~rebal_mask] = np.nan
        w = w.ffill().fillna(0.0)
        return w
