"""Pairs trading on a cointegrated pair via z-score of the spread.

Construction
------------
1. Run an Engle-Granger cointegration test on the first ``fit_window`` bars
   of the (y, x) pair. If the p-value exceeds ``coint_pvalue`` the strategy
   emits all-zero weights and logs a warning — there's no statistical basis
   to expect mean reversion.
2. Estimate a rolling hedge ratio ``beta_t`` over ``beta_window`` bars via
   the closed-form OLS estimator ``cov(y, x) / var(x)``. Avoids importing
   statsmodels for every bar.
3. Define the spread ``s_t = y_t - beta_t * x_t`` and its rolling z-score
   over ``z_window`` bars.
4. State machine:
     * ``z < -entry_z``  → long  y, short x.
     * ``z > +entry_z``  → short y, long  x.
     * ``|z| < exit_z``  → flat.
   The previous state ffills until a new entry or exit triggers.

Limitations (deliberate, document them)
---------------------------------------
* Cointegration is tested *once* at the head of the series, not rolled. A
  pair that decohereres later will still get traded. Phase 5 (selection) is
  where rolling cointegration belongs.
* The "y" symbol is taken to be the first column and "x" the second. Pass
  the prices DataFrame with the columns in your desired role order.
* Beta is the ratio of weights when ``beta_hedge`` is on. Otherwise we treat
  both legs symmetrically (``±weight_per_leg``); this leaves a residual
  dollar-exposure when ``beta != 1`` but matches the simpler textbook write-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from trading.core.logging import logger
from trading.strategies.base import Strategy, StrategyParams, register


class PairsParams(StrategyParams):
    fit_window: int = Field(default=252, ge=30)
    beta_window: int = Field(default=60, ge=10)
    z_window: int = Field(default=60, ge=10)
    entry_z: float = Field(default=2.0, gt=0.0)
    exit_z: float = Field(default=0.5, ge=0.0)
    coint_pvalue: float = Field(default=0.05, gt=0.0, le=1.0)
    require_cointegration: bool = True
    weight_per_leg: float = Field(default=0.5, gt=0.0)
    beta_hedge: bool = False

    @model_validator(mode="after")
    def _exit_lt_entry(self) -> PairsParams:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        return self


@register
class Pairs(Strategy):
    name = "pairs"
    Params = PairsParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        if prices.shape[1] != 2:
            raise ValueError("pairs strategy requires exactly 2 symbols")

        y_col, x_col = prices.columns[0], prices.columns[1]
        y = prices[y_col]
        x = prices[x_col]

        zero_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if len(prices) < p.fit_window:
            logger.bind(strategy=self.name).warning(
                f"only {len(prices)} bars; need {p.fit_window} for cointegration test"
            )
            return zero_weights

        if p.require_cointegration:
            from statsmodels.tsa.stattools import coint  # lazy import

            head_y = y.iloc[: p.fit_window].dropna()
            head_x = x.iloc[: p.fit_window].dropna()
            if len(head_y) < p.fit_window or len(head_x) < p.fit_window:
                return zero_weights
            _, pvalue, _ = coint(head_y, head_x)
            if pvalue > p.coint_pvalue:
                logger.bind(strategy=self.name).warning(
                    f"cointegration p-value {pvalue:.3f} > threshold {p.coint_pvalue}; "
                    f"emitting no trades for ({y_col}, {x_col})"
                )
                return zero_weights

        # Rolling OLS slope: beta_t = cov(y, x) / var(x) over [t-window+1, t].
        cov = y.rolling(p.beta_window, min_periods=p.beta_window).cov(x)
        var_x = x.rolling(p.beta_window, min_periods=p.beta_window).var(ddof=1)
        beta = cov / var_x

        spread = y - beta * x
        mu = spread.rolling(p.z_window, min_periods=p.z_window).mean()
        sigma = spread.rolling(p.z_window, min_periods=p.z_window).std(ddof=1)
        # Shift the z by one bar so today's weight decision doesn't peek at
        # today's spread value (the spread uses today's close).
        z = ((spread - mu) / sigma).shift(1)

        long_y = z < -p.entry_z
        short_y = z > p.entry_z
        exit_signal = z.abs() < p.exit_z

        event = pd.Series(np.nan, index=prices.index)
        event[long_y] = 1.0
        event[short_y] = -1.0
        event[exit_signal] = 0.0
        sign = event.ffill().fillna(0.0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[y_col] = sign * p.weight_per_leg
        if p.beta_hedge:
            # Lag beta so the position uses yesterday's hedge ratio, not today's.
            weights[x_col] = -sign * beta.shift(1).fillna(0.0) * p.weight_per_leg
        else:
            weights[x_col] = -sign * p.weight_per_leg

        return weights
