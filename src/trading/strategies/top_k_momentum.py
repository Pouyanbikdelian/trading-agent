r"""Relative + absolute momentum on a top-K cross-section.

References
----------
Antonacci, G. (2014). *Dual Momentum Investing: An Innovative Strategy
for Higher Returns with Lower Risk.* McGraw-Hill.

Jegadeesh, N., Titman, S. (1993). *Returns to buying winners and selling
losers: implications for stock market efficiency.* Journal of Finance,
48(1), 65-91.

Asness, C. S., Moskowitz, T. J., Pedersen, L. H. (2013). *Value and
momentum everywhere.* Journal of Finance, 68(3), 929-985.

Construction
------------
At each rebalance bar :math:`t \in \{0, R, 2R, \ldots\}` the strategy:

1. Computes :math:`L`-bar trailing total return for every name
   :math:`i`, excluding the most recent :math:`s` bars to remove the
   short-term-reversal contamination:

   .. math::

      \mu_i(t) = \frac{P_{i, t - s}}{P_{i, t - s - L}} - 1.

2. Applies the *absolute* momentum gate: keep only names with
   :math:`\mu_i(t) > \tau_{\text{abs}}`. When :math:`\tau_{\text{abs}}=0`
   this is Antonacci's classic dual-momentum rule — names with negative
   12-1 return are dropped to cash, regardless of cross-section rank.

3. Among gated names, keep the top :math:`K` by :math:`\mu_i(t)`.

4. Sizes the kept names by **inverse realised vol** over a
   ``vol_lookback`` window (true equal-risk weighting), normalised so the
   row's gross exposure equals ``target_gross`` (default 1.0). If fewer
   than :math:`K` names clear the absolute gate, the unfilled fraction
   stays in cash — this is what bounds drawdowns in bear markets.

5. Holds the resulting weights for :math:`R` bars until the next
   rebalance.

Why this beats vanilla momentum
-------------------------------
Plain top-K momentum on equities rides every momentum factor into the
bottom in 2000-02, 2008, and 2022 because the cross-section always has
*relative* winners even when everything is falling absolutely. Adding
the absolute gate is the difference between a strategy that delivers
the 'momentum premium' and one that survives long enough to do so. The
2018-2026 NDX backtest doesn't stress this distinction much (markets
spent most of the window in bull regimes), but the parameter exists for
the regime that will eventually show up.

Parameters
----------
``k`` — number of names held when the absolute gate doesn't bind.
``lookback`` — total ranking horizon, including the skip block.
``skip`` — bars at the end of the lookback excluded from the formation
    return (default 21 ≈ one month, the classic 12-1 formulation).
``rebalance`` — bars between portfolio rebuilds. 21 ≈ monthly.
``vol_lookback`` — bars used for the inverse-vol weighting of kept names.
``abs_momentum_threshold`` — :math:`\tau_{\text{abs}}`. Set to 0 for
    classic dual-momentum; set to None to disable the absolute gate.
``target_gross`` — sum of \|weights\| at each rebalance.
``max_per_position`` — per-name cap (default 0.20 = 20%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from trading.strategies.base import Strategy, StrategyParams, register


class TopKMomentumParams(StrategyParams):
    k: int = Field(default=10, ge=1)
    lookback: int = Field(default=252, ge=21)
    skip: int = Field(default=21, ge=0)
    rebalance: int = Field(default=21, ge=1)
    vol_lookback: int = Field(default=60, ge=5)
    abs_momentum_threshold: float | None = Field(default=0.0)
    """If set, exclude names whose formation return is below this floor.
    0.0 = Antonacci's dual-momentum gate (drop losers to cash)."""
    target_gross: float = Field(default=1.0, gt=0.0)
    max_per_position: float = Field(default=0.20, gt=0.0, le=1.0)


@register
class TopKMomentum(Strategy):
    name = "top_k_momentum"
    Params = TopKMomentumParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n_t, n_n = prices.shape
        if n_n == 0:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- formation return: (P_{t-skip} / P_{t-skip-L}) - 1 -----------
        eff_lookback = p.lookback - p.skip
        if eff_lookback <= 0:
            raise ValueError("lookback must exceed skip")
        shifted = prices.shift(p.skip)
        formation = shifted.pct_change(eff_lookback)

        # --- realised vol for inverse-vol sizing -------------------------
        log_ret = np.log(prices).diff()
        vol = log_ret.rolling(p.vol_lookback, min_periods=p.vol_lookback).std(ddof=1)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- rebalance bars: every R bars, starting after warm-up --------
        warmup = max(p.lookback, p.vol_lookback)
        rebal_idx = np.arange(warmup, n_t, p.rebalance)

        # ``current_w`` holds the latest selected row; off-rebalance bars
        # carry it forward. This is what the runner sees if it queries the
        # last row.
        current_w = pd.Series(0.0, index=prices.columns)

        for i in rebal_idx:
            mu = formation.iloc[i]
            sigma = vol.iloc[i]
            if mu.isna().all() or sigma.isna().all():
                continue

            # Absolute gate
            if p.abs_momentum_threshold is not None:
                mu = mu.where(mu > p.abs_momentum_threshold)
            valid = mu.dropna()
            if valid.empty:
                # all-cash — clear positions
                current_w = pd.Series(0.0, index=prices.columns)
                weights.iloc[i] = current_w
                continue

            # Top-K by formation return among the gated names
            keep = valid.nlargest(p.k).index

            # Inverse-vol weighting of kept names
            sig_k = sigma.reindex(keep).replace([np.inf, -np.inf], np.nan).dropna()
            if sig_k.empty:
                current_w = pd.Series(0.0, index=prices.columns)
                weights.iloc[i] = current_w
                continue
            inv_vol = 1.0 / sig_k
            raw = inv_vol / inv_vol.sum()
            sized = (raw * p.target_gross).clip(upper=p.max_per_position)
            # If clipping reduced gross, that's fine — leftover is cash.
            current_w = pd.Series(0.0, index=prices.columns)
            current_w.loc[sized.index] = sized.values
            weights.iloc[i] = current_w

        # Forward-fill between rebalances. Pandas .ffill respects NaN
        # explicitly, so we use a sentinel: zero rows in `weights` at
        # non-rebalance bars get filled by the most recent rebalance row.
        on_rebal = np.zeros(n_t, dtype=bool)
        on_rebal[rebal_idx] = True
        idx_series = pd.Series(np.where(on_rebal)[0], dtype=int)
        for i in range(n_t):
            if not on_rebal[i]:
                # find the most recent rebalance bar at or before i
                last = idx_series[idx_series <= i]
                if last.empty:
                    continue
                weights.iloc[i] = weights.iloc[int(last.iloc[-1])].values
        return weights
