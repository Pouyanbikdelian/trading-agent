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

    # --- correlation diversification --------------------------------------
    corr_window: int = Field(default=63, ge=5)
    """Bars of daily-return history used to estimate the correlation matrix."""

    max_pairwise_corr: float = Field(default=0.70, gt=0.0, le=1.0)
    """Absolute pairwise correlation threshold below which a candidate is
    considered 'decorrelated' from an already-selected name."""

    min_decorrelated: int = Field(default=0, ge=0)
    """At least this many of the K selected names must mutually clear the
    pairwise-correlation threshold. 0 disables the filter and recovers
    pure top-K-by-momentum behaviour."""


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

        # Daily returns for the correlation filter, when enabled.
        ret_for_corr = prices.pct_change() if p.min_decorrelated > 0 else None

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

            # Top-K by formation return, optionally diversified
            if p.min_decorrelated > 0 and ret_for_corr is not None:
                window = ret_for_corr.iloc[max(0, i - p.corr_window + 1) : i + 1]
                keep = _decorrelated_topk(
                    valid,
                    window,
                    k=p.k,
                    min_decorrelated=p.min_decorrelated,
                    max_pairwise_corr=p.max_pairwise_corr,
                )
            else:
                keep = list(valid.nlargest(p.k).index)

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

        # Forward-fill between rebalances using pandas' vectorised ffill.
        # Non-rebalance rows are zero (the default) — we mark them NaN
        # first so ffill carries the most recent rebalance row forward.
        on_rebal = np.zeros(n_t, dtype=bool)
        on_rebal[rebal_idx] = True
        non_rebal_mask = ~on_rebal
        if non_rebal_mask.any():
            # weights.values can be read-only on newer pandas; iloc-assignment
            # always writes back through the BlockManager safely.
            weights.iloc[non_rebal_mask] = np.nan
            weights = weights.ffill()
        weights = weights.fillna(0.0)
        return weights


def _decorrelated_topk(
    candidates: pd.Series,
    returns_window: pd.DataFrame,
    *,
    k: int,
    min_decorrelated: int,
    max_pairwise_corr: float,
) -> list[str]:
    r"""Greedy correlation-diversified top-K selection.

    Candidates are sorted by ``candidates`` (formation return) descending.
    We then walk the list and admit names in two phases:

    *Phase 1* — fill up to ``min_decorrelated`` slots. A candidate is
    admitted only if its maximum absolute correlation with the already-
    admitted set is below ``max_pairwise_corr``.

    *Phase 2* — fill remaining slots up to ``k`` with the next-best-
    momentum names regardless of correlation.

    This guarantees at least ``min_decorrelated`` of the final basket are
    pairwise low-correlation, while still allocating most capital to the
    strongest trends. If too few candidates can pass Phase 1 we fall back
    to pure momentum order; this is non-fatal — the system still trades.
    """
    sorted_candidates = candidates.sort_values(ascending=False).index.tolist()
    if not sorted_candidates or k <= 0:
        return []
    if min_decorrelated <= 0 or len(returns_window) < 5:
        return sorted_candidates[:k]

    # Performance: the greedy algorithm only ever looks down to the top-K
    # bracket plus enough overflow to fill phase-2. A pool of 5K candidates
    # is conservative; computing the corr matrix over more is wasted work
    # (O(N^2) per rebalance times thousands of rebalances). On a sp500
    # universe this caps the corr matrix at ~50 x 50 instead of 500 x 500.
    pool_size = min(len(sorted_candidates), max(k * 5, 30))
    pool = sorted_candidates[:pool_size]
    common = [s for s in pool if s in returns_window.columns]
    if not common:
        return sorted_candidates[:k]
    corr = returns_window[common].corr().abs()

    decorrelated: list[str] = [common[0]]
    overflow: list[str] = []

    for sym in common[1:]:
        if len(decorrelated) >= min_decorrelated:
            overflow.append(sym)
            continue
        if sym not in corr.index:
            overflow.append(sym)
            continue
        # Max absolute correlation with already-admitted set
        cmax = float(corr.loc[sym, decorrelated].max())
        if not np.isfinite(cmax):
            # NaN correlation (e.g. constant returns) — treat as zero
            cmax = 0.0
        if cmax < max_pairwise_corr:
            decorrelated.append(sym)
        else:
            overflow.append(sym)

    combined = decorrelated + overflow
    return combined[:k]
