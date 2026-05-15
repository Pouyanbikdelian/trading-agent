r"""Long-term momentum with *lazy rotation* — a model-picked core sleeve.

Motivation
----------
The standard top-K momentum strategy in this repo rebalances on a fixed
schedule (e.g. quarterly) and refreshes its basket every cycle, which
generates turnover on names that are still strong but no longer #1. For
a portfolio's *core* sleeve the right behaviour is the opposite: pick
long-term winners by a slow trend signal, then **hold them until they
clearly stop working** rather than churning the basket each quarter.

Construction
------------
At each rebalance bar:

1. Compute :math:`L`-bar trailing total return (default 504 bars ≈ 24
   months — long enough that day-to-day noise washes out and only
   structural winners survive).
2. Compute a *long-trend* gate: a name only qualifies if its current
   price is above its :math:`T`-bar simple moving average (default 200).
   This is Faber's classic regime filter — when SPY trades below its
   200-day MA, momentum strategies historically lose money; this gate
   sidesteps that.
3. Rank gated names by their :math:`L`-bar return; the top :math:`K`
   are the *candidate basket*.
4. **Replacement test.** For each currently-held name, only swap it out
   if BOTH:

   - it has fallen out of the top :math:`K + B` (a "tolerance band"
     around the basket boundary), AND
   - the best candidate not currently held has a momentum score at
     least ``replacement_threshold`` (e.g. 1.10 = 10%) higher.

   Otherwise: keep the existing holding. This is the "pick and hold
   unless something much better comes along" rule.
5. Size the basket by **inverse realised volatility** (true equal-risk
   weighting), capped at ``max_per_position`` and normalised to
   ``target_gross``.

Why this exists separately from top_k_momentum
----------------------------------------------
The two strategies look superficially similar but optimise different
objectives:

- ``top_k_momentum`` — maximises forward expected return given a fixed
  rebalance cadence. Suited to a satellite sleeve. Turnover is high.
- ``long_term_momentum`` — maximises *holding period* of structural
  winners. Turnover is low; tax efficiency is high (most lots qualify
  for long-term capital gains). Suited to a core sleeve.

Use the latter as the core in a core-satellite portfolio, sized at 50%
of capital by default per ``config/portfolio.example.yaml``.

Parameters
----------
``k`` — basket size when fully invested.
``lookback`` — momentum measurement window. Default 504 ≈ 24 months.
``trend_filter`` — long-trend SMA window. Default 200. Set to 0 to
    disable.
``rebalance`` — bars between *checks* (not necessarily swaps). Default
    63 ≈ quarterly.
``replacement_threshold`` — challenger's momentum must exceed
    incumbent's by this multiplicative factor before a swap fires.
    1.10 = 10% better. 1.00 = swap on any rank flip.
``replacement_band`` — incumbent stays in the basket as long as it
    remains in the top ``k + replacement_band``. Wider band = stickier
    basket.
``vol_lookback`` — bars used for inverse-vol sizing.
``target_gross`` — sum of \|weights\| at each rebalance.
``max_per_position`` — per-name cap.
``min_trend_score`` — minimum trailing return for a name to qualify
    (absolute gate). Default 0 = Antonacci-style "no losers allowed".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from trading.strategies.base import Strategy, StrategyParams, register


class LongTermMomentumParams(StrategyParams):
    k: int = Field(default=10, ge=1)
    lookback: int = Field(default=504, ge=21)
    trend_filter: int = Field(default=200, ge=0)
    rebalance: int = Field(default=63, ge=1)
    replacement_threshold: float = Field(default=1.10, ge=1.0)
    replacement_band: int = Field(default=5, ge=0)
    vol_lookback: int = Field(default=60, ge=5)
    target_gross: float = Field(default=1.0, gt=0.0)
    max_per_position: float = Field(default=0.20, gt=0.0, le=1.0)
    min_trend_score: float = Field(default=0.0)


@register
class LongTermMomentum(Strategy):
    name = "long_term_momentum"
    Params = LongTermMomentumParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n_t, n_n = prices.shape
        if n_n == 0:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Trailing return over the full lookback (no skip — at 504 bars,
        # one-month reversal is noise).
        mom = prices.pct_change(p.lookback)

        # Long-trend gate: price above SMA(trend_filter). Disabled if 0.
        if p.trend_filter > 0:
            sma = prices.rolling(p.trend_filter, min_periods=p.trend_filter).mean()
            in_trend = prices > sma
        else:
            in_trend = pd.DataFrame(True, index=prices.index, columns=prices.columns)

        # Realised vol for sizing
        log_ret = np.log(prices).diff()
        vol = log_ret.rolling(p.vol_lookback, min_periods=p.vol_lookback).std(ddof=1)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        warmup = max(p.lookback, p.trend_filter, p.vol_lookback)
        rebal_idx = np.arange(warmup, n_t, p.rebalance)

        current_basket: list[str] = []
        current_w = pd.Series(0.0, index=prices.columns)

        for i in rebal_idx:
            mu = mom.iloc[i]
            sigma = vol.iloc[i]
            gate = in_trend.iloc[i]
            if mu.isna().all() or sigma.isna().all():
                weights.iloc[i] = current_w
                continue

            # Apply absolute gate + long-trend gate
            qualified = mu.where(gate & (mu > p.min_trend_score)).dropna()

            if qualified.empty:
                # market regime is hostile — go to cash
                current_basket = []
                current_w = pd.Series(0.0, index=prices.columns)
                weights.iloc[i] = current_w
                continue

            ranked = qualified.sort_values(ascending=False)
            top_k = list(ranked.head(p.k).index)
            wide_band = set(ranked.head(p.k + p.replacement_band).index)

            # --- lazy rotation rule ---------------------------------------
            new_basket = _apply_lazy_rotation(
                current=current_basket,
                top_k=top_k,
                wide_band=wide_band,
                ranked_scores=ranked,
                k=p.k,
                replacement_threshold=p.replacement_threshold,
            )

            # Inverse-vol sizing of the resulting basket
            sig_k = sigma.reindex(new_basket).replace([np.inf, -np.inf], np.nan).dropna()
            if sig_k.empty:
                current_basket = []
                current_w = pd.Series(0.0, index=prices.columns)
                weights.iloc[i] = current_w
                continue
            inv_vol = 1.0 / sig_k
            raw = inv_vol / inv_vol.sum()
            sized = (raw * p.target_gross).clip(upper=p.max_per_position)

            current_basket = list(sized.index)
            current_w = pd.Series(0.0, index=prices.columns)
            current_w.loc[sized.index] = sized.values
            weights.iloc[i] = current_w

        # Forward-fill between rebalances
        on_rebal = np.zeros(n_t, dtype=bool)
        on_rebal[rebal_idx] = True
        non_rebal_mask = ~on_rebal
        if non_rebal_mask.any():
            weights.iloc[non_rebal_mask] = np.nan
            weights = weights.ffill()
        weights = weights.fillna(0.0)
        return weights


def _apply_lazy_rotation(
    *,
    current: list[str],
    top_k: list[str],
    wide_band: set[str],
    ranked_scores: pd.Series,
    k: int,
    replacement_threshold: float,
) -> list[str]:
    r"""Decide which holdings stay and which get replaced.

    First call (or after a flush): the basket equals ``top_k``. Subsequent
    calls keep an incumbent as long as it is still inside the wide
    tolerance band, only swapping it out when a fresh candidate beats it
    by the multiplicative threshold.
    """
    if not current:
        return top_k

    # Incumbents that fell out of the wide band — they're definitively
    # gone. Anything still in the wide band stays for now.
    kept = [s for s in current if s in wide_band]
    open_slots = k - len(kept)

    if open_slots < 0:
        # Basket is overfull (e.g. shrinking k). Drop the weakest kept.
        kept_scores = ranked_scores.reindex(kept).sort_values(ascending=False)
        return list(kept_scores.head(k).index)

    # Candidates: names in top_k not already held.
    candidates = [s for s in top_k if s not in kept]

    # For each kept-but-now-weakest slot, see if the best non-kept
    # candidate clears the replacement threshold.
    kept_scores = ranked_scores.reindex(kept).dropna()
    final = list(kept)

    for cand in candidates:
        if open_slots > 0:
            final.append(cand)
            open_slots -= 1
            continue
        cand_score = ranked_scores.get(cand, np.nan)
        if not np.isfinite(cand_score) or not kept_scores.size:
            continue
        # Identify the weakest still-in-final incumbent.
        weakest = kept_scores.idxmin()
        weakest_score = float(kept_scores.loc[weakest])
        # Avoid division by tiny/negative scores: only fire the
        # multiplicative rule when both sides are positive.
        if weakest_score > 0 and cand_score >= replacement_threshold * weakest_score:
            final = [s for s in final if s != weakest]
            final.append(cand)
            kept_scores = kept_scores.drop(weakest)

    # If we still have empty slots (incumbents flushed but no candidates),
    # fill with whatever top_k names remain unassigned.
    if len(final) < k:
        for sym in top_k:
            if sym not in final:
                final.append(sym)
                if len(final) >= k:
                    break

    return final[:k]
