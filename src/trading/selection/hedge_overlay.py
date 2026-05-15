r"""Proportional market-hedge overlay.

Given a long-only weights frame and a market benchmark (default SPY),
add a *negative* weight on the benchmark sized to absorb a fraction of
the portfolio's rolling beta to that benchmark. The result is a
partially beta-neutral portfolio whose gross exposure goes up
(long names + short hedge) but whose net market exposure goes down.

Why proportional, not on/off
----------------------------
The classic "halve to cash when VIX spikes" overlay is jarring — it
generates a single huge turnover event and is easy to wrong-time. A
proportional beta hedge instead **moves smoothly**:

* When the market is calm (low VIX), the hedge fraction is small.
* As realized risk rises, the hedge scales up.
* The portfolio's high-conviction long names are never sold.

This matches Yan's "very dynamic… not halv… rather hedges or diversify"
specification. Tax-wise it's cleaner too: instead of selling lots and
realising gains, we simply add a short on a broad-market ETF.

Math
----
At each bar :math:`t`:

1. Compute the portfolio return :math:`r^p_t = \sum_i w_{t-1,i} r_{t,i}`
   using the held-yesterday convention (no lookahead).
2. Compute the rolling beta of :math:`r^p` to the benchmark return
   :math:`r^m` over the last ``beta_lookback`` bars:

   .. math::

      \beta_t = \frac{\text{cov}(r^p, r^m)}{\text{var}(r^m)}.

3. Compute a *hedge intensity* :math:`\alpha_t \in [0, 1]`. If a VIX
   series is supplied, intensity rises with VIX percentile:

   .. math::

      \alpha_t = \text{clip}\big(\text{VIXpct}_t \cdot \text{vix\_scale} + \text{floor}, 0, \text{cap}\big).

   If no VIX series is supplied, :math:`\alpha_t` is fixed at
   ``base_intensity``.

4. The hedge weight is :math:`-\alpha_t \beta_t \cdot \text{gross}_t`,
   where :math:`\text{gross}_t = \sum_i |w_{t,i}|`.

5. Cap the hedge at ``max_hedge`` (default 0.5 = 50% short SPY at most).

The overlay leaves the long sleeve untouched and only adds/updates the
benchmark column. This makes it composable with other overlays
(vol-target, dip-buy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def beta_hedge(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    benchmark_prices: pd.Series | None = None,
    beta_lookback: int = 63,
    base_intensity: float = 0.5,
    vix_series: pd.Series | None = None,
    vix_floor: float = 0.0,
    vix_scale: float = 1.0,
    intensity_cap: float = 1.0,
    max_hedge: float = 0.5,
) -> pd.DataFrame:
    r"""Append a proportional negative weight on ``benchmark``.

    The benchmark must either be a column in ``prices`` *or* supplied
    via ``benchmark_prices``. If both are absent the function returns
    ``weights`` unchanged (graceful no-op — easier to compose).

    Parameters
    ----------
    weights, prices
        Long-sleeve weights and the price frame used to compute returns.
    benchmark
        Column name to use for the hedge. Added to ``weights`` if not
        already present.
    benchmark_prices
        Optional explicit benchmark price series (e.g. an SPY series
        loaded outside ``prices``). Aligned to ``weights.index``.
    beta_lookback
        Rolling window for the beta regression. 63 bars ≈ one quarter.
    base_intensity
        Fraction of measured beta to hedge when no VIX signal is given,
        or the additive baseline when one is. 0.0 = no hedge, 1.0 = full
        beta neutrality.
    vix_series
        Optional. If supplied, hedge intensity rises with the VIX
        percentile rank computed within the series.
    vix_floor, vix_scale, intensity_cap
        Linear mapping from VIX percentile to hedge intensity:
        ``α = clip(pct * vix_scale + vix_floor, 0, intensity_cap)``.
    max_hedge
        Hard cap on the absolute hedge weight, expressed as fraction of
        portfolio gross.
    """
    if beta_lookback < 5:
        raise ValueError("beta_lookback must be >= 5")

    common_idx = weights.index.intersection(prices.index)
    w = weights.loc[common_idx].astype(float).fillna(0.0).copy()
    p = prices.loc[common_idx].astype(float)

    # Resolve the benchmark price series.
    bench_p: pd.Series | None
    if benchmark_prices is not None:
        bench_p = benchmark_prices.reindex(common_idx).astype(float)
    elif benchmark in p.columns:
        bench_p = p[benchmark]
    else:
        return weights  # no-op: nothing to hedge against

    bench_ret = bench_p.pct_change().fillna(0.0)

    # Long-sleeve return at each bar (held-yesterday convention).
    asset_ret = p.reindex(columns=w.columns).pct_change().fillna(0.0)
    port_ret = (w.shift(1).fillna(0.0) * asset_ret).sum(axis=1)

    # Rolling beta of port_ret on bench_ret
    cov = port_ret.rolling(beta_lookback, min_periods=beta_lookback).cov(bench_ret)
    var = bench_ret.rolling(beta_lookback, min_periods=beta_lookback).var(ddof=1)
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)

    # Hedge intensity (alpha) per bar
    if vix_series is not None and not vix_series.empty:
        vix_aligned = vix_series.reindex(common_idx).ffill()
        # Percentile rank within the supplied series — values in [0, 1].
        pct = vix_aligned.rank(pct=True)
        alpha = (pct * vix_scale + vix_floor).clip(lower=0.0, upper=intensity_cap)
    else:
        alpha = pd.Series(base_intensity, index=common_idx)

    # Use yesterday's beta with today's weights (no lookahead).
    beta = beta.shift(1).fillna(0.0)
    alpha = alpha.shift(1).fillna(0.0)

    gross = w.abs().sum(axis=1)
    hedge_weight = (-alpha * beta * gross).clip(lower=-max_hedge, upper=max_hedge)
    # Only short the market; if portfolio beta is negative we don't go long.
    hedge_weight = hedge_weight.clip(upper=0.0)

    out = w.copy()
    if benchmark not in out.columns:
        out[benchmark] = 0.0
    out[benchmark] = out[benchmark] + hedge_weight
    return out
