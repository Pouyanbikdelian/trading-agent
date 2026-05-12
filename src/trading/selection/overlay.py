"""Portfolio vol-targeting overlay.

Given a combined weights frame and the asset prices it was generated on,
scale the entire portfolio at each bar so the rolling realized portfolio
vol matches a fixed annualized target.

Math
----
* Asset returns: ``r_t = price_t / price_{t-1} - 1``.
* Held weights are ``w_{t-1}`` (no lookahead — same convention as the engine).
* Portfolio return: ``rp_t = sum_i w_{t-1,i} * r_{t,i}``.
* Realized vol at bar t: ``σ_t = std(rp_{t - lookback + 1 .. t}) * sqrt(periods_per_year)``.
* Scale factor: ``s_t = clip(target_vol / σ_t, 0, max_leverage)``, *shifted by 1
  bar* so today's weight uses yesterday's measured vol.

The overlay multiplies every weight by ``s_t``, leaving gross/net composition
intact — only the total size moves.

Edge cases
----------
* Realized vol of zero (e.g. zero-weight bars during warm-up) becomes
  unbounded — we clip and leave the scale at ``max_leverage``. That's
  consistent with "if vol is undetectably small, lever up to your cap."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    target_vol: float,
    lookback: int,
    periods_per_year: int,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale ``weights`` so the rolling realized portfolio vol ≈ ``target_vol``.

    ``target_vol`` is annualized (e.g. 0.10 = 10% / year).
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    common_idx = weights.index.intersection(prices.index)
    common_cols = weights.columns.intersection(prices.columns)
    w = weights.loc[common_idx, common_cols].astype(float).fillna(0.0)
    p = prices.loc[common_idx, common_cols].astype(float)

    asset_ret = p.pct_change().fillna(0.0)
    port_ret = (w.shift(1).fillna(0.0) * asset_ret).sum(axis=1)
    realized_vol = port_ret.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(
        periods_per_year
    )

    # Per-bar scale factor — shift so today's weights use yesterday's measurement.
    scale = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(upper=max_leverage).shift(1)
    # Warm-up bars (rolling NaN, plus the one we lose to the shift) → no scaling.
    scale = scale.fillna(1.0)

    return w.mul(scale, axis=0)
