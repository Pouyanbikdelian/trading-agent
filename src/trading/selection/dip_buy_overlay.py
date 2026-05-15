r"""Buy-the-dip overlay.

When a name we already hold drops materially from its recent peak,
*add* to the position (within the per-name cap) instead of trimming.
The intuition is "I picked this name on a trend signal — a 5% pullback
doesn't invalidate the trend, it improves the entry."

This overlay is designed to compose with the long-term momentum core:
the core picks structural winners and is reluctant to swap them out;
the dip-buy overlay leans into temporary weakness on those same names.

Math
----
For each name :math:`i` and each bar :math:`t`:

1. Track the *peak* price observed while the name has been held:

   .. math::

      \text{peak}_{i,t} = \max_{s \le t,\ w_{i,s} > 0} P_{i,s}.

   The peak resets to the current price whenever a position is opened
   from flat. (We don't average down on a position we never owned —
   that would be picking falling knives.)

2. Compute the drawdown from peak:

   .. math::

      d_{i,t} = \frac{\text{peak}_{i,t} - P_{i,t}}{\text{peak}_{i,t}}.

3. If :math:`d_{i,t} \ge \text{trigger}` (e.g. 0.05 = 5%) and the
   underlying trend is still up (price above its long SMA, when the
   filter is enabled), boost the weight:

   .. math::

      w'_{i,t} = \min\big(w_{i,t} \cdot (1 + \text{boost} \cdot d_{i,t}/\text{trigger}),\ \text{max\_per\_position}\big).

4. The boost is *level-scaled* by the drawdown: a 5% dip adds the
   minimum boost, a 10% dip adds 2× that, etc., up to the per-name cap.

5. Cool-down: after a successful dip-buy, the name's peak is reset to
   the current price. This prevents the overlay from boosting again on
   the same drawdown.

Why not just hand it to the risk manager?
-----------------------------------------
The risk manager's job is to *gate* orders against a fixed limit set;
it doesn't form views. This overlay forms a view ("buy more on dips of
held names"), produces target weights consistent with that view, and
hands the weights to the risk manager exactly as any strategy would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dip_buy(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    trigger: float = 0.05,
    boost: float = 0.20,
    max_per_position: float = 0.20,
    trend_filter: int = 200,
) -> pd.DataFrame:
    r"""Boost held weights into drawdowns from peak.

    Parameters
    ----------
    weights, prices
        Target weights and aligned prices. ``prices`` must include every
        column in ``weights``.
    trigger
        Drawdown-from-peak below which the dip-buy fires (default 5%).
    boost
        Per-trigger multiplicative boost to the weight (default 20%).
        Scaled by ``d / trigger`` so deeper dips boost more.
    max_per_position
        Hard ceiling on any single name. The boost never breaches this.
    trend_filter
        Long SMA window — the dip-buy only fires when price is still
        above the SMA. Set to 0 to disable the filter (will buy any dip).
    """
    if trigger <= 0:
        raise ValueError("trigger must be positive")
    if boost < 0:
        raise ValueError("boost must be non-negative")

    common_idx = weights.index.intersection(prices.index)
    w = weights.loc[common_idx].astype(float).fillna(0.0).copy()
    p = prices.reindex_like(w).astype(float)

    if trend_filter > 0:
        sma = p.rolling(trend_filter, min_periods=trend_filter).mean()
        in_trend = p > sma
    else:
        in_trend = pd.DataFrame(True, index=w.index, columns=w.columns)

    out = w.copy()
    # Iterate by column — the peak-tracking is sequential per name but
    # each column is independent so we don't pay a cross-sectional cost.
    for sym in w.columns:
        if sym not in p.columns:
            continue
        w_col = w[sym].to_numpy()
        p_col = p[sym].to_numpy()
        trend_col = (
            in_trend[sym].to_numpy() if sym in in_trend.columns else np.ones_like(p_col, dtype=bool)
        )
        out_col = out[sym].to_numpy().copy()

        held = False
        peak = np.nan
        for t in range(len(w_col)):
            wt = w_col[t]
            pt = p_col[t]
            if not np.isfinite(pt):
                continue
            if wt > 0:
                if not held:
                    held = True
                    peak = pt
                else:
                    peak = max(peak, pt) if np.isfinite(peak) else pt
                # Drawdown from peak
                if np.isfinite(peak) and peak > 0:
                    d = (peak - pt) / peak
                    if d >= trigger and bool(trend_col[t]):
                        scale = 1.0 + boost * (d / trigger)
                        boosted = min(wt * scale, max_per_position)
                        out_col[t] = boosted
                        # Reset the peak so we don't fire repeatedly on
                        # the same drawdown — next dip-buy needs a new
                        # high water mark.
                        peak = pt
            else:
                held = False
                peak = np.nan

        out[sym] = out_col

    return out
