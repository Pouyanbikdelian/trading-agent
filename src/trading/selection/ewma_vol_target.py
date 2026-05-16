r"""EWMA (exponentially-weighted) volatility forecast — a smarter
vol-target overlay than the rolling-std default.

Why this exists
---------------
The standard ``vol_target`` overlay estimates the portfolio's realised
vol with a rolling window standard deviation. That estimator gives
**equal weight** to every bar in the window. The problem: yesterday's
shock carries the same information weight as a shock 60 bars ago — but
**volatility clusters**. After a 5σ day, tomorrow's expected vol is
markedly higher; the rolling-std barely moves until 60 bars have
elapsed.

EWMA (the Engle (1982) / RiskMetrics 1996 estimator) instead gives
each past observation a weight that decays geometrically:

.. math::

   \sigma^2_t = \lambda \sigma^2_{t-1} + (1 - \lambda) r^2_{t-1}

where :math:`\lambda \in [0, 1)` is the decay (default 0.94 for daily
returns, the RiskMetrics standard). Equivalently this is GARCH(1,1)
with :math:`\omega = 0` and :math:`\alpha + \beta = 1` — the
"integrated GARCH" simplification.

What changes vs the rolling-std vol_target
------------------------------------------
The scale factor used to size the portfolio still equals
``target_vol / forecast_vol``. The only thing changing is *how the
forecast is computed*. With EWMA:

* Volatility regime changes are detected ~3x faster than rolling-std
* No abrupt step-changes when a big-shock bar enters/exits the window
* Slightly smoother portfolio gross over time

Drop-in compatible: same call signature as ``vol_target`` except for
the additional ``lam`` parameter. ``lookback`` becomes a warm-up window
(after which EWMA is fully primed) — it does NOT define the estimator
window the way it does for rolling-std.

Caveats
-------
EWMA assumes returns have zero mean — fine at daily frequency where
the drift is tiny relative to the vol. It does not allow for time-
varying mean drift. For our setup this is the right trade-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    target_vol: float,
    lookback: int = 60,
    lam: float = 0.94,
    periods_per_year: int = 252,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    r"""Scale ``weights`` so the EWMA-forecast portfolio vol ≈ ``target_vol``.

    Parameters
    ----------
    weights, prices
        Combined weights from the strategy + the asset prices on which
        portfolio returns are computed.
    target_vol
        Annualised vol target, e.g. ``0.10`` for 10% / year.
    lookback
        Warm-up bars before scaling kicks in. The estimator itself is
        recursive and uses all available history past the warm-up.
    lam
        EWMA decay. ``0.94`` is RiskMetrics daily; smaller values weight
        recent observations more aggressively. Must be in ``(0, 1)``.
    periods_per_year
        252 for daily equities, 365 for crypto, 12 for monthly.
    max_leverage
        Hard cap on the scale factor. ``1.0`` (default) = downsize-only;
        the overlay never adds gross.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    common_idx = weights.index.intersection(prices.index)
    common_cols = weights.columns.intersection(prices.columns)
    w = weights.loc[common_idx, common_cols].astype(float).fillna(0.0)
    p = prices.loc[common_idx, common_cols].astype(float)

    asset_ret = p.pct_change().fillna(0.0)
    # Held-yesterday convention: today's return uses yesterday's weights.
    port_ret = (w.shift(1).fillna(0.0) * asset_ret).sum(axis=1)

    # EWMA variance recursion. Seed with the rolling std over the warm-up
    # window so we don't artificially start at zero.
    n = len(port_ret)
    var_arr = np.zeros(n)
    seed_var = float(port_ret.iloc[:lookback].var(ddof=1)) if n >= lookback else 0.0
    var_arr[: min(lookback, n)] = seed_var
    for t in range(lookback, n):
        prev_var = var_arr[t - 1] if var_arr[t - 1] > 0 else seed_var
        var_arr[t] = lam * prev_var + (1.0 - lam) * port_ret.iloc[t - 1] ** 2

    realised_vol = pd.Series(np.sqrt(var_arr) * np.sqrt(periods_per_year), index=port_ret.index)

    scale = (target_vol / realised_vol).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(upper=max_leverage)
    # Shift so today's weights use yesterday's forecast (no lookahead).
    scale = scale.shift(1).fillna(1.0)

    return w.mul(scale, axis=0)
