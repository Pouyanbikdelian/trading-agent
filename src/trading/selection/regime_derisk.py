r"""Regime-based de-risk overlay — convert to cash in broken markets.

Replaces the short-SPY hedge with a long-only, leverage-free defensive
overlay. Cuts gross exposure when the broad market is structurally
broken, then restores it when the market repairs.

Why this instead of a short hedge
---------------------------------
Most retail brokerage accounts (cash accounts, ISAs, registered
retirement accounts, etc.) can't short. And even when they can, an
on-top short adds gross exposure, breaching the no-leverage rule we
operate under. A *de-risk* overlay solves both: it never adds positions,
only scales the existing book down toward cash when conditions warrant.

It is also designed to **ignore noise**. The rule only fires on a
sustained breakdown of the benchmark (typically SPY) — not a 1-day
gap, a 3-day pullback, or a noisy vol spike that mean-reverts. That's
the difference between "tactical de-risk" and "panic selling every
wiggle." The cost is some lag at major tops, which is the right
trade-off: we'd rather catch most of a 30% drawdown after it starts
than churn the book every time a 5% pullback shows up.

The rule (Faber-style, with hysteresis)
---------------------------------------
Two regimes, with a configurable smooth interpolation between:

* **Healthy.** Benchmark close > SMA(``trend_window``) for at least
  ``confirm_days`` consecutive days. Full target gross is held.
* **Broken.** Benchmark close < SMA(``trend_window``) for at least
  ``confirm_days`` consecutive days. Gross is multiplied by
  ``derisk_scale`` (default 0.3 = 30% invested, 70% cash).

A second, deeper de-risk stage fires on the **"death cross"**
(SMA(``fast_window``) < SMA(``trend_window``)) — the more severe of
the two regimes wins.

Hysteresis (the ``confirm_days`` rule) is what keeps the overlay from
toggling on/off every other day during a choppy market. The cost is
~``confirm_days`` bars of slippage at regime transitions; the benefit
is no whipsaw.

Math
----
Let :math:`B_t` be the benchmark close at bar :math:`t`,
:math:`S^L_t` the long SMA (window ``trend_window``), and
:math:`S^F_t` the fast SMA (window ``fast_window``).

State at bar :math:`t`:

.. math::

   \text{broken}_t = \mathbf{1}\{B_{t-k} < S^L_{t-k}\ \forall k \in [0, C)\}

where :math:`C = \text{confirm\_days}`.

.. math::

   \text{death\_cross}_t = \mathbf{1}\{S^F_{t-k} < S^L_{t-k}\ \forall k \in [0, C)\}

Scale factor:

.. math::

   s_t = \begin{cases}
     \text{deep\_derisk\_scale} & \text{if death\_cross}_t \\
     \text{derisk\_scale}       & \text{if broken}_t \\
     1                          & \text{otherwise}
   \end{cases}

Output weights: :math:`w'_{t,i} = s_t \cdot w_{t,i}`. The composition
of the book never changes — only its total size.

Returning to full exposure works the same way: once the benchmark
closes above its long SMA for ``confirm_days`` consecutive bars, the
scale returns to 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_derisk(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark: str = "SPY",
    benchmark_prices: pd.Series | None = None,
    trend_window: int = 200,
    fast_window: int = 50,
    confirm_days: int = 5,
    derisk_scale: float = 0.30,
    deep_derisk_scale: float = 0.10,
) -> pd.DataFrame:
    r"""Scale ``weights`` toward cash when the benchmark trend is broken.

    Parameters
    ----------
    weights, prices
        Long-only weights and the price frame that produced them. The
        overlay never adds new columns or flips signs.
    benchmark
        Column in ``prices`` to use as the regime signal (default SPY).
    benchmark_prices
        Optional explicit benchmark series. Aligned to ``weights.index``.
        Useful when SPY isn't already in your universe's price frame.
    trend_window
        Long SMA window. 200 ≈ classic Faber.
    fast_window
        Fast SMA window for the death-cross stage.
    confirm_days
        Number of consecutive days the breakdown signal must persist
        before the overlay reacts. 5 ≈ one trading week — short enough
        to catch real bears, long enough to ignore noise.
    derisk_scale
        Multiplier applied to ``weights`` when the broken-trend regime
        is confirmed. 0.30 = 70% to cash.
    deep_derisk_scale
        Multiplier applied during the death-cross regime (more severe).
        0.10 = 90% to cash.

    Returns
    -------
    Re-scaled ``weights``. Index and columns match the input.
    """
    if not 0.0 <= deep_derisk_scale <= derisk_scale <= 1.0:
        raise ValueError("scales must satisfy 0 <= deep_derisk_scale <= derisk_scale <= 1.0")
    if confirm_days < 1:
        raise ValueError("confirm_days must be >= 1")
    if trend_window <= fast_window:
        raise ValueError("trend_window must exceed fast_window")

    common_idx = weights.index.intersection(prices.index)
    w = weights.loc[common_idx].astype(float).copy()
    p = prices.loc[common_idx]

    # Resolve the benchmark price series.
    bench: pd.Series | None
    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(common_idx).astype(float)
    elif benchmark in p.columns:
        bench = p[benchmark].astype(float)
    else:
        # No benchmark available — overlay is a no-op rather than a crash.
        # Makes the live runner robust to missing-data cycles.
        return weights

    sma_long = bench.rolling(trend_window, min_periods=trend_window).mean()
    sma_fast = bench.rolling(fast_window, min_periods=fast_window).mean()

    below_long = (bench < sma_long).astype(int)
    fast_below_long = (sma_fast < sma_long).astype(int)

    # "Persistent" = condition held for `confirm_days` consecutive bars.
    # A rolling-min over `confirm_days` of a 0/1 series is 1 iff every
    # bar in the window is 1. This is the cheapest way to express the
    # hysteresis condition.
    persistent_breakdown = below_long.rolling(confirm_days, min_periods=confirm_days).min() == 1
    persistent_death_cross = (
        fast_below_long.rolling(confirm_days, min_periods=confirm_days).min() == 1
    )

    # Choose the most defensive scale that applies.
    scale = pd.Series(1.0, index=common_idx)
    scale = scale.where(~persistent_breakdown, derisk_scale)
    scale = scale.where(~persistent_death_cross, deep_derisk_scale)

    # Shift by 1 — today's weights use yesterday's confirmed signal
    # (no lookahead at decision time).
    scale = scale.shift(1).fillna(1.0)
    scale = scale.replace([np.inf, -np.inf], 1.0)

    return w.mul(scale, axis=0)
