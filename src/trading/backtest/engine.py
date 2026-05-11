"""Vectorized backtester.

Inputs
------
* ``prices``: DataFrame indexed by tz-aware DatetimeIndex; one column per
  symbol; values are the price used both for return calculation and for
  marking trades (typically adjusted close).
* ``weights``: DataFrame on the same index/columns; cell ``w_{t,i}`` is the
  *target* portfolio weight for symbol ``i`` to be held *during the next bar*.
  Strategies emit ``Signal.target_weights`` and a combiner stacks them into
  this frame; for now the caller hands us the frame directly.

Convention (no lookahead)
-------------------------
* ``w_t`` is the weight decided at the close of bar ``t``.
* ``ret_{t+1} = price_{t+1} / price_t - 1`` is the next bar's return.
* Portfolio return for bar ``t+1`` = ``sum_i w_{t,i} * ret_{t+1,i}``.
* Trades happen at the close of bar ``t`` at ``price_t``. Cost is applied
  to the bar where the trade occurred.

The engine intentionally does *not* enforce gross/net exposure caps. Risk
limits are the responsibility of the risk manager (Phase 7). Backtests run on
unbounded weights so we can see what a strategy *wants* before sizing.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from trading.backtest.costs import CostModel


class BacktestResult(BaseModel):
    """Numeric output of a backtest. Frames are stored as-is (not validated)
    because pydantic doesn't natively serialize DataFrames; the engine builds
    them and we trust ourselves."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    equity: pd.Series           # equity curve, indexed by ts
    returns: pd.Series          # per-bar net portfolio return
    gross_returns: pd.Series    # before costs
    costs: pd.Series            # per-bar cost drag (positive number)
    turnover: pd.Series         # per-bar sum of |delta_weight|
    weights: pd.DataFrame       # the input weights, aligned
    trades: pd.DataFrame        # long-form ledger: ts, symbol, delta_w, price
    initial_equity: float

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_equity - 1.0


def run_vectorized(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    costs: CostModel | None = None,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Run a vectorized backtest. See module docstring for the conventions.

    The function aligns ``prices`` and ``weights`` on their intersection of
    index and columns, so a strategy that only trades a subset of the
    universe still works. NaN weights become 0 (no position).
    """
    if costs is None:
        costs = CostModel()
    if prices.empty or weights.empty:
        raise ValueError("prices and weights must be non-empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices.index must be a DatetimeIndex")
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise ValueError("weights.index must be a DatetimeIndex")

    common_idx = prices.index.intersection(weights.index)
    common_cols = prices.columns.intersection(weights.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        raise ValueError("prices and weights have no overlapping rows/columns")

    p = prices.loc[common_idx, common_cols].astype(float)
    w = weights.loc[common_idx, common_cols].astype(float).fillna(0.0)

    if p.isna().any().any():
        raise ValueError(
            "prices contains NaN after alignment; forward-fill or drop before backtesting"
        )

    # Per-symbol simple returns. First row is NaN -> 0 (no prior price).
    ret = p.pct_change().fillna(0.0)

    # Weight held *during* bar t is w_{t-1}. shift(1).fillna(0) makes the
    # convention explicit: at t=0 we hold nothing yet.
    w_held = w.shift(1).fillna(0.0)
    gross_returns = (w_held * ret).sum(axis=1)

    # Turnover: at bar t we trade from w_{t-1} (or 0 at t=0) into w_t.
    delta = (w - w.shift(1)).abs()
    delta.iloc[0] = w.iloc[0].abs()
    turnover = delta.sum(axis=1)
    cost_series = turnover * costs.fractional

    net_returns = gross_returns - cost_series
    equity = float(initial_equity) * (1.0 + net_returns).cumprod()

    # Trade ledger: every non-zero delta becomes a row.
    trades = _build_trade_ledger(delta, p)

    return BacktestResult(
        equity=equity.rename("equity"),
        returns=net_returns.rename("returns"),
        gross_returns=gross_returns.rename("gross_returns"),
        costs=cost_series.rename("costs"),
        turnover=turnover.rename("turnover"),
        weights=w,
        trades=trades,
        initial_equity=float(initial_equity),
    )


def _build_trade_ledger(delta: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a (ts x symbol) absolute-delta-weight matrix into a long-form
    trade ledger. Rows where delta == 0 are dropped — we want one row per
    actual rebalance, not one per bar."""
    if delta.empty:
        return pd.DataFrame(columns=["ts", "symbol", "delta_weight", "price"])

    stacked = delta.stack()
    nonzero = stacked[stacked > 0]
    if nonzero.empty:
        return pd.DataFrame(columns=["ts", "symbol", "delta_weight", "price"])

    # Pull the price at the trade's bar for the symbol traded.
    idx = cast(pd.MultiIndex, nonzero.index)
    ts_vals = idx.get_level_values(0)
    sym_vals = idx.get_level_values(1)
    price_vals = prices.values[
        np.searchsorted(prices.index, ts_vals),
        [prices.columns.get_loc(s) for s in sym_vals],
    ]
    return pd.DataFrame(
        {
            "ts": ts_vals,
            "symbol": sym_vals,
            "delta_weight": nonzero.values,
            "price": price_vals,
        }
    ).reset_index(drop=True)
