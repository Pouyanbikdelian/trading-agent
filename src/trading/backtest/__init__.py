"""Vectorized backtester + walk-forward harness + headline metrics.

Public surface::

    from trading.backtest import run_vectorized, BacktestResult
    from trading.backtest import CostModel, compute_metrics
    from trading.backtest import expanding   # walk-forward
"""

from __future__ import annotations

from trading.backtest.costs import ZERO_COSTS, CostModel
from trading.backtest.engine import BacktestResult, run_vectorized
from trading.backtest.metrics import (
    annualized_vol,
    average_exposure,
    average_turnover,
    cagr,
    calmar,
    compute_metrics,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
)
from trading.backtest.walkforward import Fold, expanding

__all__ = [
    "BacktestResult",
    "CostModel",
    "Fold",
    "ZERO_COSTS",
    "annualized_vol",
    "average_exposure",
    "average_turnover",
    "cagr",
    "calmar",
    "compute_metrics",
    "expanding",
    "hit_rate",
    "max_drawdown",
    "run_vectorized",
    "sharpe",
    "sortino",
    "total_return",
]
