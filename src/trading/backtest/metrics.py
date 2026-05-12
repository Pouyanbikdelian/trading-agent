"""Performance metrics for backtest results.

All functions take pandas Series of returns or an equity curve and return a
plain float. They assume the input is regularly spaced (one return per bar)
and that ``periods_per_year`` matches the bar frequency:

* daily equities (US):  252
* daily crypto:         365
* hourly (US RTH):      252 * 6.5 ≈ 1638
* hourly crypto:        24 * 365 = 8760

We deliberately do not auto-detect the periodicity — too many places to get
it wrong silently. The caller decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.backtest.engine import BacktestResult


def total_return(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Annualized geometric return. Falls back to 0 on degenerate input."""
    if len(equity) < 2:
        return 0.0
    n_periods = len(equity) - 1
    if n_periods <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return float("-inf")
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


_STD_EPS = 1e-12
"""Threshold below which we treat sample std as effectively zero.

Float roundoff makes ``pd.Series([0.01]*N).std()`` come back as ~1e-19
rather than exactly 0; without a tolerance we'd return a nonsense Sharpe of
~1e+17 for constant returns. 1e-12 sits well below realistic per-bar
return volatilities (>1e-4 even for low-vol portfolios)."""


def sharpe(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio. ``rf`` is the annual risk-free rate; we
    de-annualize it before subtracting from per-bar returns."""
    if len(returns) < 2:
        return 0.0
    rf_per_period = rf / periods_per_year
    excess = returns - rf_per_period
    std = excess.std(ddof=1)
    if std < _STD_EPS or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    """Sharpe variant using only downside deviation (returns below ``rf``)."""
    if len(returns) < 2:
        return 0.0
    rf_per_period = rf / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")  # no losses ever recorded
    dd_std = np.sqrt(np.mean(downside**2))
    if dd_std == 0:
        return 0.0
    return float(excess.mean() / dd_std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown as a negative fraction."""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(equity: pd.Series, periods_per_year: int) -> float:
    """CAGR / |max_drawdown|. Returns inf when there's no drawdown."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return float("inf")
    return float(cagr(equity, periods_per_year) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of bars with positive net return. Zero-return bars are
    excluded so flat / no-position periods don't dilute the signal."""
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def average_turnover(turnover: pd.Series) -> float:
    if len(turnover) == 0:
        return 0.0
    return float(turnover.mean())


def average_exposure(weights: pd.DataFrame) -> float:
    """Mean of sum(|weight|) across bars. 1.0 = fully invested on average."""
    if weights.empty:
        return 0.0
    return float(weights.abs().sum(axis=1).mean())


def compute_metrics(
    result: BacktestResult, periods_per_year: int, rf: float = 0.0
) -> dict[str, float]:
    """Bundle of headline metrics — what you'd print in a backtest summary."""
    return {
        "total_return": total_return(result.equity),
        "cagr": cagr(result.equity, periods_per_year),
        "ann_vol": annualized_vol(result.returns, periods_per_year),
        "sharpe": sharpe(result.returns, periods_per_year, rf=rf),
        "sortino": sortino(result.returns, periods_per_year, rf=rf),
        "max_drawdown": max_drawdown(result.equity),
        "calmar": calmar(result.equity, periods_per_year),
        "hit_rate": hit_rate(result.returns),
        "avg_turnover": average_turnover(result.turnover),
        "avg_exposure": average_exposure(result.weights),
        "n_trades": float(len(result.trades)),
    }
