"""Rank candidate strategies by deflated Sharpe.

The selection layer's job is to pick which strategies make it into the
portfolio combiner. A raw Sharpe leaderboard overfits to the strategies
you happened to test the most. ``rank_strategies`` reports both the raw
Sharpe and the deflated Sharpe (probability that the true Sharpe exceeds
the expected best-of-N null), and sorts by the deflated number.

Input
-----
``returns_by_strategy``: ``{strategy_name: per-bar net returns Series}``.
The returns should come from out-of-sample backtests — typically the
output of ``backtest.walkforward.expanding(...).returns``. Passing IS
returns gives an honest PSR but a meaningless DSR.

Output
------
A DataFrame indexed by strategy name with columns:
``sharpe`` (annualized), ``psr``, ``dsr``, ``n_obs``, ``skew``, ``kurt``.
Sorted by ``dsr`` descending.
"""

from __future__ import annotations

import math

import pandas as pd

from trading.selection.scores import (
    annualize_sharpe,
    deflated_sharpe,
    moments,
    per_period_sharpe,
    probabilistic_sharpe,
)


def rank_strategies(
    returns_by_strategy: dict[str, pd.Series],
    *,
    periods_per_year: int,
    sr_benchmark_annual: float = 0.0,
) -> pd.DataFrame:
    """Score every strategy with PSR and DSR, sort by DSR descending."""
    if not returns_by_strategy:
        return pd.DataFrame(columns=["sharpe", "psr", "dsr", "n_obs", "skew", "kurt"])

    n_trials = len(returns_by_strategy)
    rows: list[dict[str, float | str]] = []

    for name, r in returns_by_strategy.items():
        r = r.dropna()
        n = len(r)
        sr_period = per_period_sharpe(r)
        sk, kt = moments(r)
        bench_per_period = sr_benchmark_annual / math.sqrt(periods_per_year)
        psr = probabilistic_sharpe(sr_period, n, sk, kt, sr_benchmark=bench_per_period)
        dsr = deflated_sharpe(sr_period, n, sk, kt, n_trials=n_trials)
        rows.append(
            {
                "name": name,
                "sharpe": annualize_sharpe(sr_period, periods_per_year),
                "psr": psr,
                "dsr": dsr,
                "n_obs": float(n),
                "skew": sk,
                "kurt": kt,
            }
        )

    df = pd.DataFrame(rows).set_index("name").sort_values("dsr", ascending=False)
    return df
