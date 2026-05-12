"""Metrics tests — hand-computed numbers on tiny series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading.backtest import (
    ZERO_COSTS,
    annualized_vol,
    average_exposure,
    average_turnover,
    cagr,
    calmar,
    compute_metrics,
    hit_rate,
    max_drawdown,
    run_vectorized,
    sharpe,
    sortino,
    total_return,
)


def _equity(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx)


def _returns(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=idx)


def test_total_return_basic() -> None:
    assert total_return(_equity([1.0, 1.1, 1.21])) == pytest.approx(0.21)
    assert total_return(_equity([])) == 0.0


def test_cagr_compounds_correctly() -> None:
    # 252 daily returns of exactly 10% total: CAGR ≈ 10% per year.
    eq = _equity([1.0] + [1.0] * 251 + [1.1])
    # 252 periods means exactly 1.0 year of returns.
    eq = pd.Series(
        np.linspace(1.0, 1.1, 253),
        index=pd.date_range("2024-01-01", periods=253, freq="1D", tz="UTC"),
    )
    out = cagr(eq, periods_per_year=252)
    assert out == pytest.approx(0.1, rel=0.05)


def test_cagr_handles_zero_or_negative_growth() -> None:
    assert cagr(_equity([1.0, 0.0]), 252) == float("-inf")
    assert cagr(_equity([1.0]), 252) == 0.0
    assert cagr(_equity([]), 252) == 0.0


def test_max_drawdown_simple_case() -> None:
    eq = _equity([1.0, 1.2, 0.8, 1.5])
    # Peak 1.2 -> trough 0.8 = -33.33%.
    assert max_drawdown(eq) == pytest.approx(0.8 / 1.2 - 1.0)


def test_max_drawdown_zero_when_monotone() -> None:
    assert max_drawdown(_equity([1.0, 1.1, 1.2])) == 0.0


def test_sharpe_constant_returns() -> None:
    # std == 0 → defined as 0 (avoid div by zero).
    assert sharpe(_returns([0.01] * 10), 252) == 0.0


def test_sharpe_known_value() -> None:
    # Returns alternating +1%/-1%: mean 0, std > 0 → Sharpe = 0.
    r = _returns([0.01, -0.01] * 50)
    assert sharpe(r, 252) == pytest.approx(0.0, abs=1e-9)


def test_sortino_infinite_when_no_losses() -> None:
    r = _returns([0.01] * 10)
    assert math.isinf(sortino(r, 252))


def test_sortino_finite_with_losses() -> None:
    r = _returns([0.02, -0.01, 0.01, -0.005, 0.015])
    out = sortino(r, 252)
    assert math.isfinite(out)
    # Downside-only volatility is smaller than full-sample volatility,
    # so Sortino > Sharpe for the same series.
    assert out > sharpe(r, 252)


def test_calmar_inf_with_no_drawdown() -> None:
    eq = _equity([1.0, 1.1, 1.2])
    assert calmar(eq, 252) == float("inf")


def test_hit_rate_ignores_zeros() -> None:
    r = _returns([0.0, 0.01, 0.0, -0.01, 0.01])
    # Non-zero bars: [+0.01, -0.01, +0.01] → 2/3.
    assert hit_rate(r) == pytest.approx(2 / 3)


def test_annualized_vol_scales_with_sqrt_periods() -> None:
    r = _returns([0.01, -0.01] * 50)
    # std of [+0.01, -0.01]*N is 0.01 (with ddof=1, ~0.01005).
    v = annualized_vol(r, 252)
    assert v == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_average_turnover_and_exposure() -> None:
    t = _returns([0.5, 0.5, 0.0])
    assert average_turnover(t) == pytest.approx(1 / 3)

    w = pd.DataFrame(
        {"A": [0.5, 0.5, 0.0], "B": [0.5, -0.5, 0.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"),
    )
    assert average_exposure(w) == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_compute_metrics_returns_full_bundle() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=idx)
    weights = pd.DataFrame({"A": [1.0] * 10}, index=idx)
    r = run_vectorized(prices, weights, costs=ZERO_COSTS)
    m = compute_metrics(r, periods_per_year=252)
    # Every advertised metric is present.
    for key in (
        "total_return",
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "avg_turnover",
        "avg_exposure",
        "n_trades",
    ):
        assert key in m
        assert isinstance(m[key], float)
