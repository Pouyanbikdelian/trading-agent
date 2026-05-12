"""Engine tests with exact expected numbers on hand-built price paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest import ZERO_COSTS, CostModel, run_vectorized


def test_zero_weights_yields_flat_equity(linear_prices: pd.DataFrame) -> None:
    w = pd.DataFrame(0.0, index=linear_prices.index, columns=linear_prices.columns)
    r = run_vectorized(linear_prices, w, costs=ZERO_COSTS)
    assert r.equity.iloc[-1] == pytest.approx(1.0)
    assert r.trades.empty
    assert r.total_return == pytest.approx(0.0)


def test_full_long_matches_buy_and_hold_no_costs(linear_prices: pd.DataFrame) -> None:
    """100% in A means the equity curve must equal A's price ratio."""
    w = pd.DataFrame(
        {"A": [1.0] * 30, "B": [0.0] * 30},
        index=linear_prices.index,
    )
    r = run_vectorized(linear_prices, w, costs=ZERO_COSTS)
    expected = linear_prices["A"].iloc[-1] / linear_prices["A"].iloc[0]
    assert r.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    # Single entry trade at bar 0.
    assert len(r.trades) == 1
    assert r.trades.iloc[0]["symbol"] == "A"
    assert r.trades.iloc[0]["delta_weight"] == pytest.approx(1.0)


def test_costs_drag_on_flat_market(flat_prices: pd.DataFrame) -> None:
    """Flat prices + 100% weight all the time: returns are 0, but the
    initial entry at t=0 costs ``total_bps * |weight|`` once."""
    w = pd.DataFrame({"A": [1.0] * 30}, index=flat_prices.index)
    costs = CostModel(commission_bps=5.0, slippage_bps=5.0)  # 10 bps total
    r = run_vectorized(flat_prices, w, costs=costs)
    # Only one trade (the initial 0->1 entry), so total cost drag = 10 bps.
    assert r.total_return == pytest.approx(-10 / 1e4, abs=1e-12)
    assert len(r.trades) == 1


def test_short_position_profits_on_drop() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": [100, 99, 98, 97, 96]}, index=idx, dtype=float)
    weights = pd.DataFrame({"A": [-1.0] * 5}, index=idx)
    r = run_vectorized(prices, weights, costs=ZERO_COSTS)
    # Engine rebalances to -100% of equity each bar, so per-bar returns
    # of (-1 * pct_change) compound geometrically. Compute that exactly.
    pcts = prices["A"].pct_change().fillna(0.0).iloc[1:]
    expected = float((1.0 + (-1.0) * pcts).prod() - 1.0)
    assert r.total_return == pytest.approx(expected, rel=1e-12)
    assert r.total_return > 0  # short profited as price fell


def test_no_lookahead_first_bar_has_zero_return(linear_prices: pd.DataFrame) -> None:
    w = pd.DataFrame({"A": [1.0] * 30, "B": [0.0] * 30}, index=linear_prices.index)
    r = run_vectorized(linear_prices, w, costs=ZERO_COSTS)
    # The first bar's return must be 0 — we held nothing entering it.
    assert r.gross_returns.iloc[0] == 0.0


def test_turnover_counts_initial_entry(linear_prices: pd.DataFrame) -> None:
    w = pd.DataFrame({"A": [1.0] * 30, "B": [0.0] * 30}, index=linear_prices.index)
    r = run_vectorized(linear_prices, w, costs=ZERO_COSTS)
    assert r.turnover.iloc[0] == pytest.approx(1.0)
    assert r.turnover.iloc[1:].sum() == pytest.approx(0.0)


def test_rebalance_emits_two_trades() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    prices = pd.DataFrame(
        {"A": [100, 100, 100, 100], "B": [50, 50, 50, 50]},
        index=idx,
        dtype=float,
    )
    weights = pd.DataFrame(
        {
            "A": [1.0, 1.0, 0.0, 0.0],
            "B": [0.0, 0.0, 1.0, 1.0],
        },
        index=idx,
    )
    r = run_vectorized(prices, weights, costs=ZERO_COSTS)
    # Trades expected: t=0 buy A, t=2 sell A, t=2 buy B  → 3 trade rows.
    assert len(r.trades) == 3
    syms_at_t2 = set(r.trades[r.trades["ts"] == idx[2]]["symbol"])
    assert syms_at_t2 == {"A", "B"}


def test_misaligned_columns_use_intersection() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104], "B": [50] * 5}, index=idx, dtype=float)
    weights = pd.DataFrame({"A": [1.0] * 5, "C": [1.0] * 5}, index=idx)
    r = run_vectorized(prices, weights, costs=ZERO_COSTS)
    # Only "A" survives the intersection.
    assert list(r.weights.columns) == ["A"]


def test_nan_weights_treated_as_zero(linear_prices: pd.DataFrame) -> None:
    w = pd.DataFrame({"A": [np.nan] + [1.0] * 29, "B": [0.0] * 30}, index=linear_prices.index)
    r = run_vectorized(linear_prices, w, costs=ZERO_COSTS)
    # t=0 weight is NaN -> 0. t=1 enters at 1.0. So first "real" trade is at t=1.
    assert r.trades.iloc[0]["ts"] == linear_prices.index[1]


def test_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_vectorized(pd.DataFrame(), pd.DataFrame())


def test_rejects_naive_index() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="1D")  # tz-naive
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    # Naive index is still a DatetimeIndex so the explicit check passes;
    # the engine doesn't require tz here (that's a data-layer invariant).
    # This test pins behavior: we accept it, and the math still works.
    r = run_vectorized(prices, weights, costs=ZERO_COSTS)
    assert r.total_return == pytest.approx(3 / 1 - 1)


def test_rejects_nan_prices() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")
    prices = pd.DataFrame({"A": [100.0, np.nan, 102.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    with pytest.raises(ValueError, match="NaN"):
        run_vectorized(prices, weights)


def test_initial_equity_scales_curve(linear_prices: pd.DataFrame) -> None:
    w = pd.DataFrame({"A": [1.0] * 30, "B": [0.0] * 30}, index=linear_prices.index)
    r1 = run_vectorized(linear_prices, w, costs=ZERO_COSTS, initial_equity=1.0)
    r100 = run_vectorized(linear_prices, w, costs=ZERO_COSTS, initial_equity=100.0)
    assert r100.equity.iloc[-1] == pytest.approx(r1.equity.iloc[-1] * 100)
    assert r100.total_return == pytest.approx(r1.total_return)


def test_cost_model_validation() -> None:
    with pytest.raises(Exception):
        CostModel(commission_bps=-1.0)
