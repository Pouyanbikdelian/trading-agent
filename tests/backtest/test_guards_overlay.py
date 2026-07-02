"""Guard overlay — hermetic, synthetic paths with known outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest.costs import CostModel
from trading.backtest.guards_overlay import run_with_guards

_N = 120


def _frame(path: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2025-01-06", periods=len(path))
    prices = pd.DataFrame({"XYZ": path}, index=idx)
    weights = pd.DataFrame({"XYZ": np.ones(len(path))}, index=idx)
    return prices, weights


def test_trailing_stop_caps_crash() -> None:
    # Ramp +50% then crash to 60: the trail should exit near the top.
    up = np.linspace(100, 150, 60)
    down = np.linspace(150, 60, 60)
    prices, weights = _frame(np.concatenate([up, down]))
    naked = (prices["XYZ"].iloc[-1] / prices["XYZ"].iloc[0]) - 1
    res = run_with_guards(
        prices, weights, costs=CostModel(commission_bps=0, slippage_bps=0), cycle_every=10_000
    )
    assert res.stop_exits == 1
    # Guarded book keeps most of the ramp; naked loses money overall.
    assert res.equity.iloc[-1] - 1 > 0.2 > naked


def test_static_tp_exits_at_target() -> None:
    prices, weights = _frame(np.linspace(100, 200, _N))  # +100% smooth ramp
    res = run_with_guards(
        prices,
        weights,
        costs=CostModel(commission_bps=0, slippage_bps=0),
        tp_pct=25.0,
        cycle_every=10_000,
    )
    assert res.tp_exits == 1
    # Exited at ~+25%; final equity far below the +100% naked ride.
    assert 0.2 < res.equity.iloc[-1] - 1 < 0.35


def test_cycle_reenters_after_stop() -> None:
    # Crash triggers the stop, then a recovery: the weekly cycle re-buys
    # (strategy still wants the name), so the book participates again.
    path = np.concatenate(
        [np.linspace(100, 140, 40), np.linspace(140, 90, 20), np.linspace(90, 160, 60)]
    )
    prices, weights = _frame(path)
    res = run_with_guards(
        prices, weights, costs=CostModel(commission_bps=0, slippage_bps=0), cycle_every=5
    )
    assert res.stop_exits >= 1
    assert res.reentries >= 2  # initial entry + at least one re-entry
    assert res.equity.iloc[-1] > 1.0


def test_stop_distance_clamped() -> None:
    # A silent tape (tiny vol) must still get the trail_min floor: an 8%
    # dip should NOT stop out when min is widened to 10.
    rng = np.random.default_rng(3)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.0005, _N))
    calm[60] = calm[59] * 0.94  # one -6% shock
    prices, weights = _frame(calm)
    res = run_with_guards(
        prices,
        weights,
        costs=CostModel(commission_bps=0, slippage_bps=0),
        trail_min_pct=10.0,
        cycle_every=10_000,
    )
    assert res.stop_exits == 0


def test_nan_prices_rejected() -> None:
    prices, weights = _frame(np.linspace(100, 110, _N))
    prices.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_with_guards(prices, weights)


def test_ratchet_exits_where_classic_holds() -> None:
    """+50% ramp then a 5% dip: classic 8% trail holds through it, the
    ratcheted trail (floor .4, tighten 1.2 -> 3.2% at +50%) locks the gain."""
    path = np.concatenate([np.linspace(100, 150, 80), np.linspace(150, 142, 20)])
    prices, weights = _frame(path)
    kw = dict(costs=CostModel(commission_bps=0, slippage_bps=0), cycle_every=10_000)
    classic = run_with_guards(prices, weights, **kw)
    ratchet = run_with_guards(prices, weights, ratchet_floor=0.4, ratchet_tighten=1.2, **kw)
    assert classic.stop_exits == 0
    assert ratchet.stop_exits == 1
    assert ratchet.equity.iloc[-1] > classic.equity.iloc[-1]


def test_ratchet_disabled_matches_classic_exactly() -> None:
    rng = np.random.default_rng(11)
    path = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, _N))
    prices, weights = _frame(path)
    kw = dict(costs=CostModel(commission_bps=1, slippage_bps=2), cycle_every=5)
    a = run_with_guards(prices, weights, **kw)
    b = run_with_guards(prices, weights, ratchet_floor=None, ratchet_tighten=None, **kw)
    c = run_with_guards(prices, weights, ratchet_floor=1.0, ratchet_tighten=0.0, **kw)
    assert a.equity.equals(b.equity) and a.equity.equals(c.equity)
