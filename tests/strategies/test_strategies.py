"""Behavior tests for each built-in strategy.

These tests use hand-shaped price paths so we can assert specific
state-machine transitions rather than aggregate statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.backtest import ZERO_COSTS, run_vectorized
from trading.strategies import (
    Donchian,
    EmaCross,
    RiskParity,
    Rsi2,
    XSecMomentum,
    ZScoreMeanRev,
)

# ---------------------------------------------------------------- Donchian ----


def test_donchian_long_after_breakout(idx_300d: pd.DatetimeIndex) -> None:
    # Flat for 30 bars then a jump higher → breakout on bar 30 of the trend.
    p = pd.DataFrame({"A": [100.0] * 30 + [120.0] * 270}, index=idx_300d)
    s = Donchian(lookback=10, allow_short=False)
    w = s.generate(p)
    # Warm-up period: zero. Post-breakout: 1.0.
    assert w["A"].iloc[:10].eq(0.0).all()
    # Bar 30 is the first close > rolling-max-of-previous-bars.
    assert w["A"].iloc[30] == pytest.approx(1.0)
    assert w["A"].iloc[40] == pytest.approx(1.0)


def test_donchian_shorts_on_downside_if_allowed(idx_300d: pd.DatetimeIndex) -> None:
    # Up then down so we get a long entry, then a short entry.
    p = pd.DataFrame(
        {"A": np.concatenate([np.full(30, 100.0), np.full(30, 120.0), np.full(240, 80.0)])},
        index=idx_300d,
    )
    s = Donchian(lookback=10, allow_short=True)
    w = s.generate(p)
    assert w["A"].iloc[35] == pytest.approx(1.0)  # long after upside break
    assert w["A"].iloc[70] == pytest.approx(-1.0)  # flipped short


def test_donchian_flat_default_after_downside(idx_300d: pd.DatetimeIndex) -> None:
    p = pd.DataFrame(
        {"A": np.concatenate([np.full(30, 100.0), np.full(30, 120.0), np.full(240, 80.0)])},
        index=idx_300d,
    )
    s = Donchian(lookback=10, allow_short=False)
    w = s.generate(p)
    assert w["A"].iloc[70] == pytest.approx(0.0)


# ---------------------------------------------------------------- EmaCross ----


def test_emacross_long_in_uptrend(trending_up: pd.DataFrame) -> None:
    s = EmaCross(fast_span=5, slow_span=20)
    w = s.generate(trending_up)
    # Past the warm-up the fast EMA must be above the slow on a monotonic uptrend.
    assert w["A"].iloc[100:].eq(1.0).all()
    # Warm-up window must be zeroed.
    assert w["A"].iloc[:20].eq(0.0).all()


def test_emacross_short_in_downtrend_when_allowed(trending_down: pd.DataFrame) -> None:
    s = EmaCross(fast_span=5, slow_span=20, allow_short=True)
    w = s.generate(trending_down)
    assert w["A"].iloc[100:].eq(-1.0).all()


def test_emacross_validates_fast_lt_slow() -> None:
    with pytest.raises(Exception, match="fast_span"):
        EmaCross(fast_span=50, slow_span=10)


# --------------------------------------------------------- XSec Momentum ----


def test_xsec_momentum_longs_winners_shorts_losers(idx_300d: pd.DatetimeIndex) -> None:
    # A trends up, B is flat, C trends down. Top quintile (1 of 3) = A,
    # bottom quintile (1 of 3) = C. Middle (B) is flat.
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 200, 300),
            "B": np.full(300, 100.0),
            "C": np.linspace(200, 100, 300),
        },
        index=idx_300d,
    )
    s = XSecMomentum(lookback=63, skip=5, rebalance=21, top_frac=1 / 3, bottom_frac=1 / 3)
    w = s.generate(prices)
    # After warm-up + first rebalance bar, A should be long, C short, B zero.
    last = w.iloc[-1]
    assert last["A"] > 0
    assert last["C"] < 0
    assert last["B"] == 0


def test_xsec_momentum_long_only(idx_300d: pd.DatetimeIndex) -> None:
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 200, 300),
            "B": np.full(300, 100.0),
            "C": np.linspace(200, 100, 300),
        },
        index=idx_300d,
    )
    s = XSecMomentum(
        lookback=63, skip=5, rebalance=21, top_frac=1 / 3, bottom_frac=1 / 3, long_only=True
    )
    w = s.generate(prices)
    # No shorts anywhere.
    assert (w >= 0).all().all()


def test_xsec_momentum_requires_two_symbols(idx_300d: pd.DatetimeIndex) -> None:
    prices = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=idx_300d)
    with pytest.raises(ValueError, match=">= 2"):
        XSecMomentum().generate(prices)


# ---------------------------------------------------------------- RSI(2) ----


def test_rsi2_entry_on_oversold_in_uptrend(idx_300d: pd.DatetimeIndex) -> None:
    # Steep trend gives SMA50 enough slack that a small one-bar dip can push
    # RSI(2) below the entry threshold without breaking the regime filter.
    p = np.linspace(100.0, 300.0, 300)
    p[-2] = p[-3] * 0.98  # mild single-bar dip
    prices = pd.DataFrame({"A": p}, index=idx_300d)
    s = Rsi2(rsi_period=2, regime_sma=50, entry_threshold=20.0, exit_sma=5)
    w = s.generate(prices)
    assert (w["A"].iloc[-2:] > 0).any()


def test_rsi2_no_entry_in_downtrend(trending_down: pd.DataFrame) -> None:
    s = Rsi2(rsi_period=2, regime_sma=50, entry_threshold=10.0, exit_sma=5)
    w = s.generate(trending_down)
    # Regime filter (price > SMA200) should keep us out for the whole period
    # after a monotonic decline — the price never sits above its own SMA.
    assert w["A"].eq(0.0).all()


# -------------------------------------------------------- ZScore meanrev ----


def test_zscore_long_when_below_mean(mean_reverting: pd.DataFrame) -> None:
    s = ZScoreMeanRev(window=10, entry_z=1.0, exit_z=0.2, allow_short=True, use_log_price=False)
    w = s.generate(mean_reverting)
    # The strategy must enter both long and short over a full sine wave.
    assert (w["A"] > 0).any()
    assert (w["A"] < 0).any()


def test_zscore_validates_exit_lt_entry() -> None:
    with pytest.raises(Exception, match="exit_z"):
        ZScoreMeanRev(window=10, entry_z=1.0, exit_z=2.0)


def test_zscore_no_short_when_disabled(mean_reverting: pd.DataFrame) -> None:
    s = ZScoreMeanRev(window=10, entry_z=1.0, exit_z=0.2, allow_short=False, use_log_price=False)
    w = s.generate(mean_reverting)
    assert (w["A"] >= 0).all()


# ----------------------------------------------------------- Risk Parity ----


def test_risk_parity_assigns_more_to_lower_vol(idx_300d: pd.DatetimeIndex) -> None:
    rng = np.random.default_rng(0)
    # A has 1% daily vol; B has 4% daily vol.
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.04, 300))),
        },
        index=idx_300d,
    )
    s = RiskParity(vol_lookback=60, rebalance=21, target_gross=1.0)
    w = s.generate(prices)
    last = w.iloc[-1]
    assert last["A"] > last["B"]
    # Gross exposure approximately equals target.
    assert last.abs().sum() == pytest.approx(1.0, rel=1e-9)


# ----------------------------- Integration: backtester accepts each output ----


@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: Donchian(lookback=20),
        lambda: EmaCross(fast_span=10, slow_span=30),
        lambda: XSecMomentum(lookback=60, skip=5, rebalance=21, top_frac=0.34, bottom_frac=0.34),
        lambda: Rsi2(),
        lambda: ZScoreMeanRev(window=20, entry_z=1.5, exit_z=0.5),
        lambda: RiskParity(vol_lookback=30, rebalance=21),
    ],
)
def test_every_strategy_round_trips_through_engine(
    three_asset_random_walk: pd.DataFrame, strategy_factory
) -> None:
    s = strategy_factory()
    w = s.generate(three_asset_random_walk)
    assert w.shape == three_asset_random_walk.shape
    result = run_vectorized(three_asset_random_walk, w, costs=ZERO_COSTS)
    assert len(result.equity) == len(three_asset_random_walk)
    assert np.isfinite(result.total_return)
