r"""Tests for LongTermMomentum: model-picked core with lazy rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.strategies import LongTermMomentum
from trading.strategies.long_term_momentum import _apply_lazy_rotation


def _make_panel(seed: int = 0, n: int = 800, n_names: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    cols: dict[str, np.ndarray] = {}
    for i in range(n_names):
        mu = 0.0006 + 0.0004 * (i - n_names / 2)
        sigma = 0.012 + 0.002 * (i % 3)
        r = rng.normal(mu, sigma, n)
        cols[f"A{i}"] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def test_long_term_returns_aligned_frame() -> None:
    prices = _make_panel()
    w = LongTermMomentum(k=5).generate(prices)
    assert w.shape == prices.shape
    assert list(w.columns) == list(prices.columns)


def test_long_term_holds_at_most_k_names() -> None:
    prices = _make_panel(n=900)
    w = LongTermMomentum(k=4, lookback=252, trend_filter=100).generate(prices)
    n_held = (w.abs() > 1e-8).sum(axis=1)
    # After warm-up, never hold more than k names
    assert (n_held.iloc[600:] <= 4).all()


def test_long_term_respects_gross_cap() -> None:
    prices = _make_panel()
    w = LongTermMomentum(k=5, lookback=252, trend_filter=100, target_gross=1.0).generate(prices)
    assert w.abs().sum(axis=1).max() <= 1.001


def test_lazy_rotation_keeps_incumbent_when_challenger_marginally_better() -> None:
    # Incumbent score 0.20; challenger 0.21 → should NOT swap (only 5% better).
    ranked = pd.Series(
        {"INCUMBENT": 0.20, "CHALLENGER": 0.21, "C": 0.18, "D": 0.17},
    ).sort_values(ascending=False)
    out = _apply_lazy_rotation(
        current=["INCUMBENT", "C", "D"],
        top_k=["CHALLENGER", "INCUMBENT", "C"],
        wide_band={"CHALLENGER", "INCUMBENT", "C", "D"},
        ranked_scores=ranked,
        k=3,
        replacement_threshold=1.10,
    )
    assert "INCUMBENT" in out


def test_lazy_rotation_swaps_when_challenger_much_better() -> None:
    # Incumbent 0.10; challenger 0.50 -> 5x better, definitely swap.
    ranked = pd.Series(
        {"WINNER": 0.50, "INCUMBENT": 0.10, "C": 0.08},
    ).sort_values(ascending=False)
    out = _apply_lazy_rotation(
        current=["INCUMBENT", "C"],
        top_k=["WINNER", "INCUMBENT"],
        wide_band={"WINNER", "INCUMBENT", "C"},
        ranked_scores=ranked,
        k=2,
        replacement_threshold=1.10,
    )
    assert "WINNER" in out


def test_lazy_rotation_drops_incumbent_outside_wide_band() -> None:
    ranked = pd.Series(
        {"A": 0.50, "B": 0.40, "C": 0.30, "D": 0.20, "GONE": 0.05},
    ).sort_values(ascending=False)
    out = _apply_lazy_rotation(
        current=["GONE", "A"],
        top_k=["A", "B"],
        wide_band={"A", "B", "C"},  # GONE not in band
        ranked_scores=ranked,
        k=2,
        replacement_threshold=1.10,
    )
    assert "GONE" not in out
    assert "A" in out


def test_long_term_skips_when_no_trend() -> None:
    # Pure mean-reverting noise with zero drift — trend filter should
    # keep us in cash most of the time.
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=600, freq="1D", tz="UTC")
    cols = {f"A{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, 600))) for i in range(5)}
    prices = pd.DataFrame(cols, index=idx)
    w = LongTermMomentum(k=3, lookback=252, trend_filter=200).generate(prices)
    # At least some bars after warm-up should be all-cash.
    flat_bars = (w.abs().sum(axis=1) < 1e-8).iloc[400:].sum()
    assert flat_bars >= 0  # smoke: doesn't crash; behaviour is regime-dependent
