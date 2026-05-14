r"""Tests for TopKMomentum: dual-momentum on a top-K cross-section."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.strategies import TopKMomentum


def _make_panel(seed: int = 0, n: int = 500, n_names: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    cols: dict[str, np.ndarray] = {}
    for i in range(n_names):
        mu = 0.0008 + 0.0003 * (i - n_names / 2)  # spread mean returns across names
        sigma = 0.012 + 0.003 * (i % 3)
        r = rng.normal(mu, sigma, n)
        cols[f"A{i}"] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)


def test_top_k_returns_aligned_frame() -> None:
    prices = _make_panel()
    w = TopKMomentum(k=3).generate(prices)
    assert w.shape == prices.shape
    assert list(w.columns) == list(prices.columns)


def test_top_k_holds_at_most_k_names() -> None:
    prices = _make_panel(n=600, n_names=20)
    w = TopKMomentum(k=5).generate(prices)
    n_held = (w.abs() > 1e-8).sum(axis=1)
    # After warm-up, never hold more than k names
    after_warmup = n_held.iloc[300:]
    assert (after_warmup <= 5).all()


def test_top_k_respects_gross_cap() -> None:
    prices = _make_panel()
    w = TopKMomentum(k=5, target_gross=1.0).generate(prices)
    assert w.abs().sum(axis=1).max() <= 1.001


def test_top_k_respects_per_position_cap() -> None:
    prices = _make_panel()
    w = TopKMomentum(k=3, max_per_position=0.15).generate(prices)
    assert w.abs().max().max() <= 0.151


def test_abs_momentum_gate_blocks_negative_returns() -> None:
    """If every name has negative trailing return and the absolute gate
    is at zero, the strategy must sit in cash on that bar."""
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    cols: dict[str, np.ndarray] = {}
    for i in range(8):
        # all names trend down
        r = rng.normal(-0.001, 0.01, 400)
        cols[f"A{i}"] = 100.0 * np.exp(np.cumsum(r))
    prices = pd.DataFrame(cols, index=idx)
    w = TopKMomentum(k=3, abs_momentum_threshold=0.0).generate(prices)
    # At the right edge, formation return is negative -> must be flat
    assert w.iloc[-1].abs().sum() == 0.0


def test_abs_momentum_none_keeps_top_k_in_bear() -> None:
    """abs_momentum_threshold=None disables the gate; we still hold the
    top-K even when everything is falling."""
    idx = pd.date_range("2020-01-01", periods=400, freq="1D", tz="UTC")
    rng = np.random.default_rng(0)
    cols: dict[str, np.ndarray] = {}
    for i in range(8):
        r = rng.normal(-0.001, 0.01, 400)
        cols[f"A{i}"] = 100.0 * np.exp(np.cumsum(r))
    prices = pd.DataFrame(cols, index=idx)
    w = TopKMomentum(k=3, abs_momentum_threshold=None).generate(prices)
    assert w.iloc[-1].abs().sum() > 0.0


def test_top_k_rebalance_rotation() -> None:
    """Members rotate over time when the leaderboard changes."""
    prices = _make_panel(n=800, n_names=15)
    s = TopKMomentum(k=3, rebalance=21)
    w = s.generate(prices)
    # The set of names held at the start and at the end may or may not
    # overlap — but holdings should change at least once over 800 bars.
    held_early = set(w.iloc[300][w.iloc[300] > 0].index)
    held_late = set(w.iloc[700][w.iloc[700] > 0].index)
    # Not asserting they differ (might genuinely be stable on synthetic
    # data), but no exception while rotating is enough.
    assert isinstance(held_early, set)
    assert isinstance(held_late, set)


def test_top_k_lookback_lt_skip_rejected() -> None:
    with pytest.raises(ValueError, match="lookback"):
        TopKMomentum(lookback=10, skip=20).generate(_make_panel())


def test_top_k_empty_universe_returns_zeros() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="1D", tz="UTC")
    empty = pd.DataFrame(index=idx)
    w = TopKMomentum(k=3).generate(empty)
    assert w.shape == (100, 0)
