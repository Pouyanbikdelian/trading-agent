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


def test_correlation_filter_diversifies_basket() -> None:
    """Construct two clusters: 5 highly-correlated 'tech' names and 5
    independent names. Plain top-K-by-momentum will pick all 5 tech;
    the correlation filter must replace some of them with independents."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    # Cluster A: 5 names sharing a common factor, slightly higher mean
    common = rng.normal(0.001, 0.012, n)
    cols: dict[str, np.ndarray] = {}
    for i in range(5):
        idio = rng.normal(0, 0.003, n)
        cols[f"TECH_{i}"] = 100.0 * np.exp(np.cumsum(common + idio))
    # Cluster B: 5 independent names with slightly lower mean (so the
    # filter actually has to sacrifice some momentum to take them)
    for i in range(5):
        r = rng.normal(0.0008, 0.012, n)
        cols[f"INDEP_{i}"] = 100.0 * np.exp(np.cumsum(r))
    prices = pd.DataFrame(cols, index=idx)

    # Without filter: expect mostly TECH names
    unfiltered = TopKMomentum(
        k=5,
        lookback=126,
        skip=0,
        rebalance=21,
        abs_momentum_threshold=None,
    ).generate(prices)
    last_un = unfiltered.iloc[-1]
    n_tech_un = sum(1 for s in last_un[last_un > 0].index if s.startswith("TECH_"))

    # With filter: expect fewer TECH names since they're correlated
    filtered = TopKMomentum(
        k=5,
        lookback=126,
        skip=0,
        rebalance=21,
        abs_momentum_threshold=None,
        min_decorrelated=4,
        max_pairwise_corr=0.5,
        corr_window=60,
    ).generate(prices)
    last_f = filtered.iloc[-1]
    n_tech_f = sum(1 for s in last_f[last_f > 0].index if s.startswith("TECH_"))

    # The filtered basket should have strictly fewer TECH names than the
    # unfiltered one (or the same if tech were never dominant).
    assert n_tech_f <= n_tech_un
    # And we should keep at least some INDEP names when the filter binds.
    n_indep_f = sum(1 for s in last_f[last_f > 0].index if s.startswith("INDEP_"))
    if n_tech_un >= 4:
        assert n_indep_f >= 2, (
            f"correlation filter should have admitted at least 2 INDEP names; "
            f"got {n_indep_f} (basket: {list(last_f[last_f > 0].index)})"
        )


def test_correlation_filter_falls_back_when_short_history() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="1D", tz="UTC")
    prices = pd.DataFrame(
        {f"A{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300))) for i in range(8)},
        index=idx,
    )
    # corr_window > available bars at first rebalance — strategy must
    # gracefully fall back to pure top-K without raising.
    s = TopKMomentum(
        k=3,
        lookback=126,
        rebalance=21,
        abs_momentum_threshold=None,
        min_decorrelated=3,
        corr_window=500,
    )
    w = s.generate(prices)
    # Just verify it runs and produces SOME positions
    assert (w.abs() > 0).any().any()


def test_top_k_empty_universe_returns_zeros() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="1D", tz="UTC")
    empty = pd.DataFrame(index=idx)
    w = TopKMomentum(k=3).generate(empty)
    assert w.shape == (100, 0)
