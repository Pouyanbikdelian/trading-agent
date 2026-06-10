"""Portfolio beta + holdings correlation — hermetic over a tiny cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.runtime import portfolio_stats as ps


@pytest.fixture
def cache(tmp_path):
    """Synthetic parquet cache: SPY (market), HI (beta ~2), LO (beta ~0)."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-06-01", periods=400)
    mkt = rng.normal(0.0004, 0.01, len(idx))
    spy = 100 * np.exp(np.cumsum(mkt))
    hi = 50 * np.exp(np.cumsum(2.0 * mkt + rng.normal(0, 0.001, len(idx))))
    lo = 80 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(idx))))
    for sub, sym, px in (("etf", "SPY", spy), ("equity", "HI", hi), ("equity", "LO", lo)):
        d = tmp_path / sub / sym
        d.mkdir(parents=True)
        pd.DataFrame({"close": px}, index=idx).to_parquet(d / "1d.parquet")
    return tmp_path


def test_portfolio_beta_weighted(cache) -> None:
    result = ps.portfolio_beta({"HI": 50_000.0, "LO": 50_000.0}, cache)
    assert result is not None
    beta, used = result
    assert used == 2
    assert 0.8 < beta < 1.4  # ~ (2.0 + 0.0) / 2 with noise


def test_portfolio_beta_missing_market_or_names(cache, tmp_path) -> None:
    assert ps.portfolio_beta({"HI": 1.0}, tmp_path / "empty") is None
    assert ps.portfolio_beta({"UNKNOWN": 1.0}, cache) is None
    assert ps.portfolio_beta({}, cache) is None


def test_holdings_correlation_and_format(cache) -> None:
    corr = ps.holdings_correlation(["HI", "LO", "SPY"], cache)
    assert corr is not None and corr.shape == (3, 3)
    assert corr.loc["HI", "SPY"] > 0.9  # built from the same shocks
    text = ps.format_correlation(corr)
    assert "Holdings correlation" in text
    assert "Most correlated" in text
    # matrix renders for small books
    assert "```" in text


def test_correlation_needs_two_names(cache) -> None:
    assert ps.holdings_correlation(["HI"], cache) is None
    assert ps.holdings_correlation(["HI", "MISSING"], cache) is None
