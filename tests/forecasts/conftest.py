"""A small synthetic market with one planted, learnable effect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SESSIONS = 820
SYMBOLS = [f"S{i:02d}" for i in range(30)]


def make_panel(seed: int = 7, sessions: int = SESSIONS, end: str = "2026-11-10"):
    """Closes for 30 stocks and SPY. Each stock has a persistent drift, so
    past winners keep winning: momentum genuinely predicts here."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=sessions, tz="UTC")
    market = rng.normal(0.0003, 0.008, sessions)
    drift = np.linspace(-0.0012, 0.0012, len(SYMBOLS))
    noise = rng.normal(0.0, 0.012, (sessions, len(SYMBOLS)))
    returns = market[:, None] + drift[None, :] + noise
    close = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=SYMBOLS)
    bench = pd.Series(400 * np.exp(np.cumsum(market)), index=index, name="SPY")
    return close, bench


@pytest.fixture
def panel():
    return make_panel()


@pytest.fixture
def data_dir(tmp_path, panel):
    """The same panel written the way the price cache stores it."""
    close, bench = panel
    root = tmp_path / "data"
    for symbol in close.columns:
        folder = root / "equity" / symbol
        folder.mkdir(parents=True)
        pd.DataFrame({"close": close[symbol], "adj_close": close[symbol]}).to_parquet(
            folder / "1D.parquet"
        )
    folder = root / "etf" / "SPY"
    folder.mkdir(parents=True)
    pd.DataFrame({"close": bench, "adj_close": bench}).to_parquet(folder / "1D.parquet")
    return root
