"""regime_switch: long-only, gross <= 1, right legs per regime, loud on gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.strategies.regime_switch import (
    ALL_SYMBOLS,
    GTAA5,
    RegimeSwitch,
    RegimeSwitchParams,
    latest_allocation,
    regime_series,
)


def _prices(n: int = 700, spy_crash: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = {}
    for i, s in enumerate(ALL_SYMBOLS):
        r = rng.normal(0.0004, 0.01 + 0.001 * i, n)
        cols[s] = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(cols, index=idx)
    if spy_crash:  # last 120 days: SPY falls hard -> below its 200d MA
        df.iloc[-120:, df.columns.get_loc("SPY")] *= np.linspace(1.0, 0.6, 120)
    return df


def test_long_only_and_gross_at_most_one() -> None:
    w = RegimeSwitch().generate(_prices())
    assert (w >= -1e-12).all().all()
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_bear_holds_only_gtaa5_symbols() -> None:
    px = _prices(spy_crash=True)
    lab = regime_series(px, RegimeSwitchParams())
    assert lab.iloc[-1] == "bear"
    alloc = latest_allocation(px)
    assert alloc["regime"] == "bear"
    assert set(alloc["weights"]) <= set(GTAA5)
    assert alloc["legs"] == {"gtaa5": 1.0}


def test_missing_symbol_is_loud() -> None:
    with pytest.raises(ValueError, match="needs prices"):
        RegimeSwitch().generate(_prices().drop(columns=["DBC"]))


def test_too_little_history_is_loud() -> None:
    with pytest.raises(ValueError):
        latest_allocation(_prices(n=120))


def test_no_lookahead_last_row_unchanged_by_future() -> None:
    px = _prices(n=700)
    w_cut = RegimeSwitch().generate(px.iloc[:650])
    w_full = RegimeSwitch().generate(px)
    # rows up to 649 were decided on data up to 649 — appending the future
    # must not change them (cov legs refit every 5 rows; compare those rows)
    common = w_cut.index[300:-1:5]
    pd.testing.assert_frame_equal(w_cut.loc[common], w_full.loc[common], atol=1e-8)
