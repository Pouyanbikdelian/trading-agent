"""Reproduce the October 14 stale-constituent outage without a data vendor."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading.data.price_quality import align_current_prices

ASOF = datetime(2026, 10, 14, 21, tzinfo=timezone.utc)


def prices(end="2026-10-14", count=252):
    return pd.Series(
        np.linspace(80, 100, count), index=pd.date_range(end=end, periods=count, freq="B", tz="UTC")
    )


def test_stale_constituent_cannot_remove_new_dates():
    source = {f"LIVE{i}": prices() for i in range(8)}
    source["WBD"] = prices("2026-10-05")
    source["PSKY"] = prices("2026-10-13")
    wide, excluded = align_current_prices(source, asof=ASOF, freq="1D")
    assert wide.index[-1].date() == ASOF.date()
    assert len(wide) == 252
    assert set(excluded) == {"WBD", "PSKY"}
    assert "WBD" not in wide and "PSKY" not in wide


def test_ipo_does_not_destroy_mature_history():
    wide, excluded = align_current_prices(
        {"A": prices(), "B": prices(), "IPO": prices(count=26)}, asof=ASOF, freq="1D"
    )
    assert len(wide) == 252
    assert "short history" in excluded["IPO"]


@pytest.mark.parametrize("kind", ["stale", "missing"])
def test_broad_coverage_loss_blocks_instead_of_ranking_survivors(kind):
    source = {"A": prices(), "B": prices("2026-10-05")}
    expected = None
    if kind == "missing":
        source.pop("B")
        expected = 10
    with pytest.raises(ValueError, match="price coverage blocked"):
        align_current_prices(source, asof=ASOF, freq="1D", expected_count=expected)


def test_whole_universe_stale_cannot_be_called_current():
    with pytest.raises(ValueError, match="newest daily bar"):
        align_current_prices({"A": prices("2026-10-05")}, asof=ASOF, freq="1D")


def test_weekend_is_not_a_data_outage():
    wide, _ = align_current_prices(
        {"A": prices("2026-10-09")},
        asof=datetime(2026, 10, 12, 8, tzinfo=timezone.utc),
        freq="1D",
    )
    assert str(wide.index[-1].date()) == "2026-10-09"


def test_future_and_invalid_prices_cannot_establish_freshness():
    p = prices()
    p.loc[p.index[-1]] = np.inf
    p.loc[pd.Timestamp("2026-10-16", tz="UTC")] = 200.0
    wide, _ = align_current_prices({"A": p}, asof=ASOF, freq="1D")
    assert str(wide.index[-1].date()) == "2026-10-13"
    assert np.isfinite(wide.values).all()


def test_interior_holes_do_not_cause_a_stale_terminal_bar():
    a = prices()
    a.iloc[100] = np.nan
    wide, _ = align_current_prices({"A": a, "B": prices()}, asof=ASOF, freq="1D")
    assert len(wide) == 251
    assert str(wide.index[-1].date()) == "2026-10-14"
