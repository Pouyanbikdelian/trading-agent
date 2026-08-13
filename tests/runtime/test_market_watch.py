"""Market watch — hermetic over a synthetic cache; network path mocked."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading.runtime import market_watch as mw


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2024-01-01", periods=400)

    def write(sub: str, sym: str, px: np.ndarray) -> None:
        d = tmp_path / "data" / sub / sym
        d.mkdir(parents=True)
        pd.DataFrame({"close": px}, index=idx).to_parquet(d / "1d.parquet")

    up = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(idx))))
    down = 100 * np.exp(np.cumsum(rng.normal(-0.001, 0.01, len(idx))))
    for i in range(60):
        write("equity", f"U{i:03d}", up * (1 + 0.001 * i))
    for i in range(20):
        write("equity", f"D{i:03d}", down * (1 + 0.001 * i))
    for sym in ("SPY", "TLT", "QQQ", "GLD", "DBC", "HYG", "IEF"):
        write("etf", sym, up)
    return tmp_path / "data"


def test_breadth_counts_trenders(cache) -> None:
    b = mw.compute_breadth(cache)
    # 60 uptrending vs 20 downtrending names -> breadth ~75%.
    assert 0.6 < b["pct_above_50"] <= 1.0
    assert 0.6 < b["pct_above_200"] <= 1.0


def test_breadth_none_when_too_few_names(tmp_path) -> None:
    out = mw.compute_breadth(tmp_path / "empty")
    assert out == {"pct_above_50": None, "pct_above_200": None}


def test_ratios_normalized_to_100_base(cache) -> None:
    r = mw.compute_ratios(cache)
    # Identical series -> ratio flat -> 100 on the 1y base.
    for k in ("spy_tlt", "qqq_spy", "gld_dbc", "hyg_ief"):
        assert r[k] == pytest.approx(100.0, abs=0.5)


def test_collect_appends_and_is_idempotent_per_day(tmp_path, cache, monkeypatch) -> None:
    monkeypatch.setattr(
        mw,
        "fetch_rates_vix",
        lambda: {"y_3m": 4.10, "y_10y": 4.42, "vix": 15.0, "vix3m": 17.0},
    )
    state = tmp_path / "state"
    r1 = mw.collect(state, cache)
    assert r1["curve_10y3m"] == pytest.approx(0.32)
    assert r1["vix_ratio"] == pytest.approx(15.0 / 17.0, abs=1e-3)

    # Second run the same day REPLACES, doesn't duplicate.
    mw.collect(state, cache)
    payload = json.loads((state / "market_watch.json").read_text())
    assert len(payload["history"]) == 1
    assert payload["latest"]["pct_above_50"] is not None


def test_startup_catchup_only_runs_after_a_missed_weekday_slot(tmp_path) -> None:
    state = tmp_path / "state"
    late = datetime(2026, 8, 12, 20, 21, tzinfo=timezone.utc)  # Wednesday
    assert mw.needs_startup_catchup(state, now=late) is True

    (state / "market_watch.json").parent.mkdir(parents=True)
    (state / "market_watch.json").write_text(
        json.dumps({"latest": {"t": "2026-08-12T20:20:00+00:00"}})
    )
    assert mw.needs_startup_catchup(state, now=late) is False
    assert (
        mw.needs_startup_catchup(state, now=datetime(2026, 8, 12, 20, 20, tzinfo=timezone.utc))
        is False
    )
    (state / "market_watch.json").unlink()
    assert (
        mw.needs_startup_catchup(state, now=datetime(2026, 8, 12, 20, 20, 1, tzinfo=timezone.utc))
        is True
    )
    assert (
        mw.needs_startup_catchup(state, now=datetime(2026, 8, 15, 22, 0, tzinfo=timezone.utc))
        is False
    )
