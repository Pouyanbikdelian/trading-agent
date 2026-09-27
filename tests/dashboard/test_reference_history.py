"""Why the live SPY and rotation panels were empty (2026-09-27), and the fix.

SPY and the sector ETFs are in no strategy universe, so the daily price
refresh fetched them only as scorecard subjects, starting 30 days back.
The 200-day line needs 210 closes and the rotation graph ~15 months, so
both said "not cached". The refresh now keeps them three years deep; the
dashboard prefers the fresher/longer SPY copy and retries an empty
rotation after minutes, not hours.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from trading.core.types import AssetClass, Instrument
from trading.dashboard import cockpit
from trading.data.cache import ParquetCache
from trading.runner.runner import REFERENCE_LOOKBACK_DAYS, _add_reference_targets
from trading.runtime.news_watch import SECTOR_ETFS


def test_refresh_keeps_spy_and_sector_etfs_three_years_deep() -> None:
    end = datetime(2026, 9, 27, tzinfo=timezone.utc)
    early = end - timedelta(days=4000)
    symbols = {"AAPL"}
    starts = {"AAPL": end - timedelta(days=30), "XLK": early}

    _add_reference_targets(symbols, starts, end=end)

    want = {"SPY", *SECTOR_ETFS, "QQQ", "TLT", "HYG", "GLD"}
    assert want <= symbols and "AAPL" in symbols
    assert starts["SPY"] == end - timedelta(days=REFERENCE_LOOKBACK_DAYS)
    assert starts["XLK"] == early  # an earlier request is never shortened
    assert starts["AAPL"] == end - timedelta(days=30)


def _bars(n: int, end: str) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=n, freq="B", tz="UTC")
    px = [400 + i * 0.5 for i in range(n)]
    df = pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px, "volume": 1.0, "adj_close": px}, index=idx
    )
    df.index.name = "ts"
    return df


def test_market_block_prefers_the_fresher_longer_spy(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    # A month-short copy under etf/ used to win just by being checked first.
    cache.write(Instrument(symbol="SPY", asset_class=AssetClass.ETF), "1D", _bars(22, "2026-09-25"))
    cache.write(
        Instrument(symbol="SPY", asset_class=AssetClass.EQUITY), "1D", _bars(600, "2026-09-25")
    )

    m = cockpit.market_block(tmp_path, {"latest": {}})

    assert len(m["spy_series"]) == 260
    assert m["spy_vs_200d"] is not None


def test_an_empty_rotation_is_retried_within_minutes(tmp_path, monkeypatch) -> None:
    from trading.dashboard import rotation

    calls = 0

    def empty(*args, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(), {}

    clock = [1_000_000.0]
    monkeypatch.setattr(rotation, "_cache", {"t": 0.0, "payload": None, "key": None})
    monkeypatch.setattr(rotation, "_load_history", empty)
    monkeypatch.setattr(rotation.time, "time", lambda: clock[0])

    rotation.build_rotation(tmp_path / "s", tmp_path / "d")
    clock[0] += 120
    rotation.build_rotation(tmp_path / "s", tmp_path / "d")
    assert calls == 1  # still negative-cached for a couple of minutes
    clock[0] += rotation._EMPTY_TTL_S
    rotation.build_rotation(tmp_path / "s", tmp_path / "d")
    assert calls == 2  # but not for two hours


def test_money_map_sizes_come_from_cached_volume(tmp_path: Path) -> None:
    from trading.dashboard.rotation import _cached_dollar_volume

    df = _bars(100, "2026-09-25")
    df["volume"] = 1_000.0
    ParquetCache(tmp_path).write(Instrument(symbol="XLK", asset_class=AssetClass.ETF), "1D", df)

    dv = _cached_dollar_volume(tmp_path, "XLK")
    assert dv is not None and dv > 400_000
    assert _cached_dollar_volume(tmp_path, "NOPE") is None
