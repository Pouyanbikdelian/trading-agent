"""ParquetCache tests — round-trip, partial fetch, force-refresh.

No network: a stub DataSource serves slices of a synthetic frame so we can
verify the cache only asks for the prefix/suffix it's missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument
from trading.data.base import BAR_COLUMNS, Frequency
from trading.data.cache import ParquetCache, _merge, _slice_inclusive
from tests.data.conftest import make_bars


class _StubSource:
    """In-memory DataSource that records every call."""

    name = "stub"

    def __init__(self, full: pd.DataFrame) -> None:
        self._full = full
        self.calls: list[tuple[datetime, datetime, str]] = []

    def get_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
    ) -> pd.DataFrame:
        self.calls.append((start, end, freq))
        mask = (self._full.index >= start) & (self._full.index <= end)
        return self._full.loc[mask].copy()


@pytest.fixture
def stub_full() -> pd.DataFrame:
    return make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=60, freq="1D")


@pytest.fixture
def cache(tmp_path: Path) -> ParquetCache:
    return ParquetCache(tmp_path / "parquet")


def test_path_layout(cache: ParquetCache, aapl: Instrument) -> None:
    p = cache.path_for(aapl, "1D")
    parts = p.relative_to(cache.root).parts
    assert parts == ("equity", "AAPL", "1D.parquet")


def test_path_safe_symbol_for_slash(cache: ParquetCache, btc: Instrument) -> None:
    p = cache.path_for(btc, "1h")
    assert p.name == "1h.parquet"
    assert "BTC_USDT" in p.parts


def test_read_missing_returns_empty(cache: ParquetCache, aapl: Instrument) -> None:
    df = cache.read(aapl, "1D")
    assert df.empty
    assert list(df.columns) == list(BAR_COLUMNS)
    assert str(df.index.tz) == "UTC"


def test_write_then_read_roundtrip(
    cache: ParquetCache, aapl: Instrument, synthetic_daily: pd.DataFrame
) -> None:
    p = cache.write(aapl, "1D", synthetic_daily)
    assert p.exists()
    out = cache.read(aapl, "1D")
    # check_freq=False — parquet does not round-trip the index's freq
    # attribute, but the rows themselves must match exactly.
    pd.testing.assert_frame_equal(out, synthetic_daily, check_freq=False)


def test_get_bars_cold_fetches_full_range(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    source = _StubSource(stub_full)
    start = stub_full.index[5].to_pydatetime()
    end = stub_full.index[20].to_pydatetime()
    out = cache.get_bars(source, aapl, start, end, "1D")
    assert len(out) == 16  # inclusive
    assert len(source.calls) == 1  # one shot, no cache existed


def test_get_bars_warm_hits_cache_only(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    source = _StubSource(stub_full)
    start = stub_full.index[10].to_pydatetime()
    end = stub_full.index[30].to_pydatetime()
    cache.get_bars(source, aapl, start, end, "1D")  # cold
    source.calls.clear()
    # Same range — should hit the cache and not call source.
    out = cache.get_bars(source, aapl, start, end, "1D")
    assert source.calls == []
    assert len(out) == 21


def test_get_bars_extends_suffix_only(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    """Cache has [10..30], caller asks [10..50] → only suffix is fetched."""
    source = _StubSource(stub_full)
    s0 = stub_full.index[10].to_pydatetime()
    e0 = stub_full.index[30].to_pydatetime()
    cache.get_bars(source, aapl, s0, e0, "1D")
    source.calls.clear()

    s1 = stub_full.index[10].to_pydatetime()
    e1 = stub_full.index[50].to_pydatetime()
    out = cache.get_bars(source, aapl, s1, e1, "1D")
    assert len(source.calls) == 1
    call_start, call_end, _ = source.calls[0]
    assert call_start == e0   # suffix begins at cached_max
    assert call_end == e1
    assert len(out) == 41


def test_get_bars_extends_prefix_only(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    source = _StubSource(stub_full)
    s0 = stub_full.index[20].to_pydatetime()
    e0 = stub_full.index[40].to_pydatetime()
    cache.get_bars(source, aapl, s0, e0, "1D")
    source.calls.clear()

    s1 = stub_full.index[5].to_pydatetime()
    out = cache.get_bars(source, aapl, s1, e0, "1D")
    assert len(source.calls) == 1
    call_start, call_end, _ = source.calls[0]
    assert call_start == s1
    assert call_end == s0     # prefix ends at cached_min
    assert len(out) == 36


def test_force_refresh_refetches_full_range(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    source = _StubSource(stub_full)
    start = stub_full.index[10].to_pydatetime()
    end = stub_full.index[20].to_pydatetime()
    cache.get_bars(source, aapl, start, end, "1D")
    source.calls.clear()

    cache.get_bars(source, aapl, start, end, "1D", force_refresh=True)
    assert len(source.calls) == 1
    assert source.calls[0][0] == start
    assert source.calls[0][1] == end


def test_naive_datetime_rejected(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    source = _StubSource(stub_full)
    naive = datetime(2024, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware"):
        cache.get_bars(source, aapl, naive, naive, "1D")


def test_inverted_range_rejected(
    cache: ParquetCache, aapl: Instrument, stub_full: pd.DataFrame
) -> None:
    source = _StubSource(stub_full)
    a = datetime(2024, 2, 1, tzinfo=timezone.utc)
    b = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="after end"):
        cache.get_bars(source, aapl, a, b, "1D")


def test_merge_dedupes_keep_last() -> None:
    a = make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3, seed=1)
    b = make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3, seed=2)
    merged = _merge(a, b)
    # All three timestamps overlap; the later frame (b) wins.
    # check_freq=False — concat drops the index's freq attribute.
    pd.testing.assert_frame_equal(merged, b, check_freq=False)


def test_slice_inclusive_bounds(synthetic_daily: pd.DataFrame) -> None:
    start = synthetic_daily.index[5].to_pydatetime()
    end = synthetic_daily.index[10].to_pydatetime()
    out = _slice_inclusive(synthetic_daily, start, end)
    assert out.index[0] == synthetic_daily.index[5]
    assert out.index[-1] == synthetic_daily.index[10]
    assert len(out) == 6
