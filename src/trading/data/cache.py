"""Parquet-backed read-through cache for bar data.

Layout::

    <data_dir>/<asset_class>/<symbol_safe>/<freq>.parquet

The cache is dumb on purpose: one file per (instrument, freq), full history
inside. Bars are unique per timestamp; on a partial overlap the cache fetches
only the missing prefix/suffix from the source and merges. We never re-fetch
ranges already on disk unless the caller explicitly passes ``force_refresh``.

This is appropriate because:

* Bar history is append-only for already-printed bars (the past doesn't
  change). Splits/dividends can shift adj_close — handle that with a
  full refresh, not by re-fetching the whole range every time.
* One file per (instrument, freq) keeps reads cheap and lets pyarrow's
  pushdown predicates do range filtering at scan time.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from trading.core.types import Instrument
from trading.data.base import (
    BAR_COLUMNS,
    DataSource,
    Frequency,
    empty_bars_frame,
    validate_bars_frame,
)


def _safe_symbol(symbol: str) -> str:
    """Make a symbol safe to use as a filesystem path component.

    ``BTC/USDT`` -> ``BTC_USDT``. Leaves everything else untouched.
    """
    return symbol.replace("/", "_").replace(":", "_")


class ParquetCache:
    """Read-through cache over a ``DataSource``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def path_for(self, instrument: Instrument, freq: Frequency) -> Path:
        return (
            self.root
            / instrument.asset_class.value
            / _safe_symbol(instrument.symbol)
            / f"{freq}.parquet"
        )

    # ------------------------------------------------------------------ io

    def read(self, instrument: Instrument, freq: Frequency) -> pd.DataFrame:
        """Read the full cached frame for an instrument/freq, or an empty
        frame matching the canonical schema if nothing is cached."""
        p = self.path_for(instrument, freq)
        if not p.exists():
            return empty_bars_frame()
        df = pd.read_parquet(p)
        # Parquet round-trips the index, but the tz attribute can come back
        # as a tzfile object rather than the original "UTC" string; force it.
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.tz_convert("UTC")
        return validate_bars_frame(df)

    def write(self, instrument: Instrument, freq: Frequency, df: pd.DataFrame) -> Path:
        """Overwrite the parquet for this instrument/freq with ``df``."""
        df = validate_bars_frame(df)
        p = self.path_for(instrument, freq)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, engine="pyarrow", index=True)
        return p

    # ----------------------------------------------------------- read-through

    def get_bars(
        self,
        source: DataSource,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return bars for ``[start, end]``, fetching only what's not cached.

        Strategy:
          1. If ``force_refresh`` or no cache exists: fetch the full range,
             merge with whatever was on disk (in case force_refresh narrows
             the request), write back.
          2. Otherwise read cached frame. Compute the missing prefix
             ``[start, cached_min)`` and missing suffix ``(cached_max, end]``,
             fetch each piece, concat, write back.
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware datetimes")
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        cached = self.read(instrument, freq)

        if force_refresh or cached.empty:
            fresh = source.get_bars(instrument, start, end, freq)
            fresh = validate_bars_frame(fresh)
            merged = _merge(cached, fresh)
            self.write(instrument, freq, merged)
            return _slice_inclusive(merged, start, end)

        cached_min = cached.index.min()
        cached_max = cached.index.max()
        pieces: list[pd.DataFrame] = [cached]

        if start < cached_min:
            prefix = source.get_bars(instrument, start, cached_min, freq)
            pieces.append(validate_bars_frame(prefix))

        if end > cached_max:
            suffix = source.get_bars(instrument, cached_max, end, freq)
            pieces.append(validate_bars_frame(suffix))

        merged = _merge(*pieces)
        if len(pieces) > 1:
            self.write(instrument, freq, merged)
        return _slice_inclusive(merged, start, end)


def _merge(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concat + dedupe (keep last) + sort by ts. Empty frames are tolerated."""
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return empty_bars_frame()
    merged = pd.concat(non_empty, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    merged.index.name = "ts"
    # Re-assert column order in case concat broke it.
    return merged.loc[:, list(BAR_COLUMNS)]


def _slice_inclusive(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Return rows where ``start <= ts <= end``. Slicing on a DatetimeIndex
    is inclusive on both ends in pandas, which matches what callers expect
    when they ask for a date range."""
    if df.empty:
        return df
    return df.loc[(df.index >= start) & (df.index <= end)]
