"""DataSource Protocol + canonical bar-frame schema.

Every adapter (yfinance, ccxt, ib-async) returns a DataFrame matching this
schema so downstream code (cache, backtester, strategies) never has to branch
on vendor.

Schema
------
* Index: ``DatetimeIndex`` named ``ts``, tz-aware (UTC).
* Columns (in order): ``open, high, low, close, volume, adj_close``.
* ``adj_close`` may be NaN where the venue doesn't expose split/div
  adjustments (FX, crypto).

Frequency
---------
We use pandas offset aliases as the canonical form (``"1D"``, ``"1H"``,
``"15min"``, ...). Each adapter translates these to its vendor's string.
Restricting to a known set means typos surface immediately instead of
silently fetching the wrong bar size.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from trading.core.types import Instrument

BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "adj_close")

# Canonical pandas offset aliases supported by the data layer. Adapters
# narrow this further (yfinance has no <1m; ccxt depends on exchange).
# Hour aliases are lowercase because pandas 2.2+ rejects "H" (use "h").
Frequency = Literal["1min", "5min", "15min", "30min", "1h", "4h", "1D", "1W"]

CANONICAL_FREQUENCIES: tuple[Frequency, ...] = (
    "1min",
    "5min",
    "15min",
    "30min",
    "1h",
    "4h",
    "1D",
    "1W",
)


@runtime_checkable
class DataSource(Protocol):
    """Anything that can return historical OHLCV bars for an instrument.

    Implementations must be safe to call repeatedly with the same arguments
    (idempotent) and must NOT mutate the instrument.
    """

    name: str

    def get_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
    ) -> pd.DataFrame: ...


def empty_bars_frame() -> pd.DataFrame:
    """Empty DataFrame with the canonical schema — useful as a fallback or
    when an adapter returns no rows."""
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)


def validate_bars_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assert + lightly normalize a bar frame to the canonical schema.

    Returns the frame (possibly with reordered columns and tz-converted
    index) so callers can chain. Raises ``ValueError`` on schema violations
    — these are bugs in the adapter, not recoverable input problems.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bar frame index must be a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("bar frame index must be timezone-aware (UTC)")
    # Normalize to UTC even if the adapter handed us a different tz.
    if str(df.index.tz) != "UTC":
        df = df.tz_convert("UTC")
    df.index.name = "ts"

    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bar frame missing required columns: {missing}")

    # Reorder + drop extra columns. Extra fields are stripped rather than
    # carried through, so the cache layer can rely on column ordering.
    df = df.loc[:, list(BAR_COLUMNS)]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # Bars should be unique per timestamp. Adapters that paginate may emit
    # boundary duplicates; we drop them here (keep first) rather than in
    # every adapter.
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    return df
