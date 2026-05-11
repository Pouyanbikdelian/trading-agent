"""Data layer: the ``DataSource`` Protocol, the Parquet cache, and adapters
for yfinance / ccxt / IBKR.

The contract every adapter satisfies (see :mod:`trading.data.base`):

* ``get_bars(instrument, start, end, freq) -> pd.DataFrame``
* index is a tz-aware ``DatetimeIndex`` named ``ts`` (UTC)
* columns are ``["open", "high", "low", "close", "volume", "adj_close"]``
* ``adj_close`` is NaN when the venue doesn't expose it (FX, crypto)
"""

from __future__ import annotations

from trading.data.base import (
    BAR_COLUMNS,
    CANONICAL_FREQUENCIES,
    DataSource,
    Frequency,
    empty_bars_frame,
    validate_bars_frame,
)
from trading.data.cache import ParquetCache

__all__ = [
    "BAR_COLUMNS",
    "CANONICAL_FREQUENCIES",
    "DataSource",
    "Frequency",
    "ParquetCache",
    "empty_bars_frame",
    "validate_bars_frame",
]
