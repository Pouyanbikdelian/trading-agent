"""ccxt adapter — public exchange OHLCV (default: Binance).

We use ccxt's *public* endpoints only — no API keys, no auth. That's enough
for research; live crypto execution can swap in an authed client later.

Pagination: exchanges cap a single fetch at ~500–1000 candles. We loop on
the ``since`` parameter until we've covered the requested range or the
exchange stops returning rows.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from trading.core.types import Instrument
from trading.data.base import BAR_COLUMNS, Frequency, empty_bars_frame, validate_bars_frame

if TYPE_CHECKING:  # pragma: no cover
    pass


# Canonical pandas alias -> ccxt timeframe.
_FREQ_TO_CCXT: dict[Frequency, str] = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "4h": "4h",
    "1D": "1d",
    "1W": "1w",
}

_DEFAULT_PAGE_LIMIT = 1000


class CcxtSource:
    """``DataSource`` adapter for ccxt-supported public exchanges."""

    name = "ccxt"

    def __init__(self, exchange_id: str = "binance", client: Any | None = None) -> None:
        self.exchange_id = exchange_id
        self._client = client

    def _exchange(self) -> Any:
        if self._client is None:
            import ccxt  # lazy import
            cls = getattr(ccxt, self.exchange_id)
            self._client = cls({"enableRateLimit": True})
        return self._client

    def get_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
    ) -> pd.DataFrame:
        if freq not in _FREQ_TO_CCXT:
            raise ValueError(f"ccxt does not support freq={freq!r}")
        timeframe = _FREQ_TO_CCXT[freq]

        client = self._exchange()
        since_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
        end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)

        rows: list[list[float]] = []
        while since_ms <= end_ms:
            batch = client.fetch_ohlcv(
                instrument.symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=_DEFAULT_PAGE_LIMIT,
            )
            if not batch:
                break
            rows.extend(batch)
            last_ts_ms = int(batch[-1][0])
            if last_ts_ms <= since_ms:
                # Exchange returned a page that doesn't advance — bail out
                # rather than loop forever.
                break
            since_ms = last_ts_ms + 1
            # ccxt respects ``enableRateLimit`` internally, but exchanges
            # sometimes still 418; tiny sleep is cheap insurance.
            time.sleep(getattr(client, "rateLimit", 0) / 1000.0)

        if not rows:
            return empty_bars_frame()

        df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.set_index("ts").drop(columns=["ts_ms"])
        df["adj_close"] = pd.NA
        # Slicing the inclusive end is the cache's job, but we trim here
        # too so callers using the adapter directly get what they asked for.
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        df = df[df.index <= end_ts]
        df = df.loc[:, list(BAR_COLUMNS)]
        return validate_bars_frame(df)
