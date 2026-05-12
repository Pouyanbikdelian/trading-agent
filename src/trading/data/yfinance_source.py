"""yfinance adapter — daily and intraday US equities / ETFs.

yfinance is free and unauthenticated; we use it for the research universe.
It is **not** suitable for execution-time prices (rate limits, occasional
silent drops). For paper/live trading we read prices from IBKR.

Caveats baked in:
* Intraday history is limited (~60 days for 1h, ~7 days for 1m).
* ``adj_close`` is populated only for equities/ETFs (yfinance handles the
  split/dividend adjustment for us when ``auto_adjust=False``).
* Volume is reported in shares.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from trading.core.types import Instrument
from trading.data.base import BAR_COLUMNS, Frequency, empty_bars_frame, validate_bars_frame

if TYPE_CHECKING:  # pragma: no cover
    import yfinance as _yf  # noqa: F401


# Canonical pandas alias -> yfinance ``interval`` string.
_FREQ_TO_YF: dict[Frequency, str] = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "4h": "1h",  # yfinance has no 4h; caller must resample.
    "1D": "1d",
    "1W": "1wk",
}


class YFinanceSource:
    """``DataSource`` adapter for the yfinance library."""

    name = "yfinance"

    def __init__(self, downloader: object | None = None) -> None:
        # The downloader is dependency-injected so tests don't import yfinance.
        # In production we lazy-import the real one on first use.
        self._downloader = downloader

    def _yf(self) -> object:
        if self._downloader is None:
            import yfinance as yf  # local import keeps cold-start cost off importing trading

            self._downloader = yf
        return self._downloader

    def get_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
    ) -> pd.DataFrame:
        if freq not in _FREQ_TO_YF:
            raise ValueError(f"yfinance does not support freq={freq!r}")
        if freq == "4h":
            raise ValueError("yfinance has no native 4h bars; resample 1h upstream.")

        interval = _FREQ_TO_YF[freq]
        yf = self._yf()
        # yfinance.download is end-exclusive — bump by one bar so the caller's
        # inclusive [start, end] matches what we return.
        end_inclusive = end + pd.Timedelta(days=1) if freq in ("1D", "1W") else end

        raw = yf.download(  # type: ignore[attr-defined]
            tickers=instrument.symbol,
            start=start,
            end=end_inclusive,
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        if raw is None or len(raw) == 0:
            return empty_bars_frame()

        # yfinance returns a multi-index when given a list of tickers; for a
        # single ticker it usually returns flat columns, but recent versions
        # can still return MultiIndex. Normalize.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = raw.rename(columns=rename)
        for col in BAR_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        # yfinance returns naive timestamps for daily and tz-aware for
        # intraday in the exchange's local time. Force everything to UTC.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "ts"

        return validate_bars_frame(df)
