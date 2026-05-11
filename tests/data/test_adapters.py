"""Adapter tests with mocked vendor clients (no network, no broker).

Each adapter accepts a dependency-injected client/downloader for exactly this
reason — we exercise the normalization logic (column rename, tz handling,
pagination loop, freq translation) without importing yfinance / ccxt / ib_async.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument
from trading.data.base import BAR_COLUMNS, validate_bars_frame
from trading.data.ccxt_source import CcxtSource
from trading.data.yfinance_source import YFinanceSource
from tests.data.conftest import make_bars


# ---------------------------------------------------------------- yfinance ----


class _FakeYf:
    """Mimics the bits of the yfinance module the adapter touches."""

    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw = raw
        self.last_kwargs: dict[str, Any] = {}

    def download(self, **kwargs: Any) -> pd.DataFrame:  # noqa: D401
        self.last_kwargs = kwargs
        return self.raw


def _yf_style_frame(periods: int = 5, tz: str | None = None) -> pd.DataFrame:
    """Build a frame in yfinance's raw shape: title-case columns + 'Adj Close'."""
    idx = pd.date_range("2024-01-01", periods=periods, freq="1D", tz=tz)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 104, periods),
            "High": np.linspace(101, 105, periods),
            "Low": np.linspace(99, 103, periods),
            "Close": np.linspace(100.5, 104.5, periods),
            "Adj Close": np.linspace(100.5, 104.5, periods),
            "Volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


def test_yfinance_renames_and_localizes_naive_index(aapl: Instrument) -> None:
    fake = _FakeYf(_yf_style_frame())
    src = YFinanceSource(downloader=fake)
    df = src.get_bars(
        aapl,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 5, tzinfo=timezone.utc),
        "1D",
    )
    assert list(df.columns) == list(BAR_COLUMNS)
    assert str(df.index.tz) == "UTC"
    assert fake.last_kwargs["interval"] == "1d"
    assert fake.last_kwargs["auto_adjust"] is False


def test_yfinance_handles_multiindex_columns(aapl: Instrument) -> None:
    raw = _yf_style_frame()
    # Recent yfinance returns a 2-level column index even for a single ticker.
    raw.columns = pd.MultiIndex.from_tuples([(c, aapl.symbol) for c in raw.columns])
    fake = _FakeYf(raw)
    src = YFinanceSource(downloader=fake)
    df = src.get_bars(
        aapl,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 5, tzinfo=timezone.utc),
        "1D",
    )
    assert list(df.columns) == list(BAR_COLUMNS)
    assert not df.empty


def test_yfinance_empty_returns_canonical_empty(aapl: Instrument) -> None:
    fake = _FakeYf(pd.DataFrame())
    src = YFinanceSource(downloader=fake)
    df = src.get_bars(
        aapl,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 5, tzinfo=timezone.utc),
        "1D",
    )
    assert df.empty
    assert list(df.columns) == list(BAR_COLUMNS)


def test_yfinance_rejects_4h(aapl: Instrument) -> None:
    src = YFinanceSource(downloader=_FakeYf(_yf_style_frame()))
    with pytest.raises(ValueError, match="4h"):
        src.get_bars(
            aapl,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 5, tzinfo=timezone.utc),
            "4h",
        )


def test_yfinance_passes_through_intraday_tz(aapl: Instrument) -> None:
    raw = _yf_style_frame(tz="America/New_York")
    fake = _FakeYf(raw)
    src = YFinanceSource(downloader=fake)
    df = src.get_bars(
        aapl,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 5, tzinfo=timezone.utc),
        "1h",
    )
    assert str(df.index.tz) == "UTC"
    assert fake.last_kwargs["interval"] == "1h"


# -------------------------------------------------------------------- ccxt ----


class _FakeCcxt:
    """Mimics the ``fetch_ohlcv`` method of a ccxt exchange."""

    rateLimit = 0  # adapter sleeps for client.rateLimit / 1000s — keep it 0

    def __init__(self, rows: list[list[float]], page_size: int = 1000) -> None:
        self._rows = rows
        self._page_size = page_size
        self.calls: list[dict[str, Any]] = []

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> list[list[float]]:
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "since": since, "limit": limit})
        page = [r for r in self._rows if r[0] >= since][: self._page_size]
        return page


def _ohlcv_rows(periods: int, step_ms: int, start_ms: int) -> list[list[float]]:
    return [
        [start_ms + i * step_ms, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0]
        for i in range(periods)
    ]


def test_ccxt_paginates_until_end(btc: Instrument) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 6, tzinfo=timezone.utc)   # 5 daily bars wanted
    start_ms = int(start.timestamp() * 1000)
    step_ms = 24 * 3600 * 1000
    rows = _ohlcv_rows(periods=5, step_ms=step_ms, start_ms=start_ms)

    fake = _FakeCcxt(rows, page_size=2)
    src = CcxtSource(exchange_id="binance", client=fake)
    df = src.get_bars(btc, start, end, "1D")

    assert len(df) == 5
    assert df["adj_close"].isna().all()
    assert str(df.index.tz) == "UTC"
    # 3 pages of 2/2/1 → 3 calls (then the 4th returns empty and breaks).
    assert len(fake.calls) >= 3
    assert fake.calls[0]["timeframe"] == "1d"


def test_ccxt_handles_empty(btc: Instrument) -> None:
    fake = _FakeCcxt([], page_size=1000)
    src = CcxtSource(client=fake)
    df = src.get_bars(
        btc,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 5, tzinfo=timezone.utc),
        "1h",
    )
    assert df.empty
    assert list(df.columns) == list(BAR_COLUMNS)


def test_ccxt_breaks_on_non_advancing_page(btc: Instrument) -> None:
    """If the exchange returns a page whose last ts <= since, we must NOT loop forever."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 10, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    # Single row whose timestamp equals `since` — the loop should break.
    fake = _FakeCcxt([[start_ms, 1, 1, 1, 1, 1]], page_size=10)
    src = CcxtSource(client=fake)
    df = src.get_bars(btc, start, end, "1D")
    # First call returns one row; second call would return same row → break.
    assert len(fake.calls) <= 2
    assert len(df) <= 1


def test_ccxt_trims_to_end(btc: Instrument) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)   # ask for 3 days
    start_ms = int(start.timestamp() * 1000)
    step_ms = 24 * 3600 * 1000
    # Exchange returns 10 rows; only 3 are within the inclusive end.
    rows = _ohlcv_rows(periods=10, step_ms=step_ms, start_ms=start_ms)
    fake = _FakeCcxt(rows, page_size=10)
    src = CcxtSource(client=fake)
    df = src.get_bars(btc, start, end, "1D")
    assert len(df) == 3
    assert df.index[-1] <= pd.Timestamp(end)


# --------------------------------------------------------------- universes ----


def test_validate_normalizes_an_adapter_frame() -> None:
    """End-to-end sanity: any frame an adapter could produce passes validation."""
    base = make_bars(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=10)
    out = validate_bars_frame(base.copy())
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_instrument_is_immutable() -> None:
    inst = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    with pytest.raises(Exception):
        inst.symbol = "MSFT"  # type: ignore[misc]
