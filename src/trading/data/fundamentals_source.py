"""Fundamentals adapter — yfinance ``Ticker.info`` with a Parquet cache.

This is the *cheap* fundamentals source: free, ad-hoc, and patchy. Use it
to bootstrap the quality screen; swap in a paid feed (Tiingo $10/mo,
Polygon $30/mo, IEX) when you need reliable coverage on small caps.

Cache layout
------------
One Parquet under ``<data_dir>/fundamentals.parquet`` with one row per
symbol and columns matching ``Fundamentals`` model fields. Refresh weekly;
the runner doesn't need real-time fundamentals.

Why a single file, not per-symbol like the bar cache: the row count is
small (~1000s) and a single file rewrites atomically, which we want when
the refresh job runs.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading.core.logging import logger
from trading.selection.screens import Fundamentals

_REQUIRED_FIELDS = (
    "symbol",
    "sector",
    "industry",
    "marketCap",
    "returnOnEquity",
    "debtToEquity",
    "profitMargins",
    "freeCashflow",
)


def fetch_fundamentals_yf(
    symbols: list[str],
    *,
    downloader: Any | None = None,
    pause_seconds: float = 0.2,
) -> dict[str, Fundamentals]:
    """Pull fundamentals from yfinance one symbol at a time.

    ``downloader`` is yfinance-shaped (anything with ``Ticker(sym).info``).
    Tests inject a fake; production calls ``yfinance`` lazily. ``pause_seconds``
    spaces requests to keep yfinance from rate-limiting us on big universes.
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf

    out: dict[str, Fundamentals] = {}
    for sym in symbols:
        try:
            info: dict[str, Any] = downloader.Ticker(sym).info or {}
        except Exception as e:
            logger.bind(symbol=sym).warning(f"fundamentals fetch failed: {e!r}")
            continue
        # Map yfinance's camelCase / mixed-case keys to our snake_case model.
        de = info.get("debtToEquity")
        # yfinance returns debt-to-equity as a percent (e.g. 137.4 = 137%).
        if de is not None:
            try:
                de = float(de) / 100.0
            except (TypeError, ValueError):
                de = None
        fcf = info.get("freeCashflow")
        out[sym] = Fundamentals(
            symbol=sym,
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=_safe_float(info.get("marketCap")),
            roe=_safe_float(info.get("returnOnEquity")),
            debt_to_equity=de,
            profit_margin=_safe_float(info.get("profitMargins")),
            free_cash_flow_positive=(
                None if fcf is None else _safe_float(fcf) is not None and _safe_float(fcf) > 0
            ),
        )
        time.sleep(pause_seconds)
    return out


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


# -------------------------------------------------------------- cache I/O ----


def write_fundamentals_cache(
    path: Path,
    fundamentals: dict[str, Fundamentals],
) -> None:
    """Overwrite the Parquet cache with one row per symbol."""
    if not fundamentals:
        return
    rows = [f.model_dump() for f in fundamentals.values()]
    df = pd.DataFrame(rows)
    df["_refreshed_at"] = datetime.now(tz=timezone.utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def read_fundamentals_cache(path: Path) -> dict[str, Fundamentals]:
    """Read the Parquet cache back into a {symbol: Fundamentals} dict.
    Returns ``{}`` if the file doesn't exist — the screen treats that as
    'no fundamentals available' and skips the quality + sector-momentum
    checks gracefully."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: dict[str, Fundamentals] = {}
    for _, row in df.iterrows():
        kwargs = {k: v for k, v in row.items() if k != "_refreshed_at"}
        # Pandas turns missing values into pd.NA / NaN; pydantic wants None.
        kwargs = {k: (None if pd.isna(v) else v) for k, v in kwargs.items()}
        try:
            f = Fundamentals(**kwargs)
        except Exception as e:
            logger.bind(symbol=kwargs.get("symbol")).warning(
                f"skipping malformed fundamentals row: {e!r}"
            )
            continue
        out[f.symbol] = f
    return out
