"""Keep an obsolete constituent from moving the whole desk back in time."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd

from trading.data.base import Frequency


def align_current_prices(
    series: Mapping[str, pd.Series],
    *,
    asof: datetime,
    freq: Frequency,
    expected_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Exclude incomplete series before intersecting dates; never invent prices.

    October 2026: WBD stopped printing on October 5, but had enough history
    to pass the IPO filter. A row-wise inner join silently removed nine days
    of fresh prices from every other name. Require the newest observed bar
    in every retained series and refuse a broad coverage loss. Four calendar
    days tolerate weekends/holidays without blessing a week-old daily tape.
    """
    instant = pd.Timestamp(asof)
    if instant.tzinfo is None:
        raise ValueError("price alignment requires an aware asof")
    clean: dict[str, pd.Series] = {}
    excluded: dict[str, str] = {}
    for symbol, values in series.items():
        if not isinstance(values.index, pd.DatetimeIndex) or values.index.tz is None:
            excluded[symbol] = "invalid timestamp index"
            continue
        values = pd.to_numeric(values, errors="coerce").sort_index()
        values = values.loc[(values.index <= instant) & np.isfinite(values) & (values > 0)]
        values = values.loc[~values.index.duplicated(keep="last")]
        if values.empty:
            excluded[symbol] = "no valid prices"
        else:
            clean[symbol] = values
    if not clean:
        raise ValueError("price coverage blocked: no valid prices")

    counts = pd.Series({s: len(v) for s, v in clean.items()})
    required = max(int(counts.median() * 0.8), min(60, int(counts.max())))
    short_count = 0
    for symbol in list(clean):
        if len(clean[symbol]) < required:
            excluded[symbol] = f"short history (<{required} bars)"
            short_count += 1
            del clean[symbol]
    if not clean:
        raise ValueError("price coverage blocked: insufficient history")

    newest = max(v.index[-1] for v in clean.values())
    if freq == "1D" and (instant.normalize() - newest.normalize()).days > 4:
        raise ValueError(f"price coverage blocked: newest daily bar is {newest.date()}")
    eligible = max(len(series), expected_count or 0) - short_count
    for symbol in list(clean):
        if clean[symbol].index[-1] != newest:
            excluded[symbol] = (
                f"stale tail {clean[symbol].index[-1].date()} (latest {newest.date()})"
            )
            del clean[symbol]
    if len(clean) < eligible * 0.8:
        raise ValueError(
            f"price coverage blocked: only {len(clean)}/{eligible} mature series "
            f"reach {newest.date()}"
        )
    wide = pd.DataFrame(clean).sort_index().dropna(how="any")
    if len(wide) < required or wide.index[-1] != newest:
        raise ValueError("price coverage blocked: insufficient aligned history")
    return wide, excluded
