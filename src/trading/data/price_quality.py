"""Keep an obsolete constituent from moving the whole desk back in time."""

from __future__ import annotations

import math
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
    of fresh prices from every other name. Exclude series that stop short
    of the reference bar and refuse a broad coverage loss. Four calendar
    days tolerate weekends/holidays without blessing a week-old daily tape.

    The reference bar is the newest bar that at least 80% of the mature
    universe reaches — not the newest bar any single series happens to have
    (2026-11-10 review). The first version required every series to end on
    the single newest bar, which inverted the October failure: one name
    AHEAD of the rest made every other name "stale". A cycle during the
    session whose refresh landed today's partial bar for a minority of
    symbols and timed out for the rest, or one weekend-trading instrument
    in an equity universe, blocked planning outright even though the whole
    universe agreed on yesterday's close. Bars newer than the reference are
    dropped by the alignment, exactly as before October.
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

    eligible = max(len(series), expected_count or 0) - short_count
    needed = max(1, math.ceil(eligible * 0.8))
    tails = sorted((v.index[-1] for v in clean.values()), reverse=True)
    if len(tails) < needed:
        raise ValueError(
            f"price coverage blocked: only {len(tails)}/{eligible} mature series "
            f"have usable prices (latest {tails[0].date()})"
        )
    # The newest bar that ``needed`` series reach. Anything a minority has
    # beyond it is ignored rather than allowed to define "current".
    reference = tails[needed - 1]
    if freq == "1D" and (instant.normalize() - reference.normalize()).days > 4:
        raise ValueError(
            f"price coverage blocked: newest daily bar reached by {needed}/{eligible} "
            f"mature series is {reference.date()}"
        )
    for symbol in list(clean):
        last = clean[symbol].index[-1]
        if last < reference:
            excluded[symbol] = f"stale tail {last.date()} (latest {reference.date()})"
            del clean[symbol]
        elif last > reference:
            clean[symbol] = clean[symbol].loc[:reference]
    wide = pd.DataFrame(clean).sort_index().dropna(how="any")
    if len(wide) < required or wide.index[-1] != reference:
        raise ValueError("price coverage blocked: insufficient aligned history")
    return wide, excluded
