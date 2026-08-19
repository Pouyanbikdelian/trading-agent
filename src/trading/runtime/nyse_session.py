"""NYSE session facts used by time-sensitive runner safeguards.

The runner must take a daily-equity baseline from a real US session, not
from whichever wall-clock day happened to run first.  This module centralises
the exchange-calendar lookup so every caller uses the same holiday, DST and
early-close rules.  It is deliberately read-only: it has no scheduler,
broker, or risk-manager dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

NYSE_TIMEZONE: Final = ZoneInfo("America/New_York")
"""IANA timezone used to resolve an instant to its NYSE calendar date."""

OPENING_CAPTURE_DURATION: Final = timedelta(minutes=5)
"""Short, half-open window in which the runner may capture a session baseline."""


@dataclass(frozen=True)
class NYSESession:
    """One scheduled NYSE cash session, with exchange-supplied UTC boundaries."""

    label: date
    market_open: datetime
    market_close: datetime


_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def _utc_instant(when: datetime) -> datetime:
    """Normalize an instant while refusing naive times at the safety boundary."""
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("NYSE session helpers require a timezone-aware datetime")
    return when.astimezone(timezone.utc)


def _utc_datetime(value: pd.Timestamp) -> datetime:
    """Convert a pandas calendar boundary to an aware UTC ``datetime``."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("NYSE calendar returned a timezone-naive session boundary")
    return timestamp.to_pydatetime().astimezone(timezone.utc)


@lru_cache(maxsize=512)
def nyse_session_on(label: date) -> NYSESession | None:
    """Return the actual NYSE session for ``label``, or ``None`` when closed.

    ``pandas_market_calendars`` ships the holiday and special-close schedule
    locally.  Caching by date keeps the one-row lookup negligible for the
    runner's minute-level health loop without making any network call.
    """
    schedule = _NYSE_CALENDAR.schedule(start_date=label, end_date=label)
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    return NYSESession(
        label=label,
        market_open=_utc_datetime(row["market_open"]),
        market_close=_utc_datetime(row["market_close"]),
    )


def current_nyse_session(when: datetime) -> NYSESession | None:
    """Session scheduled on ``when``'s current New York calendar day.

    This intentionally returns a session before its open and after its close:
    the caller often needs that day's opening boundary to decide whether a
    baseline capture is due.  Weekends and exchange holidays return ``None``.
    """
    instant = _utc_instant(when)
    return nyse_session_on(instant.astimezone(NYSE_TIMEZONE).date())


def current_nyse_session_label(when: datetime) -> date | None:
    """NYSE session label for the current New York day, if it is a session."""
    session = current_nyse_session(when)
    return session.label if session is not None else None


def opening_capture_bounds(
    when: datetime,
    *,
    duration: timedelta = OPENING_CAPTURE_DURATION,
) -> tuple[datetime, datetime] | None:
    """Return today's safe baseline-capture interval, or ``None`` when closed.

    The interval is ``[market_open, market_open + duration)`` and is capped
    at the actual close defensively.  A positive duration is required so a
    configuration typo cannot accidentally make every post-open snapshot a
    valid daily baseline.
    """
    if duration <= timedelta(0):
        raise ValueError("opening capture duration must be positive")
    session = current_nyse_session(when)
    if session is None:
        return None
    return session.market_open, min(session.market_open + duration, session.market_close)


def is_opening_capture_window(
    when: datetime,
    *,
    duration: timedelta = OPENING_CAPTURE_DURATION,
) -> bool:
    """Whether ``when`` falls inside today's half-open baseline-capture window."""
    instant = _utc_instant(when)
    bounds = opening_capture_bounds(instant, duration=duration)
    if bounds is None:
        return False
    opens_at, closes_at = bounds
    return opens_at <= instant < closes_at
