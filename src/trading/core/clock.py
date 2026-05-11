"""Timezone-aware clock + market session helpers.

A single Clock object is used everywhere instead of ``datetime.utcnow()`` so
we can deterministically replay timestamps in backtests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Anything that can give the current time. Real systems use UtcClock;
    backtests use a frozen/iterating clock."""

    def now(self) -> datetime: ...


@dataclass(frozen=True)
class UtcClock:
    """Real wall-clock time in UTC."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


@dataclass
class FixedClock:
    """Clock pinned to an instant — used in tests."""

    instant: datetime

    def now(self) -> datetime:
        return self.instant


@dataclass
class IteratingClock:
    """Clock advanced by the backtester one bar at a time."""

    current: datetime

    def now(self) -> datetime:
        return self.current

    def advance_to(self, ts: datetime) -> None:
        if ts.tzinfo is None:
            raise ValueError("Clock can only advance to timezone-aware datetimes")
        self.current = ts
