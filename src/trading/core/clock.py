"""Timezone-aware clock + market session helpers.

A single Clock object is used everywhere instead of ``datetime.utcnow()`` so
we can deterministically replay timestamps in backtests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


def artifact_age_seconds(path: Path, *, now: float | None = None) -> float | None:
    """Seconds since a state artifact was last written, or None if absent.

    SQLite-aware. In WAL mode writes land in ``<db>-wal`` and the main
    file is only touched on checkpoint, so ``runner.db``'s mtime reports
    the last CHECKPOINT, not the last write. Two independent freshness
    checks — the dashboard's ops panel and ``runtime.ops_watch`` — both
    stat'd it directly and both told the operator "broker snapshot 91h
    old" about a runner writing a snapshot every 60 seconds, on the
    morning it was armed for live trading.

    Lives here, once, because the first fix went into the dashboard alone
    and the watchdog kept lying for another hour.
    """
    import os

    now = now if now is not None else time.time()
    stamps = [
        os.stat(p).st_mtime
        for p in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm"))
        if p.exists()
    ]
    return (now - max(stamps)) if stamps else None
