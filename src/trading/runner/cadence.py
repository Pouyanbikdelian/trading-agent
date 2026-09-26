"""Every-N-weeks cadence for the rebalance cycle and the jobs derived from it.

Why (2026-09-26). The operator moved the scheduled rebalance from every
Friday to every second Friday, same time, with manual ``/cycle`` runs in
between whenever he chooses. Cron cannot say "every other Friday", and the
obvious APScheduler spelling (``week="*/2"``) counts ISO week numbers: a
53-week year puts two cycle weeks back to back (53, then 1), a 52-week
year puts three weeks between them. So the parity is taken from a fixed
anchor date instead, which is continuous across year ends.

One rule has to hold everywhere, or the pieces drift apart: the cycle, the
pre-cycle broker check, the PM decision that feeds the cycle, the
missed-cycle watchdog and the dashboard's "coming up" list must all agree
on which Fridays are cycle Fridays. They all ask :func:`in_cycle_week`, or
wrap their trigger in :class:`EveryNWeeks`, with the same two settings:

* ``CYCLE_EVERY_WEEKS`` — 1 (every week, the old behaviour) or more;
* ``CYCLE_ANCHOR_DATE`` — any date in a week that DOES have a cycle.

Weeks are Monday-based in the schedule's own time zone, so a trigger that
fires 45 minutes before the cycle, on the same local day, is always in the
same week as the cycle it prepares.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.triggers.base import BaseTrigger

#: Used only when CYCLE_EVERY_WEEKS > 1 and no anchor is configured, so the
#: parity is at least stable and documented rather than depending on the
#: day the process happened to start. Fri 2026-10-09 is two weeks after the
#: last weekly cycle (2026-09-25).
DEFAULT_ANCHOR = date(2026, 10, 9)


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def cadence_from(settings: Any) -> tuple[int, date | None]:
    """``(every_weeks, anchor)`` from settings, defaulting to weekly."""
    try:
        every = int(getattr(settings, "cycle_every_weeks", 1) or 1)
    except (TypeError, ValueError):
        every = 1
    anchor = getattr(settings, "cycle_anchor_date", None)
    return max(1, every), anchor if isinstance(anchor, date) else None


def in_cycle_week(
    when: datetime, *, every_weeks: int, anchor: date | None, tz: str = "America/New_York"
) -> bool:
    """Is ``when`` (aware) in a week that has a scheduled cycle?"""
    if every_weeks <= 1:
        return True
    if when.tzinfo is None:
        raise ValueError("when must be timezone-aware")
    local = when.astimezone(ZoneInfo(tz)).date()
    weeks = (_monday(local) - _monday(anchor or DEFAULT_ANCHOR)).days // 7
    return weeks % every_weeks == 0


class EveryNWeeks(BaseTrigger):  # type: ignore[misc]
    """A trigger that fires only on the weeks :func:`in_cycle_week` allows.

    Delegates the arithmetic to the wrapped trigger (cron, DST and all) and
    skips its fire times that fall in an off week.
    """

    def __init__(self, base: Any, *, every_weeks: int, anchor: date | None, tz: str) -> None:
        self.base = base
        self.every_weeks = max(1, int(every_weeks))
        self.anchor = anchor
        self.tz = tz

    def get_next_fire_time(
        self, previous_fire_time: datetime | None, now: datetime
    ) -> datetime | None:
        fire = self.base.get_next_fire_time(previous_fire_time, now)
        # Bounded: a daily cron at every_weeks=8 skips at most ~56 fires.
        for _ in range(70 * self.every_weeks):
            if fire is None or in_cycle_week(
                fire, every_weeks=self.every_weeks, anchor=self.anchor, tz=self.tz
            ):
                return fire
            fire = self.base.get_next_fire_time(fire, fire + timedelta(seconds=1))
        return None

    def __str__(self) -> str:
        return f"every {self.every_weeks} weeks ({self.base})"

    def __repr__(self) -> str:
        return (
            f"<EveryNWeeks every_weeks={self.every_weeks} anchor={self.anchor} base={self.base!r}>"
        )


def gate(trigger: Any, *, every_weeks: int, anchor: date | None, tz: str) -> Any:
    """``trigger`` unchanged when weekly, otherwise wrapped."""
    if trigger is None or every_weeks <= 1:
        return trigger
    return EveryNWeeks(trigger, every_weeks=every_weeks, anchor=anchor, tz=tz)


def describe(every_weeks: int, next_fire: datetime | None, tz: str) -> str:
    """The suffix an operator reads: '' when weekly, else 'every 2 weeks — next Fri 09 Oct'."""
    if every_weeks <= 1:
        return ""
    nxt = f" — next {next_fire.astimezone(ZoneInfo(tz)):%a %d %b}" if next_fire else ""
    return f", every {every_weeks} weeks{nxt}"
