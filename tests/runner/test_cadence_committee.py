"""The committee debates once per scheduled cycle, just before it (2026-09-27).

It ran Monday and Friday every week while the cycle moved to every second
Friday: four debates per decision, three feeding nothing.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from trading.runner.cadence import gate
from trading.runner.runner import _committee_trigger

NY = ZoneInfo("America/New_York")


def _fires(trigger, start: datetime, n: int) -> list[datetime]:
    out, prev, now = [], None, start
    for _ in range(n):
        nxt = trigger.get_next_fire_time(prev, now)
        out.append(nxt.astimezone(NY))
        prev, now = nxt, nxt
    return out


def test_two_hours_before_the_cycle_on_cycle_fridays_only(monkeypatch) -> None:
    monkeypatch.delenv("AGENTS_COMMITTEE_CRON", raising=False)
    monkeypatch.delenv("AGENTS_COMMITTEE_LEAD_MINUTES", raising=False)
    trig = gate(
        _committee_trigger("0 15 * * FRI", "America/New_York"),
        every_weeks=2,
        anchor=date(2026, 10, 9),
        tz="America/New_York",
    )
    fires = _fires(trig, datetime(2026, 9, 27, tzinfo=timezone.utc), 3)
    assert [f.date() for f in fires] == [date(2026, 10, 9), date(2026, 10, 23), date(2026, 11, 6)]
    assert all((f.weekday(), f.hour, f.minute) == (4, 13, 0) for f in fires)


def test_override_keeps_its_time_but_stays_on_cycle_weeks(monkeypatch) -> None:
    monkeypatch.setenv("AGENTS_COMMITTEE_CRON", "0 13 * * MON,FRI")
    trig = gate(
        _committee_trigger("0 15 * * FRI", "America/New_York"),
        every_weeks=2,
        anchor=date(2026, 10, 9),
        tz="America/New_York",
    )
    fires = _fires(trig, datetime(2026, 9, 27, tzinfo=timezone.utc), 3)
    # Mon 5 Oct and Fri 9 Oct are the cycle week; the off week is skipped.
    assert [f.date() for f in fires] == [date(2026, 10, 5), date(2026, 10, 9), date(2026, 10, 19)]
