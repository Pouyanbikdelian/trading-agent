"""Market-hours jobs run on New York wall time in both DST seasons."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from trading.runner.runner import (
    NYSE_TZ,
    _cycle_dst_warning,
    _humanize_cron,
    _nyse_rth_every_15,
    _precycle_trigger,
)

NY = ZoneInfo("America/New_York")


def _fires_on(trigger, day: datetime) -> list[datetime]:
    """All fire times on one New York calendar day."""
    start = day.replace(hour=0, minute=0, tzinfo=NY)
    end = start + timedelta(days=1)
    out, prev, now = [], None, start
    while True:
        nxt = trigger.get_next_fire_time(prev, now)
        if nxt is None or nxt >= end:
            return out
        out.append(nxt.astimezone(NY))
        prev, now = nxt, nxt + timedelta(seconds=1)


def test_guards_run_inside_rth_in_summer_and_winter() -> None:
    trig = _nyse_rth_every_15(5)
    for day in (datetime(2026, 9, 24), datetime(2026, 12, 3)):  # EDT and EST Thursdays
        fires = _fires_on(trig, day)
        assert fires[0].strftime("%H:%M") == "09:35"
        assert fires[-1].strftime("%H:%M") == "15:50"
        assert len(fires) == 26
        assert all(f.hour < 16 and (f.hour, f.minute) >= (9, 30) for f in fires)


def test_sentinel_starts_at_the_open() -> None:
    fires = _fires_on(_nyse_rth_every_15(0), datetime(2026, 12, 3))
    assert [f.strftime("%H:%M") for f in fires[:3]] == ["09:30", "09:45", "10:00"]
    assert fires[-1].strftime("%H:%M") == "15:45"


def test_no_weekend_fires() -> None:
    assert _fires_on(_nyse_rth_every_15(5), datetime(2026, 9, 26)) == []  # Saturday


def test_ny_cycle_keeps_its_distance_from_the_close_across_dst() -> None:
    from apscheduler.triggers.cron import CronTrigger

    trig = CronTrigger.from_crontab("5 17 * * FRI", timezone=NYSE_TZ)
    summer = trig.get_next_fire_time(None, datetime(2026, 9, 21, tzinfo=timezone.utc))
    winter = trig.get_next_fire_time(None, datetime(2026, 11, 2, tzinfo=timezone.utc))
    assert summer.astimezone(NY).strftime("%H:%M") == winter.astimezone(NY).strftime("%H:%M")
    assert summer.astimezone(timezone.utc).hour == 21 and winter.astimezone(timezone.utc).hour == 22


def test_precycle_pm_run_follows_the_ny_cron() -> None:
    pre = _precycle_trigger("5 17 * * FRI", NYSE_TZ, lead_minutes=45)
    fire = pre.get_next_fire_time(None, datetime(2026, 11, 2, tzinfo=timezone.utc))
    assert fire.astimezone(NY).strftime("%a %H:%M") == "Fri 16:20"


def test_utc_afternoon_cron_warns_and_ny_cron_does_not() -> None:
    assert _cycle_dst_warning("5 21 * * FRI", "UTC") is not None
    assert _cycle_dst_warning("5 17 * * FRI", NYSE_TZ) is None
    assert _cycle_dst_warning("0 3 * * SUN", "UTC") is None
    # The reverse migration slip: SCHEDULE_TZ switched, CRON still in UTC hours.
    assert _cycle_dst_warning("5 21 * * FRI", NYSE_TZ) is not None


def test_humanized_cron_names_its_timezone() -> None:
    assert _humanize_cron("5 17 * * FRI", NYSE_TZ) == "Fridays 17:05 New York"
    assert _humanize_cron("5 21 * * FRI") == "Fridays 21:05 UTC"
