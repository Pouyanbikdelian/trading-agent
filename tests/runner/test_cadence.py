"""Every-second-Friday cycles (2026-09-26): one rule, used everywhere.

The operator moved the scheduled rebalance from every Friday to every
second Friday at the same time. Cron cannot say that, and APScheduler's
``week="*/2"`` counts ISO week numbers — 2026 has 53 of them, so weeks 53
and 1 would both be "odd" and two cycles would land back to back. The
cadence is therefore taken from an anchor date, and the cycle, the PM
decision, the pre-cycle broker check, the missed-cycle watchdog and the
dashboard all use the same gate.
"""

from __future__ import annotations

import itertools
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from trading.runner.cadence import DEFAULT_ANCHOR, cadence_from, describe, gate, in_cycle_week
from trading.runner.runner import _precycle_trigger
from trading.runtime import cycle_watch

NY = "America/New_York"
CRON = "0 15 * * FRI"
ANCHOR = date(2026, 10, 9)


def _fires(trigger, start: datetime, n: int) -> list[datetime]:
    out, prev, now = [], None, start
    for _ in range(n):
        f = trigger.get_next_fire_time(prev, now)
        if f is None:
            break
        out.append(f)
        prev, now = f, f + timedelta(seconds=1)
    return out


def _biweekly():
    return gate(CronTrigger.from_crontab(CRON, timezone=NY), every_weeks=2, anchor=ANCHOR, tz=NY)


class TestTheCadence:
    def test_weekly_leaves_the_trigger_untouched(self) -> None:
        base = CronTrigger.from_crontab(CRON, timezone=NY)
        assert gate(base, every_weeks=1, anchor=ANCHOR, tz=NY) is base

    def test_every_second_friday_from_the_anchor_same_time(self) -> None:
        start = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
        fires = _fires(_biweekly(), start, 4)
        assert [f.astimezone(ZoneInfo(NY)).date() for f in fires] == [
            date(2026, 10, 9),
            date(2026, 10, 23),
            date(2026, 11, 6),
            date(2026, 11, 20),
        ]
        # 15:00 New York in both DST seasons (EDT until 1 Nov, EST after).
        assert {f.astimezone(ZoneInfo(NY)).strftime("%H:%M") for f in fires} == {"15:00"}

    def test_no_back_to_back_cycles_across_a_53_week_year(self) -> None:
        fires = _fires(_biweekly(), datetime(2026, 11, 1, tzinfo=timezone.utc), 12)
        gaps = {(b - a).days for a, b in itertools.pairwise(fires)}
        assert gaps == {14}  # ISO parity would give a 7 around 2026-W53 -> 2027-W01

    def test_the_off_weeks_are_the_ones_between(self) -> None:
        on = datetime(2026, 10, 9, 19, tzinfo=timezone.utc)
        assert in_cycle_week(on, every_weeks=2, anchor=ANCHOR, tz=NY)
        assert not in_cycle_week(on + timedelta(days=7), every_weeks=2, anchor=ANCHOR, tz=NY)
        assert not in_cycle_week(on - timedelta(days=7), every_weeks=2, anchor=ANCHOR, tz=NY)

    def test_a_pre_cycle_job_fires_on_the_cycle_friday_only(self) -> None:
        pm = gate(_precycle_trigger(CRON, NY, lead_minutes=45), every_weeks=2, anchor=ANCHOR, tz=NY)
        fires = _fires(pm, datetime(2026, 9, 26, tzinfo=timezone.utc), 2)
        assert [f.astimezone(ZoneInfo(NY)).strftime("%Y-%m-%d %H:%M") for f in fires] == [
            "2026-10-09 14:15",
            "2026-10-23 14:15",
        ]

    def test_settings_default_to_weekly_and_a_missing_anchor_is_documented(self) -> None:
        assert cadence_from(SimpleNamespace()) == (1, None)
        assert cadence_from(SimpleNamespace(cycle_every_weeks=2, cycle_anchor_date=None)) == (
            2,
            None,
        )
        assert in_cycle_week(
            datetime(2026, 10, 9, 19, tzinfo=timezone.utc), every_weeks=2, anchor=None, tz=NY
        ) == in_cycle_week(
            datetime(2026, 10, 9, 19, tzinfo=timezone.utc),
            every_weeks=2,
            anchor=DEFAULT_ANCHOR,
            tz=NY,
        )

    def test_the_banner_says_it(self) -> None:
        nxt = datetime(2026, 10, 9, 19, tzinfo=timezone.utc)
        assert describe(1, nxt, NY) == ""
        assert describe(2, nxt, NY) == ", every 2 weeks — next Fri 09 Oct"


class TestTheWatchdogKnowsAboutOffWeeks:
    def test_an_off_week_friday_is_not_a_missed_cycle(self) -> None:
        last = [{"ts": datetime(2026, 10, 9, 19, 0, 5, tzinfo=timezone.utc)}]
        # Saturday after the OFF-week Friday 16 Oct: nothing was due.
        now = datetime(2026, 10, 17, 14, tzinfo=timezone.utc)
        weekly = cycle_watch.check_missed_cycle(last, cron=CRON, tz=NY, now=now)
        biweekly = cycle_watch.check_missed_cycle(
            last, cron=CRON, tz=NY, now=now, every_weeks=2, anchor=ANCHOR
        )
        assert weekly is not None and biweekly is None

    def test_a_missing_cycle_friday_is_still_caught(self) -> None:
        last = [{"ts": datetime(2026, 10, 9, 19, 0, 5, tzinfo=timezone.utc)}]
        now = datetime(2026, 10, 24, 14, tzinfo=timezone.utc)  # day after 23 Oct
        f = cycle_watch.check_missed_cycle(
            last, cron=CRON, tz=NY, now=now, every_weeks=2, anchor=ANCHOR
        )
        assert f is not None and "2026-10-23" in f.message


class TestTheDashboardAgrees:
    def test_coming_up_shows_the_next_cycle_friday(self) -> None:
        from trading.dashboard.cockpit import schedule_block

        s = SimpleNamespace(
            cycle_every_weeks=2, cycle_anchor_date=ANCHOR, pm_pre_cycle_lead_minutes=45
        )
        jobs = {
            j["key"]: j
            for j in schedule_block(
                s, cron=CRON, tz=NY, now=datetime(2026, 10, 10, tzinfo=timezone.utc)
            )
        }
        assert jobs["cycle"]["label"] == "Rebalance cycle (every 2 weeks)"
        assert jobs["cycle"]["at"].startswith("2026-10-23T19:00")
        assert jobs["pm"]["at"].startswith("2026-10-23T18:15")
