"""Cycle-outcome watchdog: missed cycles, stuck desk, no trades."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.runtime import cycle_watch as cw

UTC = timezone.utc
CRON = "5 17 * * FRI"
NY = "America/New_York"


def _row(ts: datetime, status: str = "ok", orders: int = 3) -> dict[str, object]:
    return {"ts": ts, "status": status, "orders_submitted": orders}


def test_last_fire_follows_new_york_across_dst() -> None:
    # Summer: 17:05 EDT = 21:05 UTC. Winter: 17:05 EST = 22:05 UTC.
    summer = cw.last_scheduled_fire(CRON, NY, datetime(2026, 9, 26, 12, tzinfo=UTC))
    winter = cw.last_scheduled_fire(CRON, NY, datetime(2026, 11, 7, 12, tzinfo=UTC))
    assert summer is not None and summer.astimezone(UTC) == datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    assert winter is not None and winter.astimezone(UTC) == datetime(2026, 11, 6, 22, 5, tzinfo=UTC)


def test_last_fire_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError):
        cw.last_scheduled_fire(CRON, NY, datetime(2026, 9, 26, 12))


def test_missed_cycle_is_critical_after_grace() -> None:
    fire = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    older = [_row(fire - timedelta(days=7))]
    finding = cw.check_missed_cycle(older, cron=CRON, tz=NY, now=fire + timedelta(hours=3))
    assert finding is not None and finding.level == "critical"
    assert "Missed cycle" in finding.message


def test_no_missed_alert_inside_grace_or_when_it_ran() -> None:
    fire = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    older = [_row(fire - timedelta(days=7))]
    # Inside grace, the previous week's fire is what counts, and it ran.
    assert cw.check_missed_cycle(older, cron=CRON, tz=NY, now=fire + timedelta(hours=1)) is None
    ran = [_row(fire + timedelta(seconds=2)), *older]
    assert cw.check_missed_cycle(ran, cron=CRON, tz=NY, now=fire + timedelta(hours=3)) is None


def test_stuck_desk_needs_a_streak_of_non_executable_statuses() -> None:
    t = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    one = [_row(t, "halted_review", 0), _row(t - timedelta(days=7), "ok", 4)]
    assert cw.check_stuck_desk(one) is None
    two = [_row(t, "halted_review", 0), _row(t - timedelta(days=7), "error", 0)]
    finding = cw.check_stuck_desk(two)
    assert finding is not None and finding.level == "critical"
    assert "error" in finding.message and "halted_review" in finding.message


def test_no_orders_is_not_stuck() -> None:
    t = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    rows = [_row(t - timedelta(days=7 * i), "no_orders", 0) for i in range(3)]
    assert cw.check_stuck_desk(rows) is None


def test_no_trade_streak_is_a_warning() -> None:
    t = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    rows = [_row(t - timedelta(days=7 * i), "no_orders", 0) for i in range(4)]
    finding = cw.check_no_trades(rows)
    assert finding is not None and finding.level == "warning"
    assert cw.check_no_trades(rows[:3]) is None


def test_findings_alert_once_per_incident(tmp_path: Path) -> None:
    t = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    rows = [_row(t - timedelta(days=7 * i), "halted", 0) for i in range(4)]
    findings = cw.evaluate(rows, cron=CRON, tz=NY, now=t + timedelta(minutes=30))
    # A stuck desk implies no trades; only the stronger finding is raised.
    assert {f.key.split(":")[0] for f in findings} == {"stuck"}

    assert len(cw.unalerted(tmp_path, findings)) == 1
    assert cw.unalerted(tmp_path, findings) == []

    # A new streak (a fresh first cycle) is a new incident.
    later = [_row(t + timedelta(days=21 + 7 * i), "halted", 0) for i in range(2)]
    restart = [_row(t + timedelta(days=14), "ok", 5)]
    new = cw.evaluate(later[::-1] + restart, cron=CRON, tz=NY, now=t + timedelta(days=28, hours=1))
    assert any(f.key.startswith("stuck:") for f in cw.unalerted(tmp_path, new))


def test_corrupt_state_re_alerts_rather_than_silences(tmp_path: Path) -> None:
    (tmp_path / cw.STATE_FILE).write_text("{not json")
    f = cw.CycleWatchFinding(key="stuck:x", level="critical", message="m")
    assert cw.unalerted(tmp_path, [f]) == [f]


def test_rows_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError):
        cw.check_stuck_desk([{"ts": datetime(2026, 9, 25), "status": "halted"}] * 2)


def test_a_growing_streak_keeps_one_key(tmp_path: Path) -> None:
    """Keyed on the last healthy cycle, so each new stuck cycle is not news."""
    t = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    healthy = _row(t - timedelta(days=70), "ok", 5)
    week = [_row(t - timedelta(days=7 * i), "halted", 0) for i in range(3)]
    later = [_row(t + timedelta(days=7), "halted", 0), *week]
    k1 = cw.check_stuck_desk([*week, healthy])
    k2 = cw.check_stuck_desk([*later, healthy])
    assert k1 is not None and k2 is not None and k1.key == k2.key


def test_no_trade_warning_still_fires_without_a_stuck_desk() -> None:
    t = datetime(2026, 9, 25, 21, 5, tzinfo=UTC)
    rows = [_row(t - timedelta(days=7 * i), "no_orders", 0) for i in range(4)]
    found = cw.evaluate(rows, cron=CRON, tz=NY, now=t + timedelta(minutes=30))
    assert [f.key.split(":")[0] for f in found] == ["notrade"]


def test_stale_liveness_alerts_once_per_outage(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import os
    import time
    from types import SimpleNamespace

    import trading.runner.runner as runner_module
    from trading.runner.runner import Runner

    monkeypatch.setattr(runner_module, "settings", SimpleNamespace(state_dir=tmp_path))
    hb = tmp_path / "heartbeat.json"
    hb.write_text("{}")
    old = time.time() - 30 * 3600
    os.utime(hb, (old, old))
    sent: list[str] = []
    fake = SimpleNamespace(
        HEARTBEAT_WATCHDOG_HOURS=25.0,
        _last_success_ts=None,
        alerts=SimpleNamespace(warning=sent.append, critical=sent.append),
        cycle=SimpleNamespace(runner_store=SimpleNamespace(recent_cycles=lambda limit: [])),
        config=SimpleNamespace(schedule_cron="5 17 * * FRI", schedule_tz="America/New_York"),
    )
    for _ in range(3):
        asyncio.run(Runner._watchdog(fake))  # type: ignore[arg-type]
    liveness = [m for m in sent if "broker snapshot" in m]
    assert len(liveness) == 1
