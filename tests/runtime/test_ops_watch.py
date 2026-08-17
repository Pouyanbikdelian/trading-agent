"""Ops watchdog — hermetic: tmp state dir, no network (no tokens set)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import trading.runtime.ops_watch as ow
from trading.runtime.broker_liveness import FILENAME
from trading.runtime.ops_watch import check_broker_liveness, check_health, run_ops_watch

NOW = datetime(2026, 6, 12, 15, 0, tzinfo=timezone.utc)


class _FakeUsage:
    total, used, free = 100e9, 40e9, 60e9  # healthy 40% disk


def _healthy_host(monkeypatch) -> None:
    """The dev machine's real disk/memory must not leak into tests —
    discovered the hard way when a 95%-full laptop failed CI."""
    monkeypatch.setattr(ow.shutil, "disk_usage", lambda _: _FakeUsage())
    monkeypatch.setattr(ow, "_mem_available_mb", lambda: 4096.0)


def _touch_all(state_dir: Path) -> None:
    for rel in (
        "runner.db",
        "news.json",
        "econ_watch.json",
        "market_watch.json",
        "last_committee.json",
        "agent_pm/portfolio.json",
    ):
        p = state_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")


def _write_broker_liveness(
    state_dir: Path,
    *,
    ready: bool,
    checked_at: datetime,
    last_success_at: datetime | None,
    detail: str = "Gateway rejected authenticated request",
) -> None:
    payload: dict[str, object] = {
        "ready": ready,
        "probe": "reqCurrentTime",
        "checked_at": checked_at.isoformat(),
        "last_success_at": last_success_at.isoformat() if last_success_at else None,
    }
    if not ready:
        payload["detail"] = detail
    (state_dir / FILENAME).write_text(json.dumps(payload))


def test_missing_artifacts_are_issues(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    issues = check_health(tmp_path, now=NOW)
    assert any("missing" in i for i in issues)
    assert len(issues) >= len(("runner", "news", "econ", "macro", "committee", "pm"))


def test_fresh_artifacts_are_healthy(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    _touch_all(tmp_path)
    now = datetime.now(tz=timezone.utc)  # mtimes are real, so use real now
    issues = check_health(tmp_path, now=now)
    assert issues == []


def test_stale_artifact_flagged(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    _touch_all(tmp_path)
    old = time.time() - 200 * 3600
    os.utime(tmp_path / "news.json", (old, old))
    now = datetime.now(tz=timezone.utc)
    issues = check_health(tmp_path, now=now)
    assert any("news watch" in i and "stale" in i for i in issues)


def test_halt_state_is_reported(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    _touch_all(tmp_path)
    (tmp_path / "halt.json").write_text(json.dumps({"halted": True, "reason": "drawdown"}))
    issues = check_health(tmp_path, now=datetime.now(tz=timezone.utc))
    assert any("HALTED" in i for i in issues)


def test_failed_authenticated_broker_probe_is_an_immediate_health_issue(
    tmp_path: Path, monkeypatch
) -> None:
    _healthy_host(monkeypatch)
    _touch_all(tmp_path)
    _write_broker_liveness(
        tmp_path,
        ready=False,
        checked_at=NOW,
        last_success_at=NOW - timedelta(minutes=2),
    )

    issues = check_health(tmp_path, now=NOW)

    assert any(
        "broker API liveness: unavailable" in issue
        and "last authenticated response 2.0m ago" in issue
        for issue in issues
    )


def test_stale_successful_broker_probe_is_not_hidden_by_a_fresh_file(
    tmp_path: Path, monkeypatch
) -> None:
    _healthy_host(monkeypatch)
    _write_broker_liveness(
        tmp_path,
        ready=True,
        checked_at=NOW,
        last_success_at=NOW - timedelta(minutes=6),
    )

    issues = check_broker_liveness(tmp_path, now=NOW)

    assert issues == ["broker API liveness: last authenticated response 6.0m ago (limit 5m)"]


def test_debounce_and_recovery(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    monkeypatch.setattr(ow, "_send_ops", lambda _message: True)
    now = datetime.now(tz=timezone.utc)
    first = run_ops_watch(tmp_path, now=now)
    assert first["alerted"]  # everything missing -> alerts
    second = run_ops_watch(tmp_path, now=now)
    assert second["alerted"] == []  # debounced
    _touch_all(tmp_path)
    third = run_ops_watch(tmp_path, now=now)
    assert third["issues"] == [] and third["recovered"]


def test_failed_delivery_does_not_debounce_an_unreported_issue(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    attempts: list[str] = []
    monkeypatch.setattr(ow, "_send_ops", lambda message: attempts.append(message) and False)

    first = run_ops_watch(tmp_path, now=NOW)
    second = run_ops_watch(tmp_path, now=NOW + timedelta(minutes=1))

    assert first["alerted"] == []
    assert second["alerted"] == []
    assert len(attempts) == 2
    assert ow._load(tmp_path)["reported"] == {}


def test_failed_recovery_delivery_keeps_issue_reported(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    _touch_all(tmp_path)
    _write_broker_liveness(tmp_path, ready=False, checked_at=NOW, last_success_at=None)
    monkeypatch.setattr(ow, "_send_ops", lambda _message: True)
    run_ops_watch(tmp_path, now=NOW)

    _write_broker_liveness(tmp_path, ready=True, checked_at=NOW, last_success_at=NOW)
    monkeypatch.setattr(ow, "_send_ops", lambda _message: False)
    failed = run_ops_watch(tmp_path, now=NOW + timedelta(minutes=1))

    assert failed["recovered"] == []
    assert "broker API liveness" in ow._load(tmp_path)["reported"]


def test_recovery_is_sent_even_when_a_different_issue_is_new(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    _touch_all(tmp_path)
    sent: list[str] = []
    monkeypatch.setattr(ow, "_send_ops", lambda message: sent.append(message) or True)
    _write_broker_liveness(tmp_path, ready=False, checked_at=NOW, last_success_at=None)

    first = run_ops_watch(tmp_path, now=NOW)
    assert any("broker API liveness" in issue for issue in first["alerted"])

    _write_broker_liveness(tmp_path, ready=True, checked_at=NOW, last_success_at=NOW)
    (tmp_path / "halt.json").write_text(json.dumps({"halted": True, "reason": "test"}))
    second = run_ops_watch(tmp_path, now=NOW + timedelta(minutes=1))

    assert any("trading HALTED" in issue for issue in second["alerted"])
    assert second["recovered"] == ["broker API liveness"]
    assert any(
        message.startswith("✅ Ops watchdog: recovered — broker API liveness") for message in sent
    )
