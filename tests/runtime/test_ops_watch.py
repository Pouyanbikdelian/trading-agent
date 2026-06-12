"""Ops watchdog — hermetic: tmp state dir, no network (no tokens set)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import trading.runtime.ops_watch as ow
from trading.runtime.ops_watch import check_health, run_ops_watch

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


def test_debounce_and_recovery(tmp_path: Path, monkeypatch) -> None:
    _healthy_host(monkeypatch)
    monkeypatch.delenv("OPS_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)  # no sends in tests
    now = datetime.now(tz=timezone.utc)
    first = run_ops_watch(tmp_path, now=now)
    assert first["alerted"]  # everything missing -> alerts
    second = run_ops_watch(tmp_path, now=now)
    assert second["alerted"] == []  # debounced
    _touch_all(tmp_path)
    third = run_ops_watch(tmp_path, now=now)
    assert third["issues"] == [] and third["recovered"]
