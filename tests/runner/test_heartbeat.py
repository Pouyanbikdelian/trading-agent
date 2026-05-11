"""Heartbeat file tests."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.runner import (
    heartbeat_age_seconds,
    heartbeat_is_stale,
    read_heartbeat,
    write_heartbeat,
)


def test_write_then_read(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    write_heartbeat(path, ts=ts, status="ok", cycle_no=5)
    out = read_heartbeat(path)
    assert out is not None
    assert out["status"] == "ok"
    assert out["cycle"] == 5
    assert out["ts"] == ts.isoformat()
    assert out["pid"] == os.getpid()


def test_write_with_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    write_heartbeat(path, ts=datetime.now(timezone.utc), status="ok",
                    cycle_no=1, extra={"orders": 7, "fills": 7})
    out = read_heartbeat(path)
    assert out is not None
    assert out["orders"] == 7


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_heartbeat(tmp_path / "does-not-exist.json") is None


def test_age_seconds_present(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    write_heartbeat(path, ts=datetime.now(timezone.utc), status="ok", cycle_no=1)
    age = heartbeat_age_seconds(path)
    assert age is not None
    assert 0 <= age < 5.0   # generous bound for slow CI


def test_age_seconds_missing(tmp_path: Path) -> None:
    assert heartbeat_age_seconds(tmp_path / "absent.json") is None


def test_is_stale_when_missing(tmp_path: Path) -> None:
    assert heartbeat_is_stale(tmp_path / "absent.json", max_age_seconds=30.0) is True


def test_is_stale_old_file(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    write_heartbeat(path, ts=datetime.now(timezone.utc), status="ok", cycle_no=1)
    # Backdate the mtime by an hour.
    old = time.time() - 3600
    os.utime(path, (old, old))
    assert heartbeat_is_stale(path, max_age_seconds=30.0) is True


def test_is_stale_fresh_file(tmp_path: Path) -> None:
    path = tmp_path / "hb.json"
    write_heartbeat(path, ts=datetime.now(timezone.utc), status="ok", cycle_no=1)
    assert heartbeat_is_stale(path, max_age_seconds=30.0) is False


def test_atomic_write_overwrites(tmp_path: Path) -> None:
    """Writing twice produces only the final payload — no half-written file
    and no tmp file left behind."""
    path = tmp_path / "hb.json"
    write_heartbeat(path, ts=datetime.now(timezone.utc), status="ok", cycle_no=1)
    write_heartbeat(path, ts=datetime.now(timezone.utc), status="halted", cycle_no=2)
    out = json.loads(path.read_text())
    assert out["status"] == "halted"
    assert out["cycle"] == 2
    # Tmp suffix must have been renamed away.
    assert not (tmp_path / "hb.json.tmp").exists()


def test_write_rejects_naive_datetime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        write_heartbeat(tmp_path / "hb.json", ts=datetime(2024, 1, 1),
                        status="ok", cycle_no=1)
