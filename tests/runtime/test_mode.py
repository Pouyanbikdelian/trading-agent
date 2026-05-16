r"""Tests for the operator-mode persistence layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.runtime.mode import (
    Mode,
    PendingModeChange,
    clear_pending,
    read_mode,
    read_pending,
    write_mode,
    write_pending,
)


def test_default_mode_is_neutral(tmp_path: Path) -> None:
    state = read_mode(tmp_path / "mode.json")  # file does not exist
    assert state.mode == Mode.NEUTRAL


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "mode.json"
    s = write_mode(p, Mode.DEFENSE, set_by="cli", reason="testing")
    assert s.mode == Mode.DEFENSE
    assert s.set_by == "cli"
    again = read_mode(p)
    assert again.mode == Mode.DEFENSE
    assert again.reason == "testing"


def test_write_is_atomic_no_partial_file(tmp_path: Path) -> None:
    p = tmp_path / "mode.json"
    write_mode(p, Mode.BEAR)
    # No leftover tmp files (atomic replace must clean up).
    leftover = [f for f in tmp_path.iterdir() if f.name.startswith("mode.json.")]
    assert leftover == []


def test_corrupt_file_falls_back_to_neutral(tmp_path: Path) -> None:
    p = tmp_path / "mode.json"
    p.write_text("not json {")
    state = read_mode(p)
    assert state.mode == Mode.NEUTRAL


def test_parse_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        Mode.parse("yolo")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("bull", Mode.BULL),
        ("BULL", Mode.BULL),
        ("  defense ", Mode.DEFENSE),
        ("flatten", Mode.FLATTEN),
    ],
)
def test_parse_normalises(name: str, expected: Mode) -> None:
    assert Mode.parse(name) == expected


# ----- Pending change ------------------------------------------------------


def test_pending_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "pending.json"
    pending = PendingModeChange(
        new_mode=Mode.DEFENSE,
        requested_at=datetime.now(tz=timezone.utc).isoformat(),
        requested_by="telegram",
        reason="vol spike",
    )
    write_pending(p, pending)
    out = read_pending(p)
    assert out is not None
    assert out.new_mode == Mode.DEFENSE
    assert out.requested_by == "telegram"


def test_pending_expiry(tmp_path: Path) -> None:
    old = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
    pending = PendingModeChange(
        new_mode=Mode.BEAR,
        requested_at=old.isoformat(),
        requested_by="telegram",
        ttl_seconds=600,  # 10 minutes
    )
    assert pending.is_expired() is True


def test_pending_not_expired(tmp_path: Path) -> None:
    recent = datetime.now(tz=timezone.utc) - timedelta(seconds=5)
    pending = PendingModeChange(
        new_mode=Mode.DEFENSE,
        requested_at=recent.isoformat(),
        requested_by="telegram",
        ttl_seconds=600,
    )
    assert pending.is_expired() is False


def test_pending_clear_removes_file(tmp_path: Path) -> None:
    p = tmp_path / "pending.json"
    write_pending(
        p,
        PendingModeChange(
            new_mode=Mode.DEFENSE,
            requested_at=datetime.now(tz=timezone.utc).isoformat(),
            requested_by="telegram",
        ),
    )
    assert p.exists()
    clear_pending(p)
    assert not p.exists()
    # Clearing again is a no-op (idempotent).
    clear_pending(p)


def test_pending_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_pending(tmp_path / "pending.json") is None
