r"""Tests for the file-based command queue (bot ↔ runner)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading.runtime.commands import (
    EXECUTED_DIR,
    PENDING_DIR,
    RUNNING_DIR,
    Command,
    CommandType,
    mark_executed,
    mark_running,
    pending_commands,
    submit,
)


def test_command_round_trip() -> None:
    cmd = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 10})
    data = cmd.to_dict()
    restored = Command.from_dict(data)
    assert restored.id == cmd.id
    assert restored.type == CommandType.BUY
    assert restored.args == {"symbol": "AAPL", "qty": 10}


def test_submit_writes_pending(tmp_path: Path) -> None:
    cmd = Command.new(CommandType.SELL, args={"symbol": "MSFT"})
    path = submit(cmd, tmp_path)
    assert path.exists()
    assert path.parent.name == PENDING_DIR
    payload = json.loads(path.read_text())
    assert payload["type"] == "sell"


def test_pending_commands_oldest_first(tmp_path: Path) -> None:
    """Commands should be returned in submission order so we process
    older requests first — fairness across pending queue depth."""
    import time

    submit(Command.new(CommandType.FLATTEN), tmp_path)
    time.sleep(0.02)
    second = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 1})
    submit(second, tmp_path)
    time.sleep(0.02)
    third = Command.new(CommandType.SELL, args={"symbol": "AAPL"})
    submit(third, tmp_path)

    out = pending_commands(tmp_path)
    assert len(out) == 3
    assert out[0].type == CommandType.FLATTEN
    assert out[1].id == second.id
    assert out[2].id == third.id


def test_mark_running_moves_to_running_dir(tmp_path: Path) -> None:
    cmd = Command.new(CommandType.CLOSE, args={"symbol": "TSLA"})
    submit(cmd, tmp_path)
    running_path = mark_running(cmd, tmp_path)
    assert running_path.exists()
    assert running_path.parent.name == RUNNING_DIR
    # Pending entry is gone.
    pending_path = tmp_path / "commands" / PENDING_DIR / f"{cmd.id}.json"
    assert not pending_path.exists()


def test_mark_running_missing_raises(tmp_path: Path) -> None:
    cmd = Command.new(CommandType.BUY, args={"symbol": "AAPL", "qty": 5})
    with pytest.raises(FileNotFoundError):
        mark_running(cmd, tmp_path)


def test_mark_executed_records_status_and_result(tmp_path: Path) -> None:
    cmd = Command.new(
        CommandType.FX_CONVERT, args={"from_ccy": "CHF", "to_ccy": "USD", "amount": 50000}
    )
    submit(cmd, tmp_path)
    mark_running(cmd, tmp_path)
    mark_executed(
        cmd, tmp_path, status="ok", result={"filled_at_rate": 1.123, "usd_received": 56150}
    )
    executed_path = tmp_path / "commands" / EXECUTED_DIR / f"{cmd.id}.json"
    assert executed_path.exists()
    payload = json.loads(executed_path.read_text())
    assert payload["status"] == "ok"
    assert payload["result"]["usd_received"] == 56150
    # Running entry cleaned up
    running_path = tmp_path / "commands" / RUNNING_DIR / f"{cmd.id}.json"
    assert not running_path.exists()


def test_corrupt_pending_file_is_quarantined(tmp_path: Path) -> None:
    """A garbage file in pending/ shouldn't break the watcher — it
    should be moved aside (renamed .corrupt) and we keep going."""
    pending_dir = tmp_path / "commands" / PENDING_DIR
    pending_dir.mkdir(parents=True)
    (pending_dir / "bad.json").write_text("not json {")
    # Should not raise
    cmds = pending_commands(tmp_path)
    assert cmds == []
    # The corrupt file got renamed
    assert (pending_dir / "bad.corrupt").exists()


def test_pending_skips_tmp_files(tmp_path: Path) -> None:
    """In-progress writes (.tmp suffix) must not be picked up — would
    cause a half-written-JSON read race."""
    pending_dir = tmp_path / "commands" / PENDING_DIR
    pending_dir.mkdir(parents=True)
    (pending_dir / "incomplete.json.tmp").write_text('{"id": "x"}')
    cmds = pending_commands(tmp_path)
    assert cmds == []
