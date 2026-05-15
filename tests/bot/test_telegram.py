r"""Tests for the Telegram command bot.

Network is mocked via monkeypatch — these are hermetic unit tests and
can run in CI without TELEGRAM_BOT_TOKEN set.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading.bot import telegram as telegram_module
from trading.bot.telegram import (
    _atomic_write_json,
    _cmd_halt,
    _cmd_heartbeat,
    _cmd_resume,
    _cmd_status,
    _dispatch,
)


def _settings_stub(state_dir: Path) -> SimpleNamespace:
    """Drop-in replacement for the frozen pydantic Settings — only the
    fields the bot reads."""
    return SimpleNamespace(
        state_dir=state_dir,
        trading_env="research",
        is_live_armed=lambda: False,
    )


def test_atomic_write_json_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "halt.json"
    _atomic_write_json(p, {"halted": True, "reason": "test"})
    assert json.loads(p.read_text())["halted"] is True


def test_dispatch_unknown_returns_help_hint() -> None:
    out = asyncio.run(_dispatch("/wat"))
    assert "unknown command" in (out or "")


def test_dispatch_non_command_returns_none() -> None:
    assert asyncio.run(_dispatch("hello bot")) is None


def test_cmd_halt_writes_halt_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_halt(["bad", "news"])
    assert "HALTED" in out
    payload = json.loads((tmp_path / "halt.json").read_text())
    assert payload["halted"] is True
    assert "bad news" in payload["reason"]
    assert payload["flatten_on_next_cycle"] is True


def test_cmd_resume_clears_halt_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _cmd_halt(["temp"])
    out = _cmd_resume()
    assert "RESUMED" in out
    payload = json.loads((tmp_path / "halt.json").read_text())
    assert payload["halted"] is False


def test_cmd_status_reports_halted_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _cmd_halt(["reason-x"])
    out = _cmd_status()
    assert "HALTED" in out
    assert "reason-x" in out


def test_cmd_status_reports_running_when_no_halt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_status()
    assert "running" in out


def test_cmd_heartbeat_with_no_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_heartbeat()
    assert "no heartbeat" in out


def test_cmd_heartbeat_with_recent_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    (tmp_path / "heartbeat.json").write_text("{}")
    out = _cmd_heartbeat()
    assert "ago" in out


@pytest.mark.parametrize(
    "cmd, marker",
    [
        ("/start", "commands"),
        ("/help", "commands"),
    ],
)
def test_dispatch_help_variants(cmd: str, marker: str) -> None:
    out = asyncio.run(_dispatch(cmd))
    assert out is not None and marker in out
