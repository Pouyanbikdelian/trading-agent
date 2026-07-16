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


def test_dispatch_plain_text_routes_to_copilot(monkeypatch) -> None:
    """Since 2026-07-16 plain chat goes to the read-only copilot instead
    of being ignored — the bot IS the assistant, no /ask prefix needed."""
    seen: dict[str, str] = {}

    async def fake_copilot(question: str, symbol: str | None = None) -> str:
        seen["q"] = question
        return "copilot reply"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake_copilot)
    out = asyncio.run(_dispatch("why did we buy MU last week?"))
    assert out == "copilot reply"
    assert seen["q"] == "why did we buy MU last week?"


def test_dispatch_tiny_fragment_gets_hint_not_llm(monkeypatch) -> None:
    async def fake_copilot(question: str, symbol: str | None = None) -> str:
        raise AssertionError("copilot must not be called for fragments")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake_copilot)
    out = asyncio.run(_dispatch("ok"))
    assert out is not None and "full question" in out
    assert asyncio.run(_dispatch("")) is None  # empty stays silent


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


# ---------------------------------------------------------------------------
# Snapshot age warning — audit fix #6
#
# May 2026 incident: /positions and /balances showed snapshot data from
# 64 minutes earlier while the broker was actually flat. Operators
# couldn't tell. These tests pin the thresholds and message format.
# ---------------------------------------------------------------------------


def test_snapshot_age_warning_silent_for_fresh_snapshot() -> None:
    from datetime import datetime, timedelta, timezone

    from trading.bot.telegram import _snapshot_age_warning

    fresh = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
    assert _snapshot_age_warning(fresh) is None


def test_snapshot_age_warning_fires_for_old_snapshot() -> None:
    from datetime import datetime, timedelta, timezone

    from trading.bot.telegram import _snapshot_age_warning

    # 45 minutes is past the 30-min threshold but still under 1h so the
    # formatter prints it in minutes rather than hours.
    stale = datetime.now(tz=timezone.utc) - timedelta(minutes=45)
    msg = _snapshot_age_warning(stale)
    assert msg is not None
    assert "old" in msg.lower() or "stale" in msg.lower()
    assert "45" in msg


def test_snapshot_age_warning_hours_format() -> None:
    from datetime import datetime, timedelta, timezone

    from trading.bot.telegram import _snapshot_age_warning

    stale = datetime.now(tz=timezone.utc) - timedelta(hours=3)
    msg = _snapshot_age_warning(stale)
    assert msg is not None
    # Should use "h" suffix for hour-scale staleness, not minutes
    first_line = msg.split("\n", 1)[0]
    assert "h" in first_line


def test_snapshot_age_warning_handles_naive_datetime() -> None:
    """RunnerStore can return naive datetimes from older rows — accept either."""
    from datetime import datetime, timedelta, timezone

    from trading.bot.telegram import _snapshot_age_warning

    stale = (datetime.now(tz=timezone.utc) - timedelta(minutes=45)).replace(tzinfo=None)
    msg = _snapshot_age_warning(stale)
    assert msg is not None


# ------------------------------------------------- copilot plain-text sending


def test_copilot_reply_is_plain_reply(tmp_path: Path, monkeypatch) -> None:
    """LLM answers must bypass Telegram markdown: legacy markdown eats
    underscore pairs, so 'NOW_agent_pm_simulated_book' rendered as
    'NOWagentpmsimulatedbook' (operator report, 2026-07-16)."""
    import trading.copilot as copilot_pkg
    from trading.bot.telegram import PlainReply

    stub = _settings_stub(tmp_path)
    stub.data_dir = tmp_path / "data"
    monkeypatch.setattr(telegram_module, "settings", stub)
    monkeypatch.setattr(copilot_pkg, "answer", lambda q, **kw: "answer with_under_scores intact")
    out = asyncio.run(_dispatch("what is the PM book holding today?"))
    assert isinstance(out, PlainReply)
    assert out == "answer with_under_scores intact"

    # Errors from the copilot are bot-authored (markdown-safe) — not plain.
    def boom(q, **kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(copilot_pkg, "answer", boom)
    out = asyncio.run(_dispatch("what is the PM book holding today?"))
    assert not isinstance(out, PlainReply)
    assert "copilot error" in (out or "")


def test_send_plain_omits_parse_mode() -> None:
    from trading.bot.telegram import _send

    sent: list[dict] = []

    class FakeResp:
        status_code = 200
        text = ""

    class FakeClient:
        async def post(self, url, json=None, timeout=None):
            sent.append(json)
            return FakeResp()

    asyncio.run(_send(FakeClient(), "tok", "42", "a_b_c", plain=True))
    asyncio.run(_send(FakeClient(), "tok", "42", "a_b_c"))
    assert "parse_mode" not in sent[0]  # plain: text goes out verbatim
    assert sent[1].get("parse_mode") == "Markdown"  # default path unchanged
