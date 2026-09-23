"""The poll loop must survive whatever the operator types.

Regression tests for 2026-09-23: an unbalanced quote killed the bot, an
edited message re-executed, and any handler exception dropped the message.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from trading.bot import telegram as tg


def _settings_stub(state_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(state_dir=state_dir, trading_env="research", is_live_armed=lambda: False)


def _capture_replies(monkeypatch) -> list[str]:
    sent: list[str] = []

    async def fake_send_reply(_client, _token, _chat, text):
        sent.append(text)

    monkeypatch.setattr(tg, "_send_reply", fake_send_reply)
    return sent


def _update(text: str, *, edited: bool = False, chat: str = "42") -> dict:
    body = {"chat": {"id": chat}, "text": text}
    return {"update_id": 1, ("edited_message" if edited else "message"): body}


def test_halt_with_an_apostrophe_still_halts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tg, "settings", _settings_stub(tmp_path))

    out = asyncio.run(tg._dispatch("/halt it's bad data")) or ""

    assert "HALTED" in out
    payload = json.loads((tmp_path / "halt.json").read_text())
    assert payload["halted"] is True
    assert "it's bad data" in payload["reason"]


def test_unbalanced_quote_in_any_command_does_not_raise(monkeypatch) -> None:
    out = asyncio.run(tg._dispatch('/wat "unclosed'))
    assert out is not None


def test_edited_message_is_never_executed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tg, "settings", _settings_stub(tmp_path))
    sent = _capture_replies(monkeypatch)
    dispatched: list[str] = []

    async def fake_dispatch(text, replied_to=None):
        dispatched.append(text)
        return "ran"

    monkeypatch.setattr(tg, "_dispatch", fake_dispatch)

    asyncio.run(tg._handle_message_update(None, "t", "42", _update("/flatten", edited=True)))

    assert dispatched == []
    assert sent == [tg._EDIT_IGNORED_REPLY]


def test_edited_plain_text_is_ignored_silently(monkeypatch) -> None:
    sent = _capture_replies(monkeypatch)

    async def boom(*_a, **_k):
        raise AssertionError("must not dispatch")

    monkeypatch.setattr(tg, "_dispatch", boom)
    asyncio.run(tg._handle_message_update(None, "t", "42", _update("typo fix", edited=True)))
    assert sent == []


def test_a_new_message_is_still_dispatched(monkeypatch) -> None:
    sent = _capture_replies(monkeypatch)

    async def fake_dispatch(text, replied_to=None):
        return f"ran {text}"

    monkeypatch.setattr(tg, "_dispatch", fake_dispatch)
    asyncio.run(tg._handle_message_update(None, "t", "42", _update("/status")))
    assert sent == ["ran /status"]


def test_handler_exception_is_reported_not_fatal(monkeypatch) -> None:
    sent = _capture_replies(monkeypatch)

    async def exploding(text, replied_to=None):
        raise RuntimeError("disk full")

    monkeypatch.setattr(tg, "_dispatch", exploding)
    asyncio.run(tg._handle_message_update(None, "t", "42", _update("/status")))
    assert len(sent) == 1 and "command failed" in sent[0] and "disk full" in sent[0]


def test_unauthorized_chat_is_ignored(monkeypatch) -> None:
    sent = _capture_replies(monkeypatch)

    async def boom(*_a, **_k):
        raise AssertionError("must not dispatch")

    monkeypatch.setattr(tg, "_dispatch", boom)
    asyncio.run(tg._handle_message_update(None, "t", "42", _update("/halt", chat="999")))
    assert sent == []
