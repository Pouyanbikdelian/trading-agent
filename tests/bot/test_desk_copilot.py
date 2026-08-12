"""Approval-gated watchlist and lesson changes from the Telegram copilot."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from trading.bot import telegram as telegram_module
from trading.bot.desk import DeskChangeStore
from trading.bot.telegram import _dispatch
from trading.core.clock import FixedClock
from trading.memory.store import MemoryStore


def _watchlist(path: Path, *symbols: str) -> None:
    path.write_text("watchlist:\n" + "".join(f"  - {s}\n" for s in symbols))


def _store(tmp_path: Path, clock: FixedClock | None = None) -> DeskChangeStore:
    watchlist = tmp_path / "config" / "watchlist.yaml"
    watchlist.parent.mkdir()
    _watchlist(watchlist, "IONQ")
    return DeskChangeStore(tmp_path / "state", watchlist, clock=clock)


def _settings_stub(state_dir: Path, watchlist_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state_dir=state_dir,
        data_dir=state_dir / "data",
        watchlist_path=watchlist_path,
        trading_env="research",
        is_live_armed=lambda: False,
    )


def test_watchlist_change_is_a_proposal_until_approved(tmp_path: Path) -> None:
    store = _store(tmp_path)

    staged = store.propose_watchlist_add("NVDA")

    assert staged.proposal is not None
    assert store.watchlist.items() == ["IONQ"]
    result = store.approve(staged.proposal.id)
    assert "Added `NVDA`" in result.message
    assert store.watchlist.items() == ["IONQ", "NVDA"]

    events = [json.loads(line)["event"] for line in store.audit_path.read_text().splitlines()]
    assert events == ["proposed", "applied"]
    assert "no longer pending" in store.approve(staged.proposal.id).message


def test_expired_proposal_cannot_mutate_watchlist(tmp_path: Path) -> None:
    now = datetime(2026, 8, 12, tzinfo=timezone.utc)
    clock = FixedClock(now)
    store = _store(tmp_path, clock)
    staged = store.propose_watchlist_add("NVDA")
    assert staged.proposal is not None

    clock.instant = now + timedelta(minutes=11)
    assert "no longer pending" in store.approve(staged.proposal.id).message
    assert store.watchlist.items() == ["IONQ"]
    assert json.loads(store.path.read_text())[-1]["status"] == "expired"


def test_undo_is_approval_gated_and_linked_to_the_prior_change(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = store.propose_watchlist_add("NVDA").proposal
    assert original is not None
    store.approve(original.id)

    undo = store.propose_undo_last_watchlist_change().proposal
    assert undo is not None
    assert undo.kind == "watchlist_remove"
    assert undo.payload["undo_of"] == original.id
    assert store.watchlist.items() == ["IONQ", "NVDA"]

    applied = store.approve(undo.id)
    assert "reverted" in applied.message
    assert store.watchlist.items() == ["IONQ"]


def test_superseding_an_operator_lesson_preserves_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mem = MemoryStore(tmp_path / "state" / "memory")
    old_id = mem.add_lesson("Old operator lesson about quantum exposure.", tags="operator strong")
    mem.close()

    staged = store.propose_lesson_supersede(
        old_id, "I want continuous measured quantum exposure when valuations are attractive."
    )
    assert staged.proposal is not None
    result = store.approve(staged.proposal.id)
    assert "superseded" in result.message

    mem = MemoryStore(tmp_path / "state" / "memory")
    old = next(row for row in mem.lessons() if row["id"] == old_id)
    new = next(row for row in mem.lessons() if row["id"] != old_id)
    assert old["status"] == "retired"
    assert "Superseded by" in old["retired_why"]
    assert new["status"] == "established"
    assert "continuous measured quantum" in new["statement"]
    mem.close()


def test_historian_lesson_cannot_be_silently_rewritten(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mem = MemoryStore(tmp_path / "state" / "memory")
    lesson_id = mem.add_lesson("Historian finding about semiconductor breadth.", tags="historian")
    mem.close()

    assert (
        "not an active operator lesson"
        in store.propose_lesson_supersede(
            lesson_id, "Replacement statement that should never be written."
        ).message
    )
    assert "not an active operator lesson" in store.propose_lesson_archive(lesson_id).message


def test_plain_english_watchlist_request_stages_then_plain_approve_applies(
    tmp_path: Path, monkeypatch
) -> None:
    watchlist = tmp_path / "config" / "watchlist.yaml"
    watchlist.parent.mkdir()
    _watchlist(watchlist, "IONQ")
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path / "state", watchlist))

    async def no_llm(*args, **kwargs):
        raise AssertionError("a deterministic desk request must not call the LLM")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", no_llm)
    proposal = asyncio.run(_dispatch("I like NVDA, can you add it to our list"))
    assert isinstance(proposal, telegram_module.ButtonReply)
    assert "Add `NVDA`" in proposal
    assert "NVDA" not in DeskChangeStore(tmp_path / "state", watchlist).watchlist.items()

    applied = asyncio.run(_dispatch("approve"))
    assert applied is not None and "Added `NVDA`" in applied
    assert DeskChangeStore(tmp_path / "state", watchlist).watchlist.items() == ["IONQ", "NVDA"]

    undo = asyncio.run(_dispatch("undo last watchlist change"))
    assert isinstance(undo, telegram_module.ButtonReply)
    assert "undo" in undo.lower()
    assert DeskChangeStore(tmp_path / "state", watchlist).watchlist.items() == ["IONQ", "NVDA"]
    reverted = asyncio.run(_dispatch("approve"))
    assert reverted is not None and "reverted" in reverted
    assert DeskChangeStore(tmp_path / "state", watchlist).watchlist.items() == ["IONQ"]


def test_lesson_memory_request_is_answered_from_the_real_store(tmp_path: Path, monkeypatch) -> None:
    watchlist = tmp_path / "config" / "watchlist.yaml"
    watchlist.parent.mkdir()
    _watchlist(watchlist, "IONQ")
    state = tmp_path / "state"
    mem = MemoryStore(state / "memory")
    lid = mem.add_lesson("Quantum exposure needs position-size discipline.", tags="operator strong")
    mem.close()
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(state, watchlist))

    async def no_llm(*args, **kwargs):
        raise AssertionError("lesson listing must not call the LLM")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", no_llm)
    out = asyncio.run(_dispatch("tell me more what's inside our lessons memory"))
    assert out is not None and lid in out
    assert "Quantum exposure" in out


def test_stale_desk_button_cannot_approve_a_newer_proposal(tmp_path: Path, monkeypatch) -> None:
    from trading.bot import keyboards
    from trading.bot.telegram import _handle_callback

    watchlist = tmp_path / "config" / "watchlist.yaml"
    watchlist.parent.mkdir()
    _watchlist(watchlist, "IONQ")
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path / "state", watchlist))
    store = DeskChangeStore(tmp_path / "state", watchlist)
    old = store.propose_watchlist_add("NVDA").proposal
    assert old is not None
    store.cancel(old.id)
    newest = store.propose_watchlist_add("MSFT").proposal
    assert newest is not None

    out = asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_DESK_APPROVE, old.id)))
    assert "no longer pending" in out
    assert store.watchlist.items() == ["IONQ"]
    assert store.pending(newest.id) is not None
