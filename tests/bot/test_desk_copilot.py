"""Approval-gated watchlist and lesson changes from the Telegram copilot."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_desk_change_boundary_never_imports_execution() -> None:
    """Watchlist/lesson staging must not acquire an order-capable dependency."""
    code = (
        "import sys\n"
        "import trading.bot.desk\n"
        "bad = [m for m in sys.modules if m.startswith('trading.execution')]\n"
        "assert not bad, bad\n"
    )

    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, proc.stderr


def test_watchlist_change_is_a_proposal_until_approved(tmp_path: Path) -> None:
    store = _store(tmp_path)
    baseline = store.watchlist.config_path.read_text()

    staged = store.propose_watchlist_add("NVDA")

    assert staged.proposal is not None
    assert store.watchlist.items() == ["IONQ"]
    result = store.approve(staged.proposal.id)
    assert "Added `NVDA`" in result.message
    assert store.watchlist.items() == ["IONQ", "NVDA"]
    # The bot's production config mount is deliberately read-only. A
    # Telegram edit must therefore leave the versioned baseline untouched
    # and persist only the state-volume overlay shared with the dashboard.
    assert store.watchlist.config_path.read_text() == baseline
    assert json.loads(store.watchlist.overrides_path.read_text()) == {
        "added": ["NVDA"],
        "removed": [],
    }

    events = [json.loads(line)["event"] for line in store.audit_path.read_text().splitlines()]
    assert events == ["proposed", "applied"]
    assert "no longer pending" in store.approve(staged.proposal.id).message


def test_multi_watchlist_removal_is_one_atomic_approved_change(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _watchlist(store.watchlist.config_path, "IONQ", "RGTI", "FORM", "NVDA")

    staged = store.propose_watchlist_remove_many(["IONQ", "RGTI", "FORM"])

    assert staged.proposal is not None
    assert staged.proposal.kind == "watchlist_remove_many"
    assert store.watchlist.items() == ["IONQ", "RGTI", "FORM", "NVDA"]
    assert "Remove 3 names" in telegram_module._desk_reply(staged)

    result = store.approve(staged.proposal.id)

    assert "Removed 3 names" in result.message
    assert store.watchlist.items() == ["NVDA"]
    audit = [json.loads(line) for line in store.audit_path.read_text().splitlines()]
    assert audit[-1]["old_value"] == ["IONQ", "RGTI", "FORM", "NVDA"]
    assert audit[-1]["new_value"] == ["NVDA"]

    undo = store.propose_undo_last_watchlist_change().proposal
    assert undo is not None and undo.kind == "watchlist_add_many"
    assert store.approve(undo.id).message.startswith("✅ Restored 3 names")
    assert store.watchlist.items() == ["IONQ", "RGTI", "FORM", "NVDA"]


def test_multi_watchlist_change_is_all_or_nothing_when_one_symbol_is_absent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    staged = store.propose_watchlist_remove_many(["IONQ", "RGTI"])

    assert staged.proposal is None
    assert "No change staged" in staged.message
    assert store.watchlist.items() == ["IONQ"]
    assert not store.path.exists()


def test_watchlist_overrides_merge_with_later_config_edits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.propose_watchlist_add("NVDA").proposal
    assert added is not None
    store.approve(added.id)

    # A deploy can update the versioned baseline after the operator has
    # already added a name. The overlay must merge instead of replacing it.
    _watchlist(store.watchlist.config_path, "IONQ", "MSFT")
    assert store.watchlist.items() == ["IONQ", "MSFT", "NVDA"]

    removed = store.propose_watchlist_remove("IONQ").proposal
    assert removed is not None
    store.approve(removed.id)
    assert store.watchlist.items() == ["MSFT", "NVDA"]
    assert "IONQ" in store.watchlist.config_path.read_text()


def test_add_restores_a_removed_name_after_a_baseline_config_change(tmp_path: Path) -> None:
    """A stale remove marker must not hide an approved restoration forever."""
    store = _store(tmp_path)
    removed = store.propose_watchlist_remove("IONQ").proposal
    assert removed is not None
    store.approve(removed.id)
    assert store.watchlist.items() == []

    # The next deploy removes IONQ from static config too.  It is now an
    # operator addition again, and a fresh approval needs to clear the old
    # removal marker as well as add it to the overlay.
    _watchlist(store.watchlist.config_path, "MSFT")
    restored = store.propose_watchlist_add("IONQ").proposal
    assert restored is not None
    store.approve(restored.id)

    assert store.watchlist.items() == ["MSFT", "IONQ"]
    assert json.loads(store.watchlist.overrides_path.read_text()) == {
        "added": ["IONQ"],
        "removed": [],
    }


def test_watchlist_approval_works_with_read_only_config_mount(tmp_path: Path) -> None:
    """Mirror the bot container, where ``/app/config`` is mounted read-only."""
    store = _store(tmp_path)
    baseline = store.watchlist.config_path.read_text()
    store.watchlist.config_path.chmod(0o444)

    staged = store.propose_watchlist_add("NVDA").proposal
    assert staged is not None
    store.approve(staged.id)

    assert store.watchlist.items() == ["IONQ", "NVDA"]
    assert store.watchlist.config_path.read_text() == baseline
    assert store.watchlist.overrides_path.exists()


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


def test_corrupt_proposal_state_fails_closed_without_overwriting_history(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.parent.mkdir()
    store.path.write_text("{not-json")

    with pytest.raises(ValueError, match="proposal state is invalid"):
        store.propose_watchlist_add("NVDA")
    assert store.path.read_text() == "{not-json"
    assert "Could not read the desk change" in store.approve().message


def test_failed_ledger_finalization_never_claims_an_applied_change_failed(
    tmp_path: Path, monkeypatch
) -> None:
    store = _store(tmp_path)
    staged = store.propose_watchlist_add("NVDA").proposal
    assert staged is not None

    def fail_finalize(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_replace", fail_finalize)
    result = store.approve(staged.id)

    assert store.watchlist.items() == ["IONQ", "NVDA"]
    assert "was applied" in result.message
    assert "Do not approve it again" in result.message


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


def test_plain_english_multi_watchlist_removal_is_one_proposal(tmp_path: Path, monkeypatch) -> None:
    watchlist = tmp_path / "config" / "watchlist.yaml"
    watchlist.parent.mkdir()
    _watchlist(watchlist, "IONQ", "RGTI", "FORM", "NVDA")
    state = tmp_path / "state"
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(state, watchlist))

    async def no_llm(*args, **kwargs):
        raise AssertionError("a deterministic multi-ticker request must not call the LLM")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", no_llm)
    proposal = asyncio.run(_dispatch("remove IONQ, RGTI and FORM from watchlist"))

    assert isinstance(proposal, telegram_module.ButtonReply)
    assert "Remove 3 names" in proposal
    assert all(symbol in proposal for symbol in ("IONQ", "RGTI", "FORM"))
    assert DeskChangeStore(state, watchlist).watchlist.items() == ["IONQ", "RGTI", "FORM", "NVDA"]

    applied = asyncio.run(_dispatch("approve"))

    assert applied is not None and "Removed 3 names" in applied
    assert DeskChangeStore(state, watchlist).watchlist.items() == ["NVDA"]


def test_free_text_desk_failure_does_not_crash_or_fall_through_to_llm(
    tmp_path: Path, monkeypatch
) -> None:
    watchlist = tmp_path / "config" / "watchlist.yaml"
    watchlist.parent.mkdir()
    _watchlist(watchlist, "IONQ")
    state = tmp_path / "state"
    state.mkdir()
    (state / "desk_change_proposals.json").write_text("{not-json")
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(state, watchlist))

    async def no_llm(*args, **kwargs):
        raise AssertionError("a broken desk ledger must not fall through to the LLM")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", no_llm)
    out = asyncio.run(_dispatch("I like NVDA, can you add it to our list"))

    assert isinstance(out, telegram_module.PlainReply)
    assert "No change was applied" in out
    assert (state / "desk_change_proposals.json").read_text() == "{not-json"


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
