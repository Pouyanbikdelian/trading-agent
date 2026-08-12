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
    out = asyncio.run(_dispatch("/wat")) or ""
    # Too short to be a question, too far from any command to suggest one.
    assert "no command" in out and "/help" in out


def test_dispatch_plain_text_routes_to_copilot(monkeypatch) -> None:
    """Since 2026-07-16 plain chat goes to the read-only copilot instead
    of being ignored — the bot IS the assistant, no /ask prefix needed."""
    seen: dict[str, str] = {}

    async def fake_copilot(question: str, symbol: str | None = None, **kw) -> str:
        seen["q"] = question
        return "copilot reply"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake_copilot)
    out = asyncio.run(_dispatch("why did we buy MU last week?"))
    assert out == "copilot reply"
    assert seen["q"] == "why did we buy MU last week?"


@pytest.mark.parametrize(
    "question",
    [
        "what are we having on hold?",
        "Which tickers are pinned?",
        "show the no-sell positions",
        "list cycle-protected names",
    ],
)
def test_plain_language_holds_inventory_reads_live_registry(
    tmp_path: Path, monkeypatch, question: str
) -> None:
    """Never let PM memory stand in for the operator's hard pins."""
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["NVDA", "CEG"]}))

    async def boom(*args, **kwargs) -> str:
        raise AssertionError("hold inventory must not call the copilot")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", boom)
    out = asyncio.run(_dispatch(question))

    assert out is not None
    assert "NVDA" in out and "CEG" in out
    assert "Pinned positions" in out


def test_hold_advice_question_still_reaches_copilot(tmp_path: Path, monkeypatch) -> None:
    """Only a list request is deterministic; advice still needs reasoning."""
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))

    async def fake(question: str, **kwargs) -> str:
        return "copilot advice"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake)
    assert asyncio.run(_dispatch("should I put CEG on hold?")) == "copilot advice"


@pytest.mark.parametrize(
    "message",
    [
        "but I have them already on hold",
        "these should be on hold: AMAT, CEG, GEV, NVDA, TER",
    ],
)
def test_existing_holds_correction_reads_registry_not_copilot(
    tmp_path: Path, monkeypatch, message: str
) -> None:
    """A correction must not become an invented PM mandate or LLM answer."""
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    (tmp_path / "holds.json").write_text(
        json.dumps({"symbols": ["AMAT", "CEG", "GEV", "NVDA", "TER"]})
    )

    async def boom(*args, **kwargs) -> str:
        raise AssertionError("existing pin confirmation must not call the copilot")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", boom)
    out = asyncio.run(_dispatch(message))

    assert out is not None
    assert "AMAT" in out and "TER" in out
    assert "Pinned positions" in out


def test_dispatch_tiny_fragment_gets_hint_not_llm(monkeypatch) -> None:
    async def fake_copilot(question: str, symbol: str | None = None, **kw) -> str:
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


def test_cmd_halt_does_not_promise_a_flatten(tmp_path: Path, monkeypatch) -> None:
    """This test used to assert ``flatten_on_next_cycle is True``, which
    pinned a flag that nothing anywhere read. /halt therefore replied
    "Next cycle will force-flatten positions" and no cycle ever did — an
    operator halting in an emergency would have believed the book was
    being closed. Halting stops trading; /flatten exits.
    """
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))

    out = _cmd_halt(["bad", "news"])

    payload = json.loads((tmp_path / "halt.json").read_text())
    assert "flatten_on_next_cycle" not in payload
    assert "flatten" not in out.lower().split("`/flatten`")[0]
    # And it must say what a halt does NOT do.
    assert "still fully exposed" in out


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


# ---------------------------------------------------------------------------
# Batch 2 — forgiving commands: typos, prose, and out-of-order commands
# ---------------------------------------------------------------------------


def test_typo_gets_a_suggestion_not_an_error() -> None:
    out = asyncio.run(_dispatch("/postions")) or ""
    assert "/positions" in out and "did you mean" in out


def test_typo_suggestion_costs_no_llm_call(monkeypatch) -> None:
    """The matcher is local. A typo must never spend money."""

    def boom(*a, **k):
        raise AssertionError("copilot must not be called for a near-miss typo")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", boom)
    assert "/approve" in (asyncio.run(_dispatch("/aprove")) or "")


def test_slashed_prose_falls_through_to_the_copilot(monkeypatch) -> None:
    """'/whats up with INTC' is an operator talking, not a bad command.

    It must not be bent into a real command — /halt was a live match for
    this string before the first-letter rule was added."""
    seen: dict[str, str] = {}

    async def fake(question: str, symbol: str | None = None, **kw) -> str:
        seen["q"] = question
        return "answered"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake)
    out = asyncio.run(_dispatch("/whats up with INTC today")) or ""
    assert out == "answered"
    assert seen["q"].startswith("whats up with INTC")  # slash stripped


def test_resume_when_not_halted_says_so(tmp_path: Path, monkeypatch) -> None:
    """A no-op '/resume' used to answer '✅ RESUMED — halt cleared', which
    reads as confirmation that a halt was lifted."""
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_resume()
    assert "not halted" in out.lower()
    assert "RESUMED" not in out


def test_resume_after_halt_still_clears(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _cmd_halt(["bad data"])
    out = _cmd_resume()
    assert "RESUMED" in out
    assert json.loads((tmp_path / "halt.json").read_text())["halted"] is False


def test_approve_with_nothing_pending_reports_state(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_approve

    stub = _settings_stub(tmp_path)
    stub.require_cycle_approval = False
    monkeypatch.setattr(telegram_module, "settings", stub)
    out = _cmd_approve([])
    assert "nothing awaiting approval" in out
    assert "/cycle" in out  # the valid next step
    assert "REQUIRE" in out  # explains why no cycle ever pauses


def test_proposal_repeats_the_exact_runner_plan_without_an_llm(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _APPROVAL_PENDING_FILE, _cmd_proposal

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _atomic_write_json(
        tmp_path / _APPROVAL_PENDING_FILE,
        {
            "id": "20260812-abcdef",
            "phase": "initial",
            "plan": {
                "base_currency": "CHF",
                "order_count": 2,
                "gross_turnover": 1_200.0,
                "turnover_pct": 1.4,
                "net_cash_change": -400.0,
                "orders": [
                    {
                        "symbol": "AMD",
                        "side": "BUY",
                        "quantity": 1,
                        "notional_base": 400.0,
                        "context": "new position",
                    },
                    {
                        "symbol": "CNC",
                        "side": "SELL",
                        "quantity": 12,
                        "notional_base": 800.0,
                        "context": "closes 12 held",
                    },
                ],
                "unchanged_symbols": ["DELL"],
            },
        },
    )

    out = _cmd_proposal()

    assert "AMD" in out and "buy 1" in out
    assert "CNC" in out and "sell 12" in out
    assert "Net cash released: CHF 400" in out
    assert "not additional exposure" in out


def test_plain_english_cycle_question_uses_the_exact_pending_plan(
    tmp_path: Path, monkeypatch
) -> None:
    from trading.bot.telegram import _APPROVAL_PENDING_FILE

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _atomic_write_json(
        tmp_path / _APPROVAL_PENDING_FILE,
        {
            "id": "20260812-abcdef",
            "phase": "initial",
            "plan": {
                "base_currency": "CHF",
                "order_count": 2,
                "gross_turnover": 1_200.0,
                "turnover_pct": 1.4,
                "net_cash_change": -400.0,
                "orders": [
                    {
                        "symbol": "AMD",
                        "side": "BUY",
                        "quantity": 1,
                        "notional_base": 400.0,
                        "context": "new position",
                    },
                    {
                        "symbol": "CNC",
                        "side": "SELL",
                        "quantity": 12,
                        "notional_base": 800.0,
                        "context": "closes 12 held",
                    },
                ],
            },
        },
    )

    async def should_not_call_copilot(*args, **kwargs):
        raise AssertionError("active-plan questions are deterministic, not LLM-routed")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", should_not_call_copilot)
    out = asyncio.run(_dispatch("I don't understand what this cycle is proposing to buy?")) or ""

    assert "AMD" in out and "CNC" in out
    assert "new position" in out and "closes 12 held" in out


def test_candidates_show_pending_alternatives_not_a_new_signal(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _APPROVAL_PENDING_FILE, _cmd_candidates

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _atomic_write_json(
        tmp_path / _APPROVAL_PENDING_FILE,
        {
            "id": "20260812-abcdef",
            "phase": "initial",
            "can_pick": True,
            "candidates": [{"symbol": "SNDK", "score": 5.39, "in_basket": False}],
        },
    )

    out = _cmd_candidates()

    assert "alternatives, not the proposed trade" in out
    assert "SNDK" in out and "/pick" in out


def test_reject_and_pick_share_the_state_aware_reply(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_pick, _cmd_reject

    stub = _settings_stub(tmp_path)
    stub.require_cycle_approval = True
    monkeypatch.setattr(telegram_module, "settings", stub)
    for out in (_cmd_reject(), _cmd_pick(["1", "3"])):
        assert "nothing awaiting approval" in out
        assert "hasn't completed a cycle yet" in out or "Last cycle" in out


def test_confirm_without_a_preview_names_the_current_mode(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_confirm

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_confirm()
    assert "nothing staged" in out
    assert "/mode" in out


def test_cancel_disambiguates_from_cancel_order_and_reject(tmp_path: Path, monkeypatch) -> None:
    """Three different 'stop that' commands is the sharpest wrong-order
    trap in the bot; the empty case should teach the difference."""
    from trading.bot.telegram import _cmd_cancel

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_cancel()
    assert "/cancel_order" in out and "/reject" in out


def test_cancel_order_rejects_an_unknown_id(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_cancel_order

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    monkeypatch.setattr(telegram_module, "_in_flight_order_ids", lambda: ["abc123def456"])
    queued: list[str] = []
    monkeypatch.setattr(
        telegram_module, "_queue_command", lambda t, a: queued.append(a["client_order_id"]) or "ok"
    )
    out = _cmd_cancel_order(["nope999"])
    assert "no in-flight order" in out
    assert not queued  # nothing queued for a bogus id


def test_cancel_order_accepts_a_unique_prefix(tmp_path: Path, monkeypatch) -> None:
    """/pending prints ids truncated to 14 chars — typing what you see
    must work."""
    from trading.bot.telegram import _cmd_cancel_order

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    monkeypatch.setattr(telegram_module, "_in_flight_order_ids", lambda: ["abc123def456789"])
    queued: list[str] = []
    monkeypatch.setattr(
        telegram_module, "_queue_command", lambda t, a: queued.append(a["client_order_id"]) or "ok"
    )
    _cmd_cancel_order(["abc123"])
    assert queued == ["abc123def456789"]  # resolved to the full id


def test_cancel_order_with_no_orders_in_flight(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_cancel_order

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    monkeypatch.setattr(telegram_module, "_in_flight_order_ids", lambda: [])
    assert "nothing to cancel" in _cmd_cancel_order(["abc"])


def test_hold_warns_when_the_symbol_is_not_in_the_book(tmp_path: Path, monkeypatch) -> None:
    """Ghost pins ate basket slots in the July 2026 audit — warn at the
    moment the pin is made."""
    from trading.bot.telegram import _cmd_hold

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    monkeypatch.setattr(telegram_module, "_holds_position", lambda s: False)
    out = _cmd_hold(["NVDA"])
    assert "pinned" in out
    assert "isn't in the current book" in out and "/unhold NVDA" in out


def test_hold_is_quiet_when_the_symbol_is_held(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_hold

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    monkeypatch.setattr(telegram_module, "_holds_position", lambda s: True)
    out = _cmd_hold(["NVDA"])
    assert "pinned" in out and "isn't in the current book" not in out


def test_usage_errors_carry_an_example(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_close, _cmd_unhold

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    assert "AAPL" in _cmd_close([])
    assert "NVDA" in _cmd_unhold([])


def test_cancel_order_when_store_unreadable_still_queues(tmp_path: Path, monkeypatch) -> None:
    """_queue_command returns None on success, so the caveat path must not
    concatenate onto it — that would raise TypeError mid-cancel."""
    from trading.bot.telegram import _cmd_cancel_order

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    monkeypatch.setattr(telegram_module, "_in_flight_order_ids", lambda: None)
    queued: list[str] = []

    def fake_queue(t: str, a: dict) -> None:
        queued.append(a["client_order_id"])
        return None

    monkeypatch.setattr(telegram_module, "_queue_command", fake_queue)
    out = _cmd_cancel_order(["abc123"])
    assert queued == ["abc123"]
    assert "couldn't verify" in out


def test_question_with_trailing_words_beats_a_weak_typo_match(monkeypatch) -> None:
    """'/hows the book' must be answered, not turned into /holds."""
    seen: dict[str, str] = {}

    async def fake(question: str, symbol: str | None = None, **kw) -> str:
        seen["q"] = question
        return "answered"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake)
    assert asyncio.run(_dispatch("/hows the book looking today")) == "answered"
    assert seen["q"].startswith("hows the book")


def test_typo_with_an_argument_is_still_suggested(monkeypatch) -> None:
    """'/aprove 80' carries an argument but is plainly a typo."""

    def boom(*a, **k):
        raise AssertionError("should suggest, not ask the copilot")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", boom)
    assert "/approve" in (asyncio.run(_dispatch("/aprove 80")) or "")


# ---------------------------------------------------------------------------
# Batch 3 — inline keyboards
# ---------------------------------------------------------------------------


def _pending_cycle(
    tmp_path: Path,
    cycle_id: str,
    *,
    can_pick: bool = True,
    phase: str = "initial",
    selectable_symbols: list[str] | None = None,
    preview_id: str | None = None,
    preview_fingerprint: str | None = None,
) -> None:
    from trading.bot.telegram import _APPROVAL_PENDING_FILE

    payload: dict[str, object] = {
        "id": cycle_id,
        "n_orders": 3,
        "deploy_pct": 40.0,
        "can_pick": can_pick,
        "phase": phase,
    }
    if selectable_symbols is not None:
        payload["selectable_symbols"] = selectable_symbols
        payload["plan_fingerprint"] = "displayed-plan"
    if preview_id is not None:
        payload["preview_id"] = preview_id
    if preview_fingerprint is not None:
        payload["preview_fingerprint"] = preview_fingerprint
    _atomic_write_json(
        tmp_path / _APPROVAL_PENDING_FILE,
        payload,
    )


def test_button_approves_the_cycle_it_was_minted_for(tmp_path: Path, monkeypatch) -> None:
    from trading.bot import keyboards
    from trading.bot.telegram import _handle_callback

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _pending_cycle(tmp_path, "abc12345deadbeef")
    out = asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_APPROVE, "abc12345")))
    assert "Approved" in out


def test_stale_button_refuses_to_approve_a_different_cycle(tmp_path: Path, monkeypatch) -> None:
    """The dangerous case: an old prompt still in the chat history. Tapping
    it must NOT approve today's basket, which the operator never saw."""
    from trading.bot import keyboards
    from trading.bot.telegram import _APPROVAL_DECISION_FILE, _handle_callback

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _pending_cycle(tmp_path, "99999999newcycle")
    out = asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_APPROVE, "abc12345")))
    assert "abc12345" in out and "99999999" in out
    assert "newest" in out.lower()
    # Nothing was decided.
    assert not (tmp_path / _APPROVAL_DECISION_FILE).exists()


def test_button_with_no_pending_cycle_explains(tmp_path: Path, monkeypatch) -> None:
    from trading.bot import keyboards
    from trading.bot.telegram import _handle_callback

    stub = _settings_stub(tmp_path)
    stub.require_cycle_approval = True
    monkeypatch.setattr(telegram_module, "settings", stub)
    out = asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_REJECT, "abc12345")))
    assert "nothing awaiting approval" in out


def test_scale_button_applies_its_percentage(tmp_path: Path, monkeypatch) -> None:
    from trading.bot import keyboards
    from trading.bot.telegram import _APPROVAL_DECISION_FILE, _handle_callback

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _pending_cycle(tmp_path, "abc12345")
    asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_APPROVE_SCALED, "abc12345:80")))
    decision = json.loads((tmp_path / _APPROVAL_DECISION_FILE).read_text())
    assert decision["action"] == "scale"
    assert decision["scale_factor"] == pytest.approx(0.8)


def test_approve_only_stages_a_bound_risk_checked_revision(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _APPROVAL_DECISION_FILE, _cmd_approve

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _pending_cycle(
        tmp_path,
        "abc12345deadbeef",
        selectable_symbols=["NVDA", "MSFT", "AMD"],
    )

    out = _cmd_approve(["only", "nvda,MSFT"])

    decision = json.loads((tmp_path / _APPROVAL_DECISION_FILE).read_text())
    assert "risk-checking" in out
    assert decision == {
        "id": "abc12345deadbeef",
        "action": "custom_request",
        "decided_at": decision["decided_at"],
        "mode": "only",
        "symbols": ["NVDA", "MSFT"],
        "plan_fingerprint": "displayed-plan",
    }


def test_approve_all_except_refuses_a_ticker_outside_the_plan(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _APPROVAL_DECISION_FILE, _cmd_approve

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _pending_cycle(tmp_path, "abc12345deadbeef", selectable_symbols=["NVDA", "MSFT"])

    out = _cmd_approve(["all", "except", "AMD"])

    assert "not a planned change" in out
    assert not (tmp_path / _APPROVAL_DECISION_FILE).exists()


def test_revised_plan_requires_its_exact_final_confirmation(tmp_path: Path, monkeypatch) -> None:
    from trading.bot import keyboards
    from trading.bot.telegram import _APPROVAL_DECISION_FILE, _cmd_approve, _handle_callback

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    _pending_cycle(
        tmp_path,
        "abc12345deadbeef",
        phase="custom_preview",
        preview_id="123456789abc",
        preview_fingerprint="preview-fingerprint",
    )

    assert "awaiting final confirmation" in _cmd_approve([])
    assert not (tmp_path / _APPROVAL_DECISION_FILE).exists()
    out = asyncio.run(
        _handle_callback(keyboards.encode(keyboards.ACT_CONFIRM_REVISED, "abc12345:123456789abc"))
    )
    assert "Confirmed revised cycle" in out
    decision = json.loads((tmp_path / _APPROVAL_DECISION_FILE).read_text())
    assert decision["action"] == "custom_confirm"
    assert decision["preview_id"] == "123456789abc"
    assert decision["preview_fingerprint"] == "preview-fingerprint"


def test_callback_cannot_run_a_command_off_the_allowlist(tmp_path: Path, monkeypatch) -> None:
    """A hostile or malformed payload must not reach a trading handler."""
    from trading.bot import keyboards
    from trading.bot.telegram import _handle_callback

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))

    def boom(*a, **k):
        raise AssertionError("dispatch must not be reached for a denied command")

    monkeypatch.setattr(telegram_module, "_dispatch", boom)
    out = asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_RUN, "/flatten")))
    assert "can't be run from a button" in out


def test_callback_from_an_older_bot_version_is_refused(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _handle_callback

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = asyncio.run(_handle_callback("0|a|abc12345"))
    assert "older version" in out


def test_stale_mode_button_refuses(tmp_path: Path, monkeypatch) -> None:
    from trading.bot import keyboards
    from trading.bot.telegram import _handle_callback, _mode_paths

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    from trading.runtime.mode import Mode, PendingModeChange, write_pending

    _, pending_path, _ = _mode_paths()
    write_pending(
        pending_path,
        PendingModeChange(
            new_mode=Mode.parse("bear"),
            requested_at=__import__("datetime")
            .datetime.now(tz=__import__("datetime").timezone.utc)
            .isoformat(),
            requested_by="test",
            reason="",
        ),
    )
    # Tapping the older "defense" preview must not apply it.
    out = asyncio.run(_handle_callback(keyboards.encode(keyboards.ACT_MODE_CONFIRM, "defense")))
    assert "defense" in out and "bear" in out


def test_mode_preview_carries_buttons(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import ButtonReply, _cmd_mode

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_mode(["defense"])
    assert isinstance(out, ButtonReply)
    labels = [b["text"] for row in out.markup["inline_keyboard"] for b in row]
    assert any("Confirm" in x for x in labels)


def test_status_carries_read_only_buttons(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import ButtonReply

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    out = _cmd_status()
    assert isinstance(out, ButtonReply)
    data = [b["callback_data"] for row in out.markup["inline_keyboard"] for b in row]
    assert all("/flatten" not in d and "/buy" not in d for d in data)


def test_unauthorized_callback_is_ignored(monkeypatch) -> None:
    """Same guard as messages: a bot added to a group must not take orders
    from anyone else in it."""
    from trading.bot.telegram import _process_callback

    answered: list[str] = []
    acted: list[str] = []

    async def fake_answer(client, token, cb_id, text=""):
        answered.append(text)

    async def fake_handle(data: str) -> str:
        acted.append(data)
        return "should not happen"

    monkeypatch.setattr(telegram_module, "_answer_callback", fake_answer)
    monkeypatch.setattr(telegram_module, "_handle_callback", fake_handle)
    cb = {"id": "1", "data": "1|a|x", "message": {"chat": {"id": "999"}, "message_id": 5}}
    asyncio.run(_process_callback(None, "tok", "123", cb))
    assert not acted
    assert answered == ["not authorized"]


def test_buttons_are_stripped_before_the_action_runs(monkeypatch) -> None:
    """Disarm first: the runner takes seconds to pick up a decision, and a
    live Approve button in that window invites a double tap."""
    from trading.bot.telegram import _process_callback

    order: list[str] = []

    async def fake_answer(client, token, cb_id, text=""):
        order.append("answer")

    async def fake_strip(client, token, chat_id, message_id):
        order.append("strip")

    async def fake_handle(data: str) -> str:
        order.append("handle")
        return "ok"

    async def fake_send(client, token, chat_id, reply):
        order.append("send")

    monkeypatch.setattr(telegram_module, "_answer_callback", fake_answer)
    monkeypatch.setattr(telegram_module, "_strip_keyboard", fake_strip)
    monkeypatch.setattr(telegram_module, "_handle_callback", fake_handle)
    monkeypatch.setattr(telegram_module, "_send_reply", fake_send)
    cb = {"id": "1", "data": "1|a|x", "message": {"chat": {"id": "123"}, "message_id": 5}}
    asyncio.run(_process_callback(None, "tok", "123", cb))
    assert order == ["answer", "strip", "handle", "send"]


def test_a_failing_button_does_not_kill_the_loop(monkeypatch) -> None:
    from trading.bot.telegram import _process_callback

    sent: list[str] = []

    async def fake_answer(client, token, cb_id, text=""):
        return None

    async def fake_strip(client, token, chat_id, message_id):
        return None

    async def fake_handle(data: str) -> str:
        raise RuntimeError("boom")

    async def fake_send(client, token, chat_id, reply):
        sent.append(str(reply))

    monkeypatch.setattr(telegram_module, "_answer_callback", fake_answer)
    monkeypatch.setattr(telegram_module, "_strip_keyboard", fake_strip)
    monkeypatch.setattr(telegram_module, "_handle_callback", fake_handle)
    monkeypatch.setattr(telegram_module, "_send_reply", fake_send)
    cb = {"id": "1", "data": "1|a|x", "message": {"chat": {"id": "123"}, "message_id": 5}}
    asyncio.run(_process_callback(None, "tok", "123", cb))  # must not raise
    assert sent and "RuntimeError" in sent[0]


# ---------------------------------------------------------------------------
# Batch 5 — mandates, reply-to-alert, accuracy discipline
# ---------------------------------------------------------------------------


def test_mandate_is_captured_without_an_llm_call(tmp_path: Path, monkeypatch) -> None:
    """A standing instruction is stored, not answered — and costs nothing."""
    stub = _settings_stub(tmp_path)
    stub.data_dir = tmp_path / "data"
    monkeypatch.setattr(telegram_module, "settings", stub)

    async def boom(*a, **k):
        raise AssertionError("copilot must not be called for a mandate")

    monkeypatch.setattr(telegram_module, "_cmd_copilot", boom)
    out = asyncio.run(_dispatch("high conviction on GS, include it next round")) or ""
    assert "Noted for the next run" in out

    from trading.copilot.mandates import MandateStore

    active = MandateStore(tmp_path).active()
    assert len(active) == 1 and "GS" in active[0].symbols


def test_mandate_echoes_the_strength_it_was_read_as(tmp_path: Path, monkeypatch) -> None:
    """Tone parsing is deterministic but not infallible; the moment to
    catch a misread is while the operator is still looking at the screen."""
    stub = _settings_stub(tmp_path)
    stub.data_dir = tmp_path / "data"
    monkeypatch.setattr(telegram_module, "settings", stub)
    out = asyncio.run(_dispatch("I want GS in the book next round")) or ""
    assert "strong" in out
    assert "/soften" in out and "/harden" in out
    assert "can't lift a position or cluster limit" in out


def test_soft_and_strong_get_different_promises(tmp_path: Path, monkeypatch) -> None:
    stub = _settings_stub(tmp_path)
    stub.data_dir = tmp_path / "data"
    monkeypatch.setattr(telegram_module, "settings", stub)
    strong = asyncio.run(_dispatch("I want GS next round")) or ""
    soft = asyncio.run(_dispatch("consider XLE next round")) or ""
    assert "unless there's a concrete reason not to" in strong
    assert "may drop it" in soft


def test_questions_still_reach_the_copilot(tmp_path: Path, monkeypatch) -> None:
    """The mandate check must not swallow ordinary questions."""
    stub = _settings_stub(tmp_path)
    stub.data_dir = tmp_path / "data"
    monkeypatch.setattr(telegram_module, "settings", stub)
    seen: dict[str, str] = {}

    async def fake(question: str, symbol: str | None = None, **kw) -> str:
        seen["q"] = question
        return "answered"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake)
    assert asyncio.run(_dispatch("did the committee consider GS?")) == "answered"
    assert "GS" in seen["q"]


def test_mandates_command_lists_and_drops(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_mandates
    from trading.copilot.mandates import MandateStore

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    assert "no standing instructions" in _cmd_mandates([])
    m = MandateStore(tmp_path).add("I want GS next round", symbols=["GS"])
    listed = _cmd_mandates([])
    assert m.id in listed and "strong" in listed
    assert "dropped" in _cmd_mandates(["drop", m.id])
    assert "no standing instructions" in _cmd_mandates([])


def test_soften_and_harden_regrade(tmp_path: Path, monkeypatch) -> None:
    from trading.bot.telegram import _cmd_restrength
    from trading.copilot.mandates import MandateStore

    monkeypatch.setattr(telegram_module, "settings", _settings_stub(tmp_path))
    m = MandateStore(tmp_path).add("consider GS next round")
    out = _cmd_restrength([m.id], "strong")
    assert "now *strong*" in out
    assert MandateStore(tmp_path).active()[0].strength == "strong"
    assert "no active mandate" in _cmd_restrength(["M999"], "soft")


def test_reply_to_an_alert_passes_it_as_context(monkeypatch) -> None:
    """'why this alert?' is only answerable if we know which alert."""
    seen: dict[str, str] = {}

    async def fake(question: str, symbol: str | None = None, **kw) -> str:
        seen["replied_to"] = kw.get("replied_to") or ""
        return "answered"

    monkeypatch.setattr(telegram_module, "_cmd_copilot", fake)
    alert = "🟠 Sentinel — CAUTION\nTripped: SPY -1.6%; INTC -5.0% (held)"
    out = asyncio.run(_dispatch("why this alert? elaborate please", replied_to=alert))
    assert out == "answered"
    assert "INTC" in seen["replied_to"]


def test_replied_to_reaches_the_copilot_evidence(tmp_path: Path, monkeypatch) -> None:
    import trading.copilot as copilot_pkg

    stub = _settings_stub(tmp_path)
    stub.data_dir = tmp_path / "data"
    monkeypatch.setattr(telegram_module, "settings", stub)
    captured: dict = {}

    def fake_answer(q, **kw):
        captured.update(kw)
        return "ok"

    monkeypatch.setattr(copilot_pkg, "answer", fake_answer)
    asyncio.run(_dispatch("why this alert?", replied_to="Sentinel CAUTION on INTC"))
    assert "INTC" in (captured.get("replied_to") or "")
