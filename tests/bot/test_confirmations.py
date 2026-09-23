"""Two-step confirmation for /flatten, /resume and large manual orders."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from trading.bot import confirmations as cf
from trading.bot import keyboards
from trading.bot import telegram as tg
from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position
from trading.runner.state import RunnerStore

T0 = datetime(2026, 9, 28, 15, tzinfo=timezone.utc)


def _stub(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state_dir=tmp_path,
        data_dir=tmp_path / "parquet",
        trading_env="research",
        is_live_armed=lambda: False,
        manual_order_confirm_pct=0.05,
        manual_order_max_pct=0.50,
    )


def _pending_commands(tmp_path: Path) -> list[dict]:
    d = tmp_path / "commands" / "pending"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def _snapshot(tmp_path: Path, *, equity: float = 100_000.0) -> None:
    pos = Position(
        instrument=Instrument(symbol="NVDA", asset_class=AssetClass.EQUITY),
        quantity=40,
        avg_price=150.0,
        unrealized_pnl=40 * 30.0,  # mark 180
    )
    RunnerStore(tmp_path / "runner.db").save_snapshot(
        AccountSnapshot(
            ts=T0, cash=equity - 7200, equity=equity, positions={pos.instrument.key: pos}
        )
    )


@pytest.fixture
def bot(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(tg, "settings", _stub(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- the slot


def test_take_requires_the_matching_token_and_clears_on_success(tmp_path: Path) -> None:
    staged = cf.stage(tmp_path, "flatten", {}, "flatten every position", now=T0)
    assert isinstance(cf.take(tmp_path, "WRONG", now=T0), str)
    assert cf.peek(tmp_path) is not None  # a mismatch never cancels the real one
    assert cf.take(tmp_path, staged.token.lower(), now=T0) == staged
    assert cf.peek(tmp_path) is None


def test_expired_confirmation_is_refused_and_cleared(tmp_path: Path) -> None:
    staged = cf.stage(tmp_path, "resume", {}, "resume", now=T0)
    out = cf.take(tmp_path, staged.token, now=T0 + cf.TTL + timedelta(seconds=1))
    assert isinstance(out, str) and "expired" in out
    assert cf.peek(tmp_path) is None


def test_new_stage_replaces_old(tmp_path: Path) -> None:
    first = cf.stage(tmp_path, "flatten", {}, "a", now=T0)
    second = cf.stage(tmp_path, "resume", {}, "b", now=T0)
    assert isinstance(cf.take(tmp_path, first.token, now=T0), str)
    assert cf.take(tmp_path, second.token, now=T0) == second


def test_unknown_kinds_cannot_be_staged(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cf.stage(tmp_path, "halt", {}, "no")


@pytest.mark.parametrize(
    ("notional", "equity", "expected"),
    [
        (1_000, 100_000, "direct"),
        (5_000, 100_000, "confirm"),
        (60_000, 100_000, "refuse"),
        (None, 100_000, "confirm"),
        (1_000, None, "confirm"),
        (1_000, 0.0, "confirm"),
    ],
)
def test_size_classification(notional, equity, expected) -> None:
    decision, _share = cf.needs_confirmation(notional, equity, confirm_pct=0.05, max_pct=0.5)
    assert decision == expected


# ---------------------------------------------------------------- the bot


def test_flatten_stages_and_only_confirm_queues_it(bot: Path) -> None:
    _snapshot(bot)
    out = tg._cmd_flatten()
    assert "Flatten everything" in out and "1 position" in out
    assert _pending_commands(bot) == []
    staged = cf.peek(bot)
    assert staged is not None and staged.kind == "flatten"

    reply = tg._cmd_confirm([staged.token])
    assert "confirmed" in reply
    assert [c["type"] for c in _pending_commands(bot)] == ["flatten"]


def test_small_buy_goes_straight_through(bot: Path) -> None:
    _snapshot(bot)
    assert tg._cmd_buy(["AAPL", "10", "150"]) is None  # 1.5% of equity
    assert [c["type"] for c in _pending_commands(bot)] == ["buy"]


def test_large_buy_stages(bot: Path) -> None:
    _snapshot(bot)
    out = tg._cmd_buy(["AAPL", "50", "150"])  # 7.5%
    assert "Confirm" in out and "7.5%" in out
    assert _pending_commands(bot) == []


def test_huge_buy_is_refused_outright(bot: Path) -> None:
    _snapshot(bot)
    out = tg._cmd_buy(["AAPL", "400", "150"])  # 60%
    assert out.startswith("❌ refused") and "MANUAL_ORDER_MAX_PCT" in out
    assert _pending_commands(bot) == [] and cf.peek(bot) is None


def test_unsized_buy_is_treated_as_large(bot: Path) -> None:
    _snapshot(bot)
    out = tg._cmd_buy(["ZZZZ", "3"])  # no limit, no position, no cached close
    assert "size unknown" in out
    assert _pending_commands(bot) == []


def test_sell_all_uses_the_position_value(bot: Path) -> None:
    _snapshot(bot, equity=100_000)
    out = tg._cmd_sell(["NVDA", "all"])  # 40 x 180 = 7,200 = 7.2%
    assert "7.2%" in out
    out2 = tg._cmd_close(["NVDA"])
    assert "7.2%" in out2  # restaging replaces; still not queued
    assert _pending_commands(bot) == []


def test_resume_when_halted_needs_confirmation(bot: Path) -> None:
    tg._cmd_halt(["loss limit"])
    out = tg._cmd_resume()
    assert "loss limit" in out
    assert json.loads((bot / "halt.json").read_text())["halted"] is True


def test_cancel_discards_the_staged_command(bot: Path) -> None:
    tg._cmd_flatten()
    out = tg._cmd_cancel()
    assert "cancelled" in out and cf.peek(bot) is None


def test_bare_confirm_refuses_to_guess_between_mode_and_command(bot: Path) -> None:
    tg._cmd_mode(["defense"])
    tg._cmd_flatten()
    out = tg._cmd_confirm([])
    assert "two things are waiting" in out
    assert _pending_commands(bot) == []


def test_bare_confirm_with_only_a_command_confirms_it(bot: Path) -> None:
    tg._cmd_flatten()
    tg._cmd_confirm([])
    assert [c["type"] for c in _pending_commands(bot)] == ["flatten"]


def test_button_is_bound_to_its_token(bot: Path) -> None:
    tg._cmd_flatten()
    stale = cf.peek(bot)
    tg._cmd_resume()  # not halted -> no stage; flatten still staged
    tg._cmd_halt(["x"])
    tg._cmd_resume()  # replaces the flatten
    out = asyncio.run(tg._handle_callback(keyboards.encode(keyboards.ACT_CMD_CONFIRM, stale.token)))
    assert "not the current one" in out
    assert _pending_commands(bot) == []
    current = cf.peek(bot)
    out = asyncio.run(
        tg._handle_callback(keyboards.encode(keyboards.ACT_CMD_CONFIRM, current.token))
    )
    assert "RESUMED" in out


def test_halt_is_never_staged(bot: Path) -> None:
    out = tg._cmd_halt(["now"])
    assert "HALTED" in out and cf.peek(bot) is None
