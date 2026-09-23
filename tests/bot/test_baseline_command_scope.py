r"""`/baseline reset` must re-stamp the book the kill switches measure.

2026-09-14, live. Seven positions were pinned with `/hold`, so the switches
measured the MANAGED book — CHF 45,578 — while the account held CHF 83,773.
`_cmd_baseline` computed the desk figure correctly, used it for every
percentage it printed, and then passed the *account* figure to the write.

    daily open 83,773 on a book worth 45,578  =  -45.59%  against a -0.60% limit

Permanent, and immune to the command meant to clear it: each re-run wrote the
same wrong number. The desk sat locked with zero positions and a fake loss.

This file pins the contract at the call site. The refusal inside
`reset_equity_baseline` is covered in tests/risk/test_baseline_reset.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import trading.bot.telegram as telegram_module
from trading.core.types import AccountSnapshot
from trading.risk.halt_file import write_halt_state
from trading.risk.limits import HaltState

NOW = datetime(2026, 9, 14, 17, 10, tzinfo=timezone.utc)
ACCOUNT_EQUITY = 83_772.99
DESK_EQUITY = 45_577.83
PINNED = ("AVAV", "CEG", "GEV", "GLD", "NVDA", "QNT", "SHA")


@pytest.fixture
def desk(monkeypatch, tmp_path: Path):
    """A live desk with pinned positions and a managed-scope baseline."""
    write_halt_state(
        tmp_path,
        HaltState(
            equity_high_watermark=45_608.82,
            daily_equity_open=45_534.40,
            daily_baseline_currency="CHF",
            baseline_scope="managed",
        ),
    )
    monkeypatch.setattr(
        telegram_module,
        "settings",
        SimpleNamespace(state_dir=tmp_path, trading_env="live", is_live_armed=lambda: True),
    )
    snap = AccountSnapshot(ts=NOW, cash=ACCOUNT_EQUITY, equity=ACCOUNT_EQUITY, base_currency="CHF")
    monkeypatch.setattr(telegram_module, "_latest_account_snapshot", lambda: snap)

    import trading.runner.holds as holds_module
    import trading.runner.managed_account as managed_module

    monkeypatch.setattr(holds_module, "load_holds", lambda *a, **kw: set(PINNED))
    monkeypatch.setattr(
        managed_module,
        "managed_view",
        lambda *a, **kw: SimpleNamespace(
            changed=True,
            excluded=set(PINNED),
            account=snap.model_copy(update={"equity": DESK_EQUITY, "scope": "managed"}),
        ),
    )

    captured: dict[str, object] = {}

    def _spy(state_dir, **kwargs):
        captured.update(kwargs)
        before = HaltState(equity_high_watermark=45_608.82, daily_equity_open=45_534.40)
        return before, before.replace(
            equity_high_watermark=float(kwargs["equity"]),
            daily_equity_open=float(kwargs["equity"]),
        )

    import trading.risk.halt_file as halt_module

    monkeypatch.setattr(halt_module, "reset_equity_baseline", _spy)
    return captured


def test_the_reset_is_stamped_with_desk_equity_not_account_equity(desk) -> None:
    """The regression, in the exact shape it occurred."""
    telegram_module._cmd_baseline(["reset"])

    assert desk["equity"] == pytest.approx(DESK_EQUITY)
    assert desk["equity"] != pytest.approx(ACCOUNT_EQUITY)


def test_the_stored_scope_is_passed_through_so_the_write_can_refuse(desk) -> None:
    """Belt and braces: even if a future caller regresses the figure, the
    scope tag lets `reset_equity_baseline` catch it."""
    telegram_module._cmd_baseline(["reset"])

    assert desk["scope"] == "managed"


def test_the_confirmation_quotes_what_was_actually_written(desk) -> None:
    """Printing the account figure while writing the desk figure would have
    hidden this bug just as effectively as the bug itself."""
    out = telegram_module._cmd_baseline(["reset"])

    assert f"{DESK_EQUITY:,.2f}" in out
    assert f"{ACCOUNT_EQUITY:,.2f}" not in out


def test_failed_managed_valuation_never_resets_against_account(desk, monkeypatch) -> None:
    import trading.runner.managed_account as managed_module

    def broken(*args, **kwargs):
        raise ValueError("missing FX")

    monkeypatch.setattr(managed_module, "managed_view", broken)
    out = telegram_module._cmd_baseline(["reset"])
    assert "nothing was reset" in out
    assert desk == {}


def test_unknown_managed_baseline_does_not_publish_false_returns(desk) -> None:
    out = telegram_module._cmd_baseline([])
    assert "returns unavailable" in out
    assert "drawdown vs peak" not in out


def test_an_unpinned_account_still_resets_against_the_whole_account(
    monkeypatch, tmp_path: Path
) -> None:
    """`desk_equity` equals `equity` whenever nothing is pinned — the fix
    must not change the ordinary case."""
    write_halt_state(
        tmp_path,
        HaltState(
            equity_high_watermark=90_000.0,
            daily_equity_open=88_000.0,
            daily_baseline_currency="CHF",
        ),
    )
    monkeypatch.setattr(
        telegram_module,
        "settings",
        SimpleNamespace(state_dir=tmp_path, trading_env="live", is_live_armed=lambda: True),
    )
    snap = AccountSnapshot(ts=NOW, cash=ACCOUNT_EQUITY, equity=ACCOUNT_EQUITY, base_currency="CHF")
    monkeypatch.setattr(telegram_module, "_latest_account_snapshot", lambda: snap)

    captured: dict[str, object] = {}

    def _spy(state_dir, **kwargs):
        captured.update(kwargs)
        before = HaltState(equity_high_watermark=90_000.0, daily_equity_open=88_000.0)
        return before, before

    import trading.risk.halt_file as halt_module

    monkeypatch.setattr(halt_module, "reset_equity_baseline", _spy)

    telegram_module._cmd_baseline(["reset"])

    assert captured["equity"] == pytest.approx(ACCOUNT_EQUITY)
    assert captured["scope"] == "account"
