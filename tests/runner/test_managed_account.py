"""A pinned position is invisible, not merely untouchable.

`/hold NVDA` used to mean only "do not trade NVDA" — the position still
counted toward the equity every order was sized against and toward the
equity both kill switches measured. On 2026-08-18 that halted the desk
over a CHF 3,677 drawdown that was almost entirely in pinned personal
names it was forbidden to touch, and then refused the trailing-stop exits
that would have protected the part it *was* running.

Two things have to hold, and the second is the one that bites:

  * the value leaves equity, so sizing and the kill switches both work on
    the desk's own book;
  * the cash stays, because all of it is the desk's to deploy.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Position,
)
from trading.runner.managed_account import managed_view

NOW = datetime(2026, 8, 22, 16, 0, tzinfo=timezone.utc)
USDCHF = 0.80


def inst(symbol: str, *, currency: str = "USD") -> Instrument:
    return Instrument(
        symbol=symbol, asset_class=AssetClass.EQUITY, currency=currency, exchange="SMART"
    )


def pos(symbol: str, qty: float, price: float, *, currency: str = "USD") -> Position:
    return Position(instrument=inst(symbol, currency=currency), quantity=qty, avg_price=price)


def account(*positions: Position, equity: float = 84_312.0, cash: float = 56_225.0):
    return AccountSnapshot(
        ts=NOW,
        cash=cash,
        equity=equity,
        positions={p.instrument.key: p for p in positions},
        base_currency="CHF",
        cash_by_currency={"CHF": 25_011.0, "USD": 39_010.0},
    )


class TestAHeldPositionLeavesTheBook:
    def test_it_is_removed_from_positions(self) -> None:
        view = managed_view(
            account(pos("NVDA", 40, 216.0), pos("AMD", 8, 462.0)),
            {"NVDA"},
            fx_rates={"USD": USDCHF},
        )

        assert [p.instrument.symbol for p in view.account.positions.values()] == ["AMD"]

    def test_its_value_leaves_equity_converted(self) -> None:
        """40 NVDA at USD 216 is USD 8,640 = CHF 6,912 — not CHF 8,640."""
        view = managed_view(account(pos("NVDA", 40, 216.0)), {"NVDA"}, fx_rates={"USD": USDCHF})

        assert view.excluded["NVDA"] == pytest.approx(6_912.0)
        assert view.account.equity == pytest.approx(84_312.0 - 6_912.0)

    def test_the_cash_is_untouched(self) -> None:
        view = managed_view(account(pos("NVDA", 40, 216.0)), {"NVDA"}, fx_rates={"USD": USDCHF})

        assert view.account.cash == 56_225.0
        assert view.account.cash_by_currency == {"CHF": 25_011.0, "USD": 39_010.0}

    def test_the_snapshot_says_which_book_it_describes(self) -> None:
        view = managed_view(account(pos("NVDA", 40, 216.0)), {"NVDA"}, fx_rates={"USD": USDCHF})

        assert view.account.scope == "managed"
        assert view.account.excluded_symbols == ("NVDA",)
        assert view.account.excluded_value == pytest.approx(6_912.0)

    def test_a_live_mark_is_preferred_over_cost_plus_pnl(self) -> None:
        held = pos("NVDA", 40, 216.0)
        view = managed_view(
            account(held),
            {"NVDA"},
            fx_rates={"USD": USDCHF},
            last_prices={held.instrument.key: 250.0},
        )

        assert view.excluded["NVDA"] == pytest.approx(40 * 250.0 * USDCHF)

    def test_matching_is_case_insensitive(self) -> None:
        view = managed_view(account(pos("NVDA", 40, 216.0)), {"nvda"}, fx_rates={"USD": USDCHF})

        assert view.changed


class TestNothingHeldChangesNothing:
    def test_no_holds_returns_the_same_account(self) -> None:
        snap = account(pos("AMD", 8, 462.0))

        assert managed_view(snap, set()).account is snap

    def test_a_hold_on_a_name_not_owned_changes_nothing(self) -> None:
        snap = account(pos("AMD", 8, 462.0))

        assert managed_view(snap, {"TSLA"}, fx_rates={"USD": USDCHF}).account is snap

    def test_an_already_managed_snapshot_is_not_stripped_twice(self) -> None:
        """Double application would subtract the same value again and
        silently halve the sizing base."""
        once = managed_view(
            account(pos("NVDA", 40, 216.0)), {"NVDA"}, fx_rates={"USD": USDCHF}
        ).account

        twice = managed_view(once, {"NVDA"}, fx_rates={"USD": USDCHF}).account

        assert twice is once
        assert twice.equity == pytest.approx(84_312.0 - 6_912.0)


class TestItRefusesToProduceNonsense:
    def test_a_pinned_book_larger_than_the_account_is_left_alone(self) -> None:
        """Only bad data can do this, and sizing against zero or negative
        equity is worse than sizing against too much."""
        snap = account(pos("NVDA", 40, 216.0), equity=5_000.0)

        view = managed_view(snap, {"NVDA"}, fx_rates={"USD": USDCHF})

        assert view.account is snap
        assert not view.changed

    def test_a_missing_fx_rate_is_reported_not_hidden(self) -> None:
        """Passing a USD number through as CHF understates the deduction
        by a fifth, which overstates the sizing base by the same. The old
        code did this everywhere and said nothing."""
        view = managed_view(account(pos("NVDA", 40, 216.0)), {"NVDA"}, fx_rates={})

        assert view.unconverted == ("NVDA",)
        assert "No FX rate" in view.note("CHF")

    def test_a_base_currency_position_needs_no_rate(self) -> None:
        view = managed_view(account(pos("NESN", 100, 90.0, currency="CHF")), {"NESN"}, fx_rates={})

        assert view.unconverted == ()
        assert view.excluded["NESN"] == pytest.approx(9_000.0)


class TestTheOperatorNote:
    def test_it_names_the_symbols_and_the_desk_equity(self) -> None:
        view = managed_view(
            account(pos("NVDA", 40, 216.0), pos("GEV", 9, 954.0)),
            {"NVDA", "GEV"},
            fx_rates={"USD": USDCHF},
        )

        note = view.note("CHF")

        assert "`GEV`" in note and "`NVDA`" in note
        assert f"{view.account.equity:,.0f}" in note


class TestTheKillSwitchesFollowTheScope:
    """The half that actually caused the 18 Aug halt."""

    def _manager(self, tmp_path):
        from trading.risk.limits import RiskLimits
        from trading.risk.manager import RiskManager

        return RiskManager(
            RiskLimits(max_drawdown_pct=0.05, max_daily_loss_pct=0.015),
            halt_state_path=tmp_path / "halt.json",
        )

    def test_switching_scope_re_stamps_instead_of_halting(self, tmp_path) -> None:
        """Managed equity against an account-scoped high-water mark reads
        as a 33% drawdown on an account that has not moved."""
        manager = self._manager(tmp_path)
        whole = account(pos("NVDA", 40, 216.0))
        manager.start_of_day(whole)
        assert manager._state.equity_high_watermark == pytest.approx(84_312.0)

        desk = managed_view(whole, {"NVDA"}, fx_rates={"USD": USDCHF}).account
        note = manager._reconcile_baseline_scope(desk)

        assert note is not None and "not a loss" in note
        assert manager._state.equity_high_watermark == pytest.approx(desk.equity)
        assert manager._state.daily_equity_open == pytest.approx(desk.equity)
        assert manager._state.baseline_scope == "managed"

    def test_a_legacy_state_is_not_treated_as_a_scope_change(self, tmp_path) -> None:
        """`baseline_scope` did not exist before 2026-08-22. Every stored
        state is account-scoped by construction, and re-stamping them all
        on first read would announce a migration that never happened."""
        manager = self._manager(tmp_path)
        whole = account(pos("AMD", 8, 462.0))
        manager.start_of_day(whole)
        manager._state = manager._state.replace(baseline_scope=None)

        assert manager._reconcile_baseline_scope(whole) is None
        assert manager._state.baseline_scope == "account"

    def test_it_re_stamps_once_not_every_evaluation(self, tmp_path) -> None:
        manager = self._manager(tmp_path)
        whole = account(pos("NVDA", 40, 216.0))
        manager.start_of_day(whole)
        desk = managed_view(whole, {"NVDA"}, fx_rates={"USD": USDCHF}).account

        assert manager._reconcile_baseline_scope(desk) is not None
        assert manager._reconcile_baseline_scope(desk) is None
        assert manager._reconcile_baseline_scope(desk) is None


class TestTheOperatorViewDoesNotMixTheTwoBooks:
    def test_baseline_quotes_percentages_against_the_desk_not_the_account(self) -> None:
        """The stored snapshot is the whole account; the baselines may be
        the desk's. Dividing one by the other reports a spectacular gain
        on an account that has not moved."""
        import inspect

        from trading.bot import telegram

        source = inspect.getsource(telegram._cmd_baseline)

        assert "desk_equity" in source
        for metric in ("drawdown vs peak", "day P&L vs stored open"):
            line = next(ln for ln in source.splitlines() if metric in ln)
            index = source.index(line)
            preceding = source[max(0, index - 400) : index]
            assert "desk_equity -" in preceding, metric

    def test_it_says_which_book_it_is_measuring(self) -> None:
        import inspect

        from trading.bot import telegram

        source = inspect.getsource(telegram._cmd_baseline)

        assert "the desk's book" in source
        assert "the whole account" in source
