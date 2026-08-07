"""The kill-switch baseline must describe THIS account.

Reconstructs 2026-08-07: a paper cycle stamped CHF 1,068,862 into the
live state directory's ``daily_equity_open`` and ``equity_high_watermark``.
An hour later the live session opened with 87,413 of equity, ``start_of_day``
no-oped because the date already matched, and the daily-loss switch fired
at -91.82%. Clearing the daily figure alone was not enough either — the
poisoned high-water mark re-halted on drawdown immediately.
"""

from __future__ import annotations

from datetime import timedelta

from trading.core.types import AccountSnapshot

PAPER_EQUITY = 1_068_862.0
LIVE_EQUITY = 87_413.0


def _poison(mgr, ts) -> None:
    """Stamp the day with a foreign account's equity, as the paper cycle did."""
    mgr.start_of_day(AccountSnapshot(ts=ts, cash=PAPER_EQUITY, equity=PAPER_EQUITY))


def test_the_2026_08_07_sequence_no_longer_halts(mgr, t0) -> None:
    _poison(mgr, t0)
    live = AccountSnapshot(ts=t0 + timedelta(hours=1), cash=LIVE_EQUITY, equity=LIVE_EQUITY)

    decision = mgr.evaluate_intraday(live)

    assert decision.action != "halt", decision.reason
    assert not mgr.is_halted()


def test_both_figures_are_reset_not_just_the_daily_open(mgr, t0) -> None:
    """The high-water mark is the half that bites second: at a 15% drawdown
    limit, 87k against a 1.07M peak re-halts the instant the daily figure
    is cleared. ``max()`` would have preserved the bogus peak."""
    _poison(mgr, t0)
    live = AccountSnapshot(ts=t0 + timedelta(hours=1), cash=LIVE_EQUITY, equity=LIVE_EQUITY)

    mgr.start_of_day(live)

    assert mgr.state.daily_equity_open == LIVE_EQUITY
    assert mgr.state.equity_high_watermark == LIVE_EQUITY


def test_the_repair_is_reported_not_silent(mgr, t0) -> None:
    _poison(mgr, t0)
    live = AccountSnapshot(ts=t0 + timedelta(hours=1), cash=LIVE_EQUITY, equity=LIVE_EQUITY)

    note = mgr.start_of_day(live)

    assert note is not None
    assert "1,068,862" in note and "87,413" in note


def test_the_note_is_read_once(mgr, t0) -> None:
    """Otherwise the runner alerts every cycle forever after one repair."""
    _poison(mgr, t0)
    live = AccountSnapshot(ts=t0 + timedelta(hours=1), cash=LIVE_EQUITY, equity=LIVE_EQUITY)
    mgr.start_of_day(live)

    assert mgr.take_baseline_note() is not None
    assert mgr.take_baseline_note() is None


def test_a_real_loss_still_halts(mgr, account_100k) -> None:
    """The whole point of the limit is that it only catches the absurd. A
    3% day against a 2% limit is exactly what the kill switch is for."""
    mgr.start_of_day(account_100k)
    bad = account_100k.model_copy(update={"cash": 97_000.0, "equity": 97_000.0})

    decision = mgr.evaluate_intraday(bad)

    assert decision.action == "halt"
    assert "daily loss" in decision.reason


def test_a_large_but_plausible_drop_is_left_alone(mgr, account_100k) -> None:
    """40% is a catastrophe, not a configuration error. Below the 50%
    sanity limit the manager must not second-guess the baseline."""
    mgr.start_of_day(account_100k)
    crash = account_100k.model_copy(update={"cash": 60_000.0, "equity": 60_000.0})

    decision = mgr.evaluate_intraday(crash)

    assert decision.action == "halt"
    assert mgr.state.daily_equity_open == 100_000.0


def test_same_day_repeat_calls_still_no_op(mgr, account_100k) -> None:
    """Idempotency within a day is preserved for the ordinary case — the
    exception is only for an implausible baseline."""
    mgr.start_of_day(account_100k)
    later = account_100k.model_copy(update={"cash": 101_000.0, "equity": 101_000.0})

    assert mgr.start_of_day(later) is None
    assert mgr.state.daily_equity_open == 100_000.0


def test_divergence_is_symmetric(mgr, account_100k, t0) -> None:
    """A deposit that multiplies the account is as much a broken baseline
    as a foreign account's figure — same repair, same alert."""
    mgr.start_of_day(account_100k)
    funded = AccountSnapshot(ts=t0, cash=500_000.0, equity=500_000.0)

    note = mgr.start_of_day(funded)

    assert note is not None
    assert mgr.state.daily_equity_open == 500_000.0


def test_no_baseline_yet_is_not_a_divergence(mgr, account_100k) -> None:
    assert mgr.baseline_divergence(account_100k.equity) is None
    assert mgr.start_of_day(account_100k) is None


def test_threshold_is_configurable(limits, tmp_path, t0) -> None:
    from trading.risk import RiskManager

    strict = RiskManager(
        limits.model_copy(update={"baseline_sanity_divergence_pct": 0.10}),
        halt_state_path=tmp_path / "halt.json",
    )
    strict.start_of_day(AccountSnapshot(ts=t0, cash=100_000.0, equity=100_000.0))

    note = strict.start_of_day(AccountSnapshot(ts=t0, cash=80_000.0, equity=80_000.0))

    assert note is not None
