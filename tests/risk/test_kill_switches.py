"""Intraday kill switches: daily-loss and peak-drawdown halts."""

from __future__ import annotations

from datetime import timedelta

from trading.core.types import AccountSnapshot


def test_daily_loss_kill_switch_fires(mgr, account_100k, t0) -> None:
    mgr.start_of_day(account_100k)
    # Drop equity 3% from open with limit at 2%.
    bad = account_100k.model_copy(update={"cash": 95_000.0, "equity": 97_000.0})
    decision = mgr.evaluate_intraday(bad)
    assert decision.action == "halt"
    assert "daily loss" in decision.reason
    assert mgr.is_halted()


def test_daily_loss_below_threshold_passes(mgr, account_100k, t0) -> None:
    mgr.start_of_day(account_100k)
    ok = account_100k.model_copy(update={"cash": 99_000.0, "equity": 99_000.0})
    decision = mgr.evaluate_intraday(ok)
    assert decision.action == "allow"
    assert not mgr.is_halted()


def test_drawdown_kill_switch_fires(mgr, t0) -> None:
    mgr.limits = mgr.limits.model_copy(update={"max_daily_loss_pct": 1.0})  # disable daily-loss
    # Start with high equity to establish HWM.
    high = AccountSnapshot(ts=t0, cash=200_000.0, equity=200_000.0)
    mgr.start_of_day(high)
    mgr.evaluate_intraday(high)  # records HWM
    # Now drop 16% from peak.
    low = AccountSnapshot(ts=t0 + timedelta(days=30), cash=168_000.0, equity=168_000.0)
    mgr.start_of_day(low)
    decision = mgr.evaluate_intraday(low)
    assert decision.action == "halt"
    assert "drawdown" in decision.reason


def test_drawdown_below_threshold_passes(mgr, t0) -> None:
    mgr.limits = mgr.limits.model_copy(update={"max_daily_loss_pct": 1.0})
    high = AccountSnapshot(ts=t0, cash=200_000.0, equity=200_000.0)
    mgr.start_of_day(high)
    mgr.evaluate_intraday(high)
    # Drop 10% — below the 15% drawdown limit.
    low = AccountSnapshot(ts=t0 + timedelta(days=10), cash=180_000.0, equity=180_000.0)
    mgr.start_of_day(low)
    decision = mgr.evaluate_intraday(low)
    assert decision.action == "allow"
    assert not mgr.is_halted()


def test_start_of_day_idempotent_within_day(mgr, account_100k, t0) -> None:
    mgr.start_of_day(account_100k)
    # Equity changes during the day — start_of_day must NOT reset the open value.
    mid = account_100k.model_copy(update={"equity": 105_000.0})
    mgr.start_of_day(mid)
    assert mgr.state.daily_equity_open == 100_000.0
    # New day → it gets re-stamped.
    next_day = mid.model_copy(update={"ts": t0.replace(day=2)})
    mgr.start_of_day(next_day)
    assert mgr.state.daily_equity_open == 105_000.0


def test_already_halted_evaluation_returns_halt(mgr, account_100k) -> None:
    mgr.halt("test halt")
    decision = mgr.evaluate_intraday(account_100k)
    assert decision.action == "halt"
    assert "already halted" in decision.reason


def test_unhalt_clears_state(mgr, account_100k) -> None:
    mgr.halt("test halt")
    mgr.unhalt()
    assert not mgr.is_halted()
    decision = mgr.evaluate_intraday(account_100k)
    assert decision.action == "allow"


def test_hwm_updates_on_new_high(mgr, t0) -> None:
    mgr.limits = mgr.limits.model_copy(update={"max_daily_loss_pct": 1.0})
    a = AccountSnapshot(ts=t0, cash=100_000.0, equity=100_000.0)
    mgr.start_of_day(a)
    mgr.evaluate_intraday(a)
    higher = a.model_copy(update={"equity": 120_000.0})
    mgr.evaluate_intraday(higher)
    assert mgr.state.equity_high_watermark == 120_000.0
    # A subsequent dip below the new HWM doesn't lower it.
    dip = higher.model_copy(update={"equity": 110_000.0})
    mgr.evaluate_intraday(dip)
    assert mgr.state.equity_high_watermark == 120_000.0
