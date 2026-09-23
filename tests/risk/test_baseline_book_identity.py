"""Pinned capital transfers must never masquerade as investment returns."""

from datetime import datetime, timedelta, timezone

import pytest

from trading.core.types import AccountSnapshot
from trading.risk.halt_file import (
    BaselineResetError,
    read_halt_state,
    reset_equity_baseline,
    write_halt_state,
)
from trading.risk.limits import RiskLimits
from trading.risk.manager import RiskManager

OPEN = datetime(2026, 9, 18, 13, 31, tzinfo=timezone.utc)


def snapshot(**updates):
    return AccountSnapshot(
        ts=OPEN,
        equity=90_000,
        cash=50_000,
        base_currency="CHF",
        scope="managed",
        excluded_symbols=("NVDA",),
        excluded_quantities={"equity:NVDA": 40.0},
    ).model_copy(update=updates)


def captured(tmp_path):
    manager = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
    manager.capture_session_open(snapshot(), session_date=OPEN.date(), captured_at=OPEN)
    assert manager.state.baseline_book_identity == snapshot().risk_book_identity
    return manager


@pytest.mark.parametrize(
    "updates",
    [
        {
            "excluded_symbols": ("NVDA", "GEV"),
            "excluded_quantities": {"equity:NVDA": 40.0, "equity:GEV": 3.0},
        },
        {"excluded_quantities": {"equity:NVDA": 50.0}},
        {"scope": "account", "excluded_symbols": (), "excluded_quantities": {}},
    ],
)
def test_transfer_blocks_execution_without_fabricating_loss_or_erasing_references(
    tmp_path, updates
):
    manager = captured(tmp_path)
    before = manager.state
    changed = snapshot(equity=70_000, **updates)
    result = manager.evaluate_session_risk(changed, session_label=OPEN.date())
    assert result.action == "reject"
    assert "baseline" in result.reason
    assert manager.state == before
    # Tomorrow's opening print cannot silently erase the old peak either.
    monday = OPEN + timedelta(days=3)
    manager.capture_session_open(
        changed.model_copy(update={"ts": monday}), session_date=monday.date(), captured_at=monday
    )
    assert manager.state == before


def test_market_loss_with_same_book_still_halts(tmp_path):
    manager = captured(tmp_path)
    result = manager.evaluate_session_risk(snapshot(equity=70_000), session_label=OPEN.date())
    assert result.action == "halt"


def test_legacy_managed_identity_is_not_silently_adopted(tmp_path):
    manager = captured(tmp_path)
    write_halt_state(tmp_path, manager.state.replace(baseline_book_identity=None))
    manager = RiskManager(RiskLimits(), halt_state_path=tmp_path / "halt.json")
    assert manager.evaluate_session_risk(snapshot(), session_label=OPEN.date()).action == "reject"


def test_explicit_snapshot_reset_accepts_transfer_and_preserves_halt(tmp_path):
    manager = captured(tmp_path)
    manager.halt("operator maintenance")
    changed = snapshot(scope="account", excluded_symbols=(), excluded_quantities={}, equity=100_000)
    _, after = reset_equity_baseline(
        tmp_path,
        equity=changed.equity,
        currency="CHF",
        observed_at=OPEN,
        snapshot=changed,
        scope=changed.scope,
        now=OPEN + timedelta(seconds=10),
        reason="book transfer reviewed",
        actor="test",
    )
    assert after.halted
    assert after.baseline_scope == "account"
    assert after.baseline_book_identity == changed.risk_book_identity


def test_concurrent_reset_survives_a_stale_manager_halt(tmp_path):
    manager = captured(tmp_path)
    changed = snapshot(equity=70_000)
    reset_equity_baseline(
        tmp_path,
        equity=changed.equity,
        currency="CHF",
        observed_at=OPEN,
        snapshot=changed,
        scope="managed",
        now=OPEN + timedelta(seconds=10),
        reason="reviewed",
        actor="test",
    )
    manager.halt("independent halt arrived after reset")
    state = read_halt_state(tmp_path)
    assert state.halted
    assert state.equity_high_watermark == 70_000
    assert state.daily_equity_open == 70_000
    assert state.baseline_book_identity == changed.risk_book_identity


def test_reset_rejects_mislabeled_amount_even_with_matching_scope(tmp_path):
    captured(tmp_path)
    with pytest.raises(BaselineResetError, match="do not match"):
        reset_equity_baseline(
            tmp_path,
            equity=100_000,
            currency="CHF",
            observed_at=OPEN,
            snapshot=snapshot(),
            scope="managed",
            now=OPEN,
            reason="test",
            actor="test",
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_equity_never_changes_a_baseline(tmp_path, bad):
    manager = captured(tmp_path)
    before = manager.state
    assert (
        manager.evaluate_session_risk(snapshot(equity=bad), session_label=OPEN.date()).action
        == "reject"
    )
    with pytest.raises(BaselineResetError):
        reset_equity_baseline(
            tmp_path,
            equity=bad,
            currency="CHF",
            observed_at=OPEN,
            now=OPEN,
            reason="test",
            actor="test",
        )
    assert manager.state == before


def test_unreadable_hold_list_is_not_an_empty_book(tmp_path):
    from trading.runner.holds import load_holds

    (tmp_path / "holds.json").write_text('{"symbols":')
    with pytest.raises(ValueError, match="Cannot read pinned holdings"):
        load_holds(tmp_path, strict=True)
