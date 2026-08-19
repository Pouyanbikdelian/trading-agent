"""Safety contract for the continuous, exchange-session risk monitor.

The runner refreshes the broker account every minute.  That must *not* make
the first snapshot after a restart look like the opening balance: a noon
restart would turn an already-lost day into a clean slate.  These tests pin
the explicit, trusted-open baseline contract used by that monitor.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from trading.core.types import AccountSnapshot
from trading.risk import RiskManager
from trading.runtime.nyse_session import NYSESession, nyse_session_on


def _session(label: date) -> NYSESession:
    session = nyse_session_on(label)
    assert session is not None, f"expected NYSE session on {label.isoformat()}"
    return session


def _snapshot(
    ts: datetime,
    equity: float = 100_000.0,
    *,
    currency: str = "CHF",
) -> AccountSnapshot:
    return AccountSnapshot(
        ts=ts,
        cash=equity,
        equity=equity,
        base_currency=currency,
    )


def test_opening_capture_records_a_trusted_session_baseline(mgr) -> None:
    session = _session(date(2026, 3, 9))  # First regular session after DST starts.
    captured_at = session.market_open + timedelta(minutes=2)
    account = _snapshot(captured_at)

    mgr.capture_session_open(
        account,
        session_date=session.label,
        captured_at=captured_at,
        source="snapshot_refresh",
    )

    state = mgr.state
    assert state.daily_equity_open == 100_000.0
    assert state.daily_baseline_session == session.label
    assert state.daily_baseline_captured_at == captured_at
    assert state.daily_baseline_source == "snapshot_refresh"
    assert state.daily_baseline_currency == "CHF"
    assert (
        mgr.evaluate_session_risk(
            account,
            session_label=session.label,
        ).action
        == "allow"
    )


def test_noon_snapshot_never_becomes_the_daily_baseline(mgr) -> None:
    session = _session(date(2026, 3, 9))
    noon = session.market_open + timedelta(hours=3)
    account = _snapshot(noon, 97_000.0)

    mgr.capture_session_open(
        account,
        session_date=session.label,
        captured_at=noon,
        source="snapshot_refresh",
    )

    state = mgr.state
    assert state.daily_equity_open == 0.0
    assert state.daily_baseline_session is None
    assert state.daily_baseline_captured_at is None

    decision = mgr.evaluate_session_risk(account, session_label=session.label)
    assert decision.action == "reject"
    assert "baseline" in decision.reason.lower()
    # Missing data is an execution block, not a permanent operator halt.
    assert not mgr.is_halted()


def test_capture_refuses_a_mismatched_session_label(mgr) -> None:
    """A caller cannot label Monday's opening snapshot as Tuesday's open."""
    session = _session(date(2026, 3, 9))
    account = _snapshot(session.market_open + timedelta(minutes=1))

    mgr.capture_session_open(
        account,
        session_date=date(2026, 3, 10),
        captured_at=account.ts,
    )

    assert mgr.state.daily_equity_open == 0.0
    assert mgr.state.daily_baseline_session is None
    assert mgr.evaluate_session_risk(account, session_label=session.label).action == "reject"


def test_daily_loss_from_trusted_open_baseline_halts(mgr) -> None:
    session = _session(date(2026, 3, 9))
    opened = _snapshot(session.market_open + timedelta(minutes=1))
    mgr.capture_session_open(
        opened,
        session_date=session.label,
        captured_at=opened.ts,
    )

    losing_snapshot = _snapshot(session.market_open + timedelta(hours=2), 97_000.0)
    decision = mgr.evaluate_session_risk(
        losing_snapshot,
        session_label=session.label,
    )

    assert decision.action == "halt"
    assert "daily loss" in decision.reason
    assert mgr.is_halted()


def test_currency_change_cannot_reuse_a_trusted_daily_open(mgr) -> None:
    """A CHF baseline must never be used as if it were a USD baseline."""
    session = _session(date(2026, 3, 9))
    opened = _snapshot(session.market_open + timedelta(minutes=1), currency="CHF")
    mgr.capture_session_open(
        opened,
        session_date=session.label,
        captured_at=opened.ts,
    )
    usd_snapshot = _snapshot(
        session.market_open + timedelta(hours=2),
        97_000.0,
        currency="USD",
    )

    decision = mgr.evaluate_session_risk(usd_snapshot, session_label=session.label)

    assert decision.action == "reject"
    assert "currency" in decision.reason.lower()
    assert not mgr.is_halted()


def test_resume_during_the_same_breach_rehalts_against_the_original_open(mgr) -> None:
    """/resume is not permission to silently re-baseline an already-lost day."""
    session = _session(date(2026, 3, 9))
    opened = _snapshot(session.market_open + timedelta(minutes=1))
    mgr.capture_session_open(
        opened,
        session_date=session.label,
        captured_at=opened.ts,
    )
    losing_snapshot = _snapshot(session.market_open + timedelta(hours=2), 97_000.0)
    assert mgr.evaluate_session_risk(losing_snapshot, session_label=session.label).action == "halt"

    mgr.unhalt()
    decision = mgr.evaluate_session_risk(losing_snapshot, session_label=session.label)

    assert decision.action == "halt"
    assert mgr.state.daily_equity_open == 100_000.0
    assert mgr.state.daily_baseline_session == session.label


def test_a_valid_open_capture_never_clears_an_existing_halt(mgr) -> None:
    session = _session(date(2026, 3, 9))
    mgr.halt("operator investigation")
    account = _snapshot(session.market_open + timedelta(minutes=1))

    mgr.capture_session_open(
        account,
        session_date=session.label,
        captured_at=account.ts,
    )

    assert mgr.is_halted()
    assert mgr.state.reason == "operator investigation"
    # Recording a genuine next-open baseline while halted is useful for a
    # later deliberate /resume, but it must never perform that resume itself.
    assert mgr.state.daily_baseline_session == session.label


def test_legacy_daily_open_without_trusted_provenance_is_not_accepted(tmp_path, limits) -> None:
    """Old halt.json files must not accidentally authorise noon execution."""
    session = _session(date(2026, 3, 9))
    path = tmp_path / "halt.json"
    path.write_text(
        json.dumps(
            {
                "halted": False,
                "reason": "",
                "halted_at": None,
                "equity_high_watermark": 100_000.0,
                "daily_equity_open": 100_000.0,
                "last_day": session.label.isoformat(),
            }
        )
    )
    legacy = RiskManager(limits, halt_state_path=path)
    noon_snapshot = _snapshot(session.market_open + timedelta(hours=3), 99_000.0)

    decision = legacy.evaluate_session_risk(noon_snapshot, session_label=session.label)

    assert decision.action == "reject"
    assert "baseline" in decision.reason.lower()
    assert not legacy.is_halted()


@pytest.mark.parametrize(
    "label",
    [
        date(2026, 3, 9),  # 13:30 UTC after spring DST shift.
        date(2026, 11, 2),  # 14:30 UTC after autumn DST shift.
    ],
)
def test_capture_uses_real_nyse_open_across_dst(mgr, label: date) -> None:
    session = _session(label)
    captured_at = session.market_open + timedelta(minutes=1)
    account = _snapshot(captured_at)

    mgr.capture_session_open(
        account,
        session_date=session.label,
        captured_at=captured_at,
    )

    assert mgr.state.daily_baseline_session == label
    assert mgr.evaluate_session_risk(account, session_label=label).action == "allow"


def test_closed_nyse_day_cannot_capture_or_authorise_daily_risk(mgr) -> None:
    # Good Friday is a NYSE holiday.  A process restart must not create a
    # baseline simply because it happens to receive an account snapshot.
    holiday = date(2026, 4, 3)
    snapshot_at = datetime(2026, 4, 3, 13, 30, tzinfo=timezone.utc)
    account = _snapshot(snapshot_at)

    mgr.capture_session_open(
        account,
        session_date=holiday,
        captured_at=snapshot_at,
    )

    assert mgr.state.daily_equity_open == 0.0
    assert mgr.state.daily_baseline_session is None
    decision = mgr.evaluate_session_risk(account, session_label=None)
    assert decision.action == "reject"
    assert "session" in decision.reason.lower()
    assert not mgr.is_halted()
