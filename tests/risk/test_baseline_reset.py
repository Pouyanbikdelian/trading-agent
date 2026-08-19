"""The escape hatch from a drawdown lockout, and its guardrails.

`equity_high_watermark` only ever ratchets up. With a tight
MAX_DRAWDOWN_PCT that makes the drawdown switch a one-way trapdoor: below
the threshold every check halts, `/resume` re-halts on the next 60s tick,
and the documented escape — a new equity high — is unreachable because a
halted book may only be reduced.

`reset_equity_baseline` is the operator's acknowledgement. It must:
  * actually end the lockout (the whole point);
  * leave the halt itself alone (accepting a loss and restarting trading
    are two decisions);
  * refuse anything it cannot verify, writing nothing;
  * leave a record, because a moved risk reference with no audit trail is
    indistinguishable from a bug.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.core.types import AccountSnapshot
from trading.risk.halt_file import (
    BaselineResetError,
    read_halt_state,
    reset_equity_baseline,
    set_halted,
    write_halt_state,
)
from trading.risk.limits import HaltState, RiskLimits
from trading.risk.manager import RiskManager

# 2026-03-09 is a real NYSE session; 14:35Z is inside its cash hours.
IN_SESSION = datetime(2026, 3, 9, 14, 35, tzinfo=timezone.utc)
SESSION = date(2026, 3, 9)
# 2026-03-07 is a Saturday.
CLOSED = datetime(2026, 3, 7, 14, 35, tzinfo=timezone.utc)


def _drawn_down(state_dir: Path) -> None:
    """The permanent-lockout shape.

    This is the morning *after* the 2026-08-18 loss: a verified NYSE-open
    baseline has been captured, so the day is flat and the daily-loss
    switch is satisfied — but equity is still ~2.1% under an all-time peak
    that nothing can lower, so the drawdown switch halts on every tick and
    the account can never buy its way back to a new high.
    """
    write_halt_state(
        state_dir,
        HaltState(
            halted=True,
            reason="drawdown -2.06% breaches limit -2.00%",
            halted_at=IN_SESSION,
            equity_high_watermark=87_900.0,
            daily_equity_open=86_087.26,
            last_day=SESSION,
            daily_baseline_session=SESSION,
            daily_baseline_captured_at=IN_SESSION,
            daily_baseline_source="snapshot_refresh",
            daily_baseline_currency="CHF",
        ),
    )


def _reset(state_dir: Path, *, equity: float = 86_087.26, **kw):
    return reset_equity_baseline(
        state_dir,
        equity=equity,
        currency=kw.pop("currency", "CHF"),
        observed_at=kw.pop("observed_at", IN_SESSION),
        reason=kw.pop("reason", "operator accepted the drawdown"),
        actor=kw.pop("actor", "telegram"),
        now=kw.pop("now", IN_SESSION + timedelta(seconds=30)),
        **kw,
    )


class TestItEndsTheLockout:
    def test_the_high_water_mark_moves_down_to_live_equity(self, tmp_path: Path) -> None:
        _drawn_down(tmp_path)

        before, after = _reset(tmp_path)

        assert before.equity_high_watermark == 87_900.0
        assert after.equity_high_watermark == 86_087.26
        assert after.daily_equity_open == 86_087.26

    def test_without_a_reset_the_drawdown_halt_is_permanent(self, tmp_path: Path) -> None:
        """Resume, and it re-halts on the very next evaluation. Forever."""
        _drawn_down(tmp_path)
        account = AccountSnapshot(
            ts=IN_SESSION,
            cash=1_000.0,
            equity=86_087.26,
            base_currency="CHF",
            positions={},
        )
        mgr = RiskManager(
            RiskLimits(max_daily_loss_pct=0.006, max_drawdown_pct=0.02),
            halt_state_path=tmp_path / "halt.json",
        )

        for _ in range(3):
            set_halted(tmp_path, halted=False)
            assert mgr.evaluate_session_risk(account, session_label=SESSION).action == "halt"

    def test_a_resumed_account_then_passes_risk_instead_of_re_halting(self, tmp_path: Path) -> None:
        """The end-to-end claim: reset, resume, and a normal cycle may run.

        Without the reset this same sequence returns ``halt`` forever —
        that is the bug this command exists for.
        """
        _drawn_down(tmp_path)
        limits = RiskLimits(max_daily_loss_pct=0.006, max_drawdown_pct=0.02)
        account = AccountSnapshot(
            ts=IN_SESSION,
            cash=1_000.0,
            equity=86_087.26,
            base_currency="CHF",
            positions={},
        )

        locked_out = RiskManager(limits, halt_state_path=tmp_path / "halt.json")
        set_halted(tmp_path, halted=False)
        assert locked_out.evaluate_session_risk(account, session_label=SESSION).action == "halt"

        _reset(tmp_path)
        set_halted(tmp_path, halted=False)
        recovered = RiskManager(limits, halt_state_path=tmp_path / "halt.json")

        assert recovered.evaluate_session_risk(account, session_label=SESSION).action == "allow"

    def test_the_daily_baseline_is_stamped_as_operator_sourced(self, tmp_path: Path) -> None:
        """Trusted, but never disguised as the opening bell."""
        _drawn_down(tmp_path)

        _, after = _reset(tmp_path, actor="telegram")

        assert after.daily_baseline_session == SESSION
        assert after.daily_baseline_source == "operator_reset:telegram"
        assert after.daily_baseline_currency == "CHF"
        assert after.daily_baseline_captured_at == IN_SESSION

    def test_off_session_it_still_lowers_the_peak_but_claims_no_daily_baseline(
        self, tmp_path: Path
    ) -> None:
        _drawn_down(tmp_path)

        _, after = _reset(tmp_path, observed_at=CLOSED, now=CLOSED + timedelta(seconds=30))

        assert after.equity_high_watermark == 86_087.26
        assert after.daily_baseline_session is None


class TestItNeverTouchesTheHalt:
    def test_a_halted_account_stays_halted_with_its_reason(self, tmp_path: Path) -> None:
        _drawn_down(tmp_path)

        _, after = _reset(tmp_path)

        assert after.halted is True
        assert after.reason == "drawdown -2.06% breaches limit -2.00%"
        assert read_halt_state(tmp_path).halted is True

    def test_an_unhalted_account_is_not_halted_by_a_reset(self, tmp_path: Path) -> None:
        write_halt_state(tmp_path, HaltState(equity_high_watermark=90_000.0))

        _, after = _reset(tmp_path)

        assert after.halted is False


class TestItRefusesWhatItCannotVerify:
    @pytest.mark.parametrize(
        ("kwargs", "fragment"),
        [
            ({"equity": 0.0}, "not positive"),
            ({"equity": -5.0}, "not positive"),
            ({"currency": "  "}, "currency is unknown"),
        ],
    )
    def test_bad_inputs_write_nothing(self, tmp_path: Path, kwargs, fragment: str) -> None:
        _drawn_down(tmp_path)

        with pytest.raises(BaselineResetError, match=fragment):
            _reset(tmp_path, **kwargs)

        assert read_halt_state(tmp_path).equity_high_watermark == 87_900.0

    def test_a_stale_snapshot_is_refused(self, tmp_path: Path) -> None:
        """Re-anchoring risk to an hour-old equity is how you re-poison it."""
        _drawn_down(tmp_path)

        with pytest.raises(BaselineResetError, match="too stale"):
            _reset(tmp_path, now=IN_SESSION + timedelta(hours=1))

        assert read_halt_state(tmp_path).equity_high_watermark == 87_900.0

    def test_a_naive_timestamp_is_refused(self, tmp_path: Path) -> None:
        _drawn_down(tmp_path)

        with pytest.raises(BaselineResetError, match="no timezone"):
            _reset(tmp_path, observed_at=IN_SESSION.replace(tzinfo=None))

    def test_a_currency_that_disagrees_with_the_stored_baseline_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The 2026-08-07 failure mode: a CHF account vs a USD baseline."""
        write_halt_state(
            tmp_path,
            HaltState(
                equity_high_watermark=87_900.0,
                daily_equity_open=86_989.82,
                daily_baseline_currency="USD",
                daily_baseline_session=SESSION,
                daily_baseline_captured_at=IN_SESSION,
                daily_baseline_source="session_open",
            ),
        )

        with pytest.raises(BaselineResetError, match="USD"):
            _reset(tmp_path, currency="CHF")

        assert read_halt_state(tmp_path).equity_high_watermark == 87_900.0


class TestItLeavesARecord:
    def test_every_reset_appends_an_audit_line(self, tmp_path: Path) -> None:
        _drawn_down(tmp_path)

        _reset(tmp_path, reason="August drawdown, accepted", actor="telegram")

        rows = [
            json.loads(line)
            for line in (tmp_path / "baseline_resets.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["reason"] == "August drawdown, accepted"
        assert rows[0]["actor"] == "telegram"
        assert rows[0]["previous_high_watermark"] == 87_900.0
        assert rows[0]["equity"] == 86_087.26
        assert rows[0]["currency"] == "CHF"
        assert rows[0]["still_halted"] is True

    def test_a_refused_reset_writes_no_audit_line(self, tmp_path: Path) -> None:
        _drawn_down(tmp_path)

        with pytest.raises(BaselineResetError):
            _reset(tmp_path, equity=0.0)

        assert not (tmp_path / "baseline_resets.jsonl").exists()
