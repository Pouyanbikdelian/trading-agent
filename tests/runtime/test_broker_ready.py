"""Pre-cycle broker readiness, and telling a dead session from a refused order.

IBKR mandates 2FA for every user and re-prompts after its daily restart
and weekly shutdown. IBC retries a missed prompt, so a missed tap is
recoverable — but only if somebody taps. Every other consequence of a
dead gateway is non-fatal; the cycle is the one moment it is not,
because a cycle that cannot reach the broker submits nothing and looks
exactly like a cycle that decided to do nothing.
"""

from __future__ import annotations

import pytest

from trading.runner.cycle import _looks_like_disconnect
from trading.runner.runner import _precycle_trigger
from trading.runtime.broker_ready import (
    check_broker_ready,
    format_not_ready_alert,
)


class _Broker:
    def __init__(self, snap=None, exc: Exception | None = None) -> None:
        self._snap, self._exc = snap, exc

    def get_account(self):
        if self._exc:
            raise self._exc
        return self._snap


class _Snap:
    def __init__(self, equity) -> None:
        self.equity = equity


class TestCheckBrokerReady:
    def test_ready_when_the_account_answers(self) -> None:
        r = check_broker_ready(_Broker(_Snap(88_265.0)))
        assert r["ready"] is True and r["equity"] == 88_265.0

    def test_not_ready_when_the_call_raises(self) -> None:
        r = check_broker_ready(_Broker(exc=ConnectionError("Not connected")))
        assert r["ready"] is False and "Not connected" in r["detail"]

    def test_a_snapshot_with_no_equity_is_not_ready(self) -> None:
        """The shape a half-authenticated gateway returns: the session
        answers, but there is no account behind it."""
        r = check_broker_ready(_Broker(_Snap(None)))
        assert r["ready"] is False and "half-open" in r["detail"]

    def test_never_raises(self) -> None:
        """This runs on a scheduler; if it can throw, it can silently
        stop running — the exact failure mode it exists to prevent."""

        class Hostile:
            def get_account(self):
                raise BaseExceptionGroup("weird", [ValueError("x")])  # noqa: F821

        try:
            out = check_broker_ready(Hostile())
        except Exception:  # pragma: no cover
            pytest.fail("check_broker_ready raised")
        assert out["ready"] is False

    def test_the_alert_leads_with_the_deadline(self) -> None:
        """The only thing that matters is how long is left to fix it."""
        msg = format_not_ready_alert({"detail": "Not connected"}, minutes_to_cycle=60)
        assert "cycle in 60 min" in msg
        assert "IBKR Mobile" in msg and "restart ib-gateway" in msg


class TestPrecycleTrigger:
    """Derived from the cycle's own cron so the two cannot drift. A check
    that fires an hour before a cycle that has since moved would report
    all-clear and let the cycle run unchecked."""

    def test_one_hour_before_the_friday_cycle(self) -> None:
        t = _precycle_trigger("5 21 * * FRI", "UTC")
        assert t is not None
        fields = {f.name: str(f) for f in t.fields}
        assert fields["hour"] == "20" and fields["minute"] == "5"
        assert fields["day_of_week"] == "fri"

    def test_it_follows_a_changed_cron(self) -> None:
        """Winter DST moves the cycle to 22:05; the warning must move too."""
        t = _precycle_trigger("5 22 * * FRI", "UTC")
        fields = {f.name: str(f) for f in t.fields}
        assert fields["hour"] == "21" and fields["minute"] == "5"

    def test_borrows_from_the_hour_when_minutes_go_negative(self) -> None:
        t = _precycle_trigger("30 16 * * MON-FRI", "UTC", lead_minutes=45)
        fields = {f.name: str(f) for f in t.fields}
        assert fields["hour"] == "15" and fields["minute"] == "45"

    def test_returns_none_rather_than_guessing_across_midnight(self) -> None:
        """Going back past midnight also shifts the day-of-week, and
        getting that subtly wrong is worse than not running."""
        assert _precycle_trigger("5 0 * * FRI", "UTC") is None

    def test_returns_none_on_shapes_it_does_not_understand(self) -> None:
        assert _precycle_trigger("*/15 * * * *", "UTC") is None
        assert _precycle_trigger("nonsense", "UTC") is None


class TestDisconnectDetection:
    """'Order rejected: insufficient margin' is information. 'Cannot reach
    the broker' means the whole batch is dead and a human must act."""

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("Not connected"),
            TimeoutError("timed out waiting for the gateway"),
            OSError("[Errno 111] Connection refused"),
            RuntimeError("Connect call failed ('127.0.0.1', 4001)"),
            OSError("no route to host"),
        ],
    )
    def test_connectivity_failures_are_recognised(self, exc: Exception) -> None:
        assert _looks_like_disconnect(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("insufficient margin"),
            RuntimeError("order size exceeds position limit"),
            ValueError("contract not found for symbol ZZZZ"),
        ],
    )
    def test_business_rejections_are_not_escalated(self, exc: Exception) -> None:
        assert _looks_like_disconnect(exc) is False
