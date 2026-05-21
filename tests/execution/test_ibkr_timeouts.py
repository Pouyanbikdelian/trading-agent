r"""Tests for the IBKR broker timeout wrappers.

These don't need a real IB Gateway — we drive the broker with a stub
``ib`` object whose methods sleep past the timeout, and assert the
broker raises BrokerTimeoutError instead of hanging the test runner.
"""

from __future__ import annotations

import time

import pytest

from trading.execution.ibkr import BrokerTimeoutError, IbkrBroker


class _SlowStubIB:
    """Stand-in for ib_async.IB() that simulates a wedged broker session.

    Every API method blocks for ``sleep`` seconds before returning. With a
    short broker timeout (e.g. 0.5s) and a longer stub sleep (e.g. 5s),
    we get a deterministic BrokerTimeoutError.
    """

    def __init__(self, *, sleep: float = 5.0) -> None:
        self._sleep = sleep
        self._connected = True

    def isConnected(self) -> bool:
        return self._connected

    def accountSummary(self):
        time.sleep(self._sleep)
        return []

    def positions(self):
        time.sleep(self._sleep)
        return []

    def fills(self):
        time.sleep(self._sleep)
        return []

    def openTrades(self):
        time.sleep(self._sleep)
        return []


@pytest.fixture
def slow_broker() -> IbkrBroker:
    broker = IbkrBroker(ib=_SlowStubIB(sleep=5.0))
    broker._connected = True
    # Shorten timeout so the test finishes fast.
    broker.DEFAULT_API_TIMEOUT_S = 0.3  # type: ignore[misc]
    return broker


def test_get_account_raises_timeout_when_ib_blocks(slow_broker: IbkrBroker) -> None:
    with pytest.raises(BrokerTimeoutError) as exc_info:
        slow_broker.get_account()
    assert "accountSummary" in str(exc_info.value)


def test_get_positions_raises_timeout(slow_broker: IbkrBroker) -> None:
    with pytest.raises(BrokerTimeoutError) as exc_info:
        slow_broker.get_positions()
    assert "positions" in str(exc_info.value)


def test_get_fills_raises_timeout(slow_broker: IbkrBroker) -> None:
    with pytest.raises(BrokerTimeoutError) as exc_info:
        slow_broker.get_fills()
    assert "fills" in str(exc_info.value)


def test_timeout_message_mentions_recovery_hint(slow_broker: IbkrBroker) -> None:
    """The error message should guide the operator to a fix, not just
    state the timeout — paper trading is where we need clarity."""
    with pytest.raises(BrokerTimeoutError) as exc_info:
        slow_broker.get_account()
    msg = str(exc_info.value)
    assert "gateway" in msg.lower()
    assert "restart" in msg.lower()


# ---------------------------------------------------------------------------
# Auto-reconnect on BrokerTimeoutError
#
# New: prod runs into IBKR Gateway Error 1100 ("connectivity lost") which
# wedges API calls until the local 30s timeout. The broker now auto-
# reconnects and retries once before surrendering. We test the retry
# path with a stub that hangs the first call and returns fast on the second.
# ---------------------------------------------------------------------------


class _FlakyIb:
    """First call blocks; subsequent calls return immediately. Mimics the
    session-drop-then-recover pattern that prompted the auto-reconnect."""

    def __init__(self, sleep_on_first: float = 5.0) -> None:
        self._sleep = sleep_on_first
        self._calls = 0
        self._connected = True

    def isConnected(self) -> bool:
        return self._connected

    async def connectAsync(self, host: str, port: int, clientId: int) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def accountSummary(self):
        self._calls += 1
        if self._calls == 1:
            time.sleep(self._sleep)
        return [
            type("R", (), {"tag": "TotalCashValue", "value": "100000"})(),
            type("R", (), {"tag": "NetLiquidation", "value": "100000"})(),
        ]

    def positions(self):
        return []


def test_bounded_auto_reconnects_and_retries(monkeypatch) -> None:
    """First accountSummary blocks past the timeout. Broker auto-reconnects
    (using stub connectAsync) and retries — the retry succeeds quickly."""
    ib = _FlakyIb(sleep_on_first=2.0)
    broker = IbkrBroker(ib=ib)
    broker._connected = True
    broker.DEFAULT_API_TIMEOUT_S = 0.3  # type: ignore[misc]

    snap = broker.get_account()
    # Two calls happened in total: one that timed out, one that succeeded.
    assert ib._calls == 2
    assert snap.cash == 100_000.0
    assert snap.equity == 100_000.0


def test_bounded_surrenders_after_second_timeout() -> None:
    """If the gateway is genuinely wedged (every call hangs), the second
    attempt also times out and we re-raise. Auto-halt counter ticks in
    the runner; the cycle aborts. We test only the surrender behavior here."""

    class _ChronicallySlow(_FlakyIb):
        def accountSummary(self):
            self._calls += 1
            time.sleep(2.0)  # both calls hang
            return []

    ib = _ChronicallySlow()
    broker = IbkrBroker(ib=ib)
    broker._connected = True
    broker.DEFAULT_API_TIMEOUT_S = 0.3  # type: ignore[misc]

    with pytest.raises(BrokerTimeoutError):
        broker.get_account()
    # Confirmed we tried twice — auto-reconnect + retry path was exercised.
    assert ib._calls == 2
