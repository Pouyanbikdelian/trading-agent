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
