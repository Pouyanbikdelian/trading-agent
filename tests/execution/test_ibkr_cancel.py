"""Cancelling an order must either work or say it didn't.

``get_open_orders`` — the source for ``/orders`` — falls back to
``broker-{orderId}`` when an order carries no ``orderRef``, which is
every order placed in TWS by hand or by an older build.
``cancel_order`` matched on ``orderRef`` alone, so the operator was
shown an id it could not match, and the miss was a silent no-op while
``_h_cancel_order`` reported ``cancelled: True``.

It closes a loop with the sell clamp: that refusal tells the operator to
cancel the working order first.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading.execution.base import BrokerError
from trading.execution.ibkr import IbkrBroker


def _trade(*, order_ref: str = "", order_id: int = 0, perm_id: int = 0):
    return SimpleNamespace(
        order=SimpleNamespace(orderRef=order_ref, orderId=order_id, permId=perm_id)
    )


class _Broker(IbkrBroker):
    """Real cancel_order, stubbed transport."""

    def __init__(self, trades):
        self._trades = trades
        self.cancelled: list[object] = []

    def _ensure_connected(self) -> None:  # type: ignore[override]
        return None

    def _bounded(self, what, fn, *, timeout=None):  # type: ignore[override]
        if what == "openTrades":
            return self._trades
        return fn()

    @property
    def _ib(self):  # type: ignore[override]
        return SimpleNamespace(
            openTrades=lambda: self._trades,
            cancelOrder=lambda o: self.cancelled.append(o),
        )


def test_cancel_by_our_own_order_ref():
    b = _Broker([_trade(order_ref="trd-abc", order_id=41)])

    b.cancel_order("trd-abc")

    assert len(b.cancelled) == 1


def test_cancel_by_the_synthesized_id_that_orders_displays():
    """The regression: an order with no orderRef is listed as broker-41,
    and that is the only id the operator can see."""
    b = _Broker([_trade(order_id=41)])

    b.cancel_order("broker-41")

    assert len(b.cancelled) == 1


def test_cancel_by_bare_order_id():
    b = _Broker([_trade(order_id=41)])

    b.cancel_order("41")

    assert len(b.cancelled) == 1


def test_a_miss_raises_instead_of_reporting_success():
    b = _Broker([_trade(order_ref="trd-abc", order_id=41)])

    with pytest.raises(BrokerError) as exc:
        b.cancel_order("trd-nope")

    assert "nothing cancelled" in str(exc.value)
    assert b.cancelled == []


def test_the_error_names_what_is_actually_working():
    """So the operator can re-issue against a real id rather than guess."""
    b = _Broker([_trade(order_ref="trd-abc", order_id=41)])

    with pytest.raises(BrokerError) as exc:
        b.cancel_order("trd-nope")

    assert "trd-abc" in str(exc.value)


def test_no_working_orders_at_all_still_raises():
    b = _Broker([])

    with pytest.raises(BrokerError, match="none"):
        b.cancel_order("trd-abc")
