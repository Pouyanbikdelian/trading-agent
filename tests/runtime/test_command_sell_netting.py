"""``/close`` and ``/flatten`` must not sell a position twice.

End-to-end version of the invariant unit-tested in
``tests/risk/test_presubmit.py``, driven through the real command
handlers. The sequence being pinned is the one the live system makes
easy: the guards run every 15 minutes through 20:59 UTC, so a
trailing-stop exit placed after the US close sits unfilled overnight.
The operator sees "🛑 Trailing stop VST — closing", reaches for
``/close VST``, and without netting that is a second full-size sell.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import AssetClass, Instrument, Order, OrderType, Position, Side, TimeInForce
from trading.runtime.command_processor import _h_close, _h_flatten, _h_sell
from trading.runtime.commands import Command, CommandType


def _instrument(symbol: str) -> Instrument:
    return Instrument(symbol=symbol, asset_class=AssetClass.EQUITY)


def _position(symbol: str, qty: float) -> Position:
    return Position(instrument=_instrument(symbol), quantity=qty, avg_price=100.0)


def _working_sell(symbol: str, qty: float) -> Order:
    return Order(
        client_order_id=f"guard-{symbol}",
        instrument=_instrument(symbol),
        side=Side.SELL,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=datetime(2026, 8, 7, 20, 50, tzinfo=timezone.utc),
    )


class _Broker:
    def __init__(self, positions=(), open_orders=()):
        self.positions = list(positions)
        self.open_orders = list(open_orders)
        self.submitted: list[Order] = []

    def get_positions(self):
        return list(self.positions)

    def get_open_orders(self):
        return list(self.open_orders)

    def submit_order(self, order: Order) -> None:
        self.submitted.append(order)


def _cmd(ctype: CommandType, **args) -> Command:
    return Command.new(ctype, args=args, requested_by="test")


def test_close_after_an_unfilled_guard_exit_is_refused(monkeypatch):
    broker = _Broker([_position("VST", 63)], [_working_sell("VST", 63)])

    with pytest.raises(ValueError) as exc:
        _h_close(_cmd(CommandType.CLOSE, symbol="VST"), broker)

    assert "already working" in str(exc.value)
    assert broker.submitted == []


def test_close_with_no_working_order_sells_the_whole_position(monkeypatch):
    broker = _Broker([_position("VST", 63)])

    _h_close(_cmd(CommandType.CLOSE, symbol="VST"), broker)

    assert len(broker.submitted) == 1
    assert broker.submitted[0].quantity == 63


def test_a_partial_working_sell_leaves_the_remainder_closable(monkeypatch):
    broker = _Broker([_position("VST", 63)], [_working_sell("VST", 20)])

    result = _h_close(_cmd(CommandType.CLOSE, symbol="VST"), broker)

    assert broker.submitted[0].quantity == 43
    assert "clamped" in result


def test_an_explicit_oversized_sell_is_clamped(monkeypatch):
    broker = _Broker([_position("VST", 63)], [_working_sell("VST", 20)])

    _h_sell(_cmd(CommandType.SELL, symbol="VST", qty=63), broker)

    assert broker.submitted[0].quantity == 43


def test_flatten_skips_a_name_already_fully_covered(monkeypatch):
    """The panic button stays unconditional about /hold, but it must not
    open a short in the names it is trying to exit."""
    broker = _Broker(
        [_position("VST", 63), _position("WMT", 10)],
        [_working_sell("VST", 63)],
    )

    result = _h_flatten(_cmd(CommandType.FLATTEN), broker)

    assert [o.instrument.symbol for o in broker.submitted] == ["WMT"]
    assert result["closed"] == ["WMT"]
    assert any("VST" in s for s in result["skipped"])


def test_flatten_still_closes_everything_when_nothing_is_working(monkeypatch):
    broker = _Broker([_position("VST", 63), _position("WMT", 10)])

    result = _h_flatten(_cmd(CommandType.FLATTEN), broker)

    assert sorted(o.instrument.symbol for o in broker.submitted) == ["VST", "WMT"]
    assert result["n_closed"] == 2
