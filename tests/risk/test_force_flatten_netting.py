"""Force-flatten must not open a short in the names it is exiting.

Third path to carry the 2026-07-15 stacking fix. The cycle got it then,
the manual command handlers on 2026-08-07, and ``force_flatten_orders``
— reached by the playbook's regime rule and by ``/approve flat`` — an
hour after that.

A full-size close submitted on top of a full-size close already working
at the broker crosses zero. In a long-only system the panic button would
leave you short the very names you were trying to get out of.
"""

from __future__ import annotations

from datetime import datetime, timezone

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
    TimeInForce,
)

TS = datetime(2026, 8, 7, 21, 5, tzinfo=timezone.utc)


def _instrument(symbol: str, cls: AssetClass = AssetClass.EQUITY) -> Instrument:
    return Instrument(symbol=symbol, asset_class=cls)


def _position(symbol: str, qty: float, cls: AssetClass = AssetClass.EQUITY) -> Position:
    return Position(instrument=_instrument(symbol, cls), quantity=qty, avg_price=100.0)


def _working(symbol: str, qty: float, side: Side = Side.SELL) -> Order:
    return Order(
        client_order_id=f"w-{symbol}",
        instrument=_instrument(symbol),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=TS,
    )


def test_without_working_orders_it_closes_everything(mgr) -> None:
    out = mgr.force_flatten_orders([_position("VST", 63)], ts=TS)
    assert len(out) == 1 and out[0].quantity == 63


def test_a_fully_covered_position_is_not_closed_twice(mgr) -> None:
    out = mgr.force_flatten_orders(
        [_position("VST", 63)], ts=TS, working_orders=[_working("VST", 63)]
    )
    assert out == []


def test_a_partially_covered_position_closes_only_the_remainder(mgr) -> None:
    out = mgr.force_flatten_orders(
        [_position("VST", 63)], ts=TS, working_orders=[_working("VST", 20)]
    )
    assert len(out) == 1 and out[0].quantity == 43
    assert out[0].side == Side.SELL


def test_an_overshooting_working_order_does_not_flip_us_short(mgr) -> None:
    """A working sell larger than the position (shouldn't happen, but the
    panic button is the wrong place to assume that) must not produce a BUY."""
    out = mgr.force_flatten_orders(
        [_position("VST", 63)], ts=TS, working_orders=[_working("VST", 100)]
    )
    assert out == []


def test_a_working_buy_increases_what_must_be_closed(mgr) -> None:
    out = mgr.force_flatten_orders(
        [_position("VST", 63)], ts=TS, working_orders=[_working("VST", 10, side=Side.BUY)]
    )
    assert len(out) == 1 and out[0].quantity == 73


def test_etf_and_equity_keys_are_matched(mgr) -> None:
    """IBKR reports ETFs as plain stocks, so a broker-side equity:GLD must
    net against a universe-side etf:GLD."""
    out = mgr.force_flatten_orders(
        [_position("GLD", 40, AssetClass.ETF)], ts=TS, working_orders=[_working("GLD", 40)]
    )
    assert out == []


def test_other_symbols_are_untouched(mgr) -> None:
    out = mgr.force_flatten_orders(
        [_position("VST", 63), _position("WMT", 10)],
        ts=TS,
        working_orders=[_working("VST", 63)],
    )
    assert [o.instrument.symbol for o in out] == ["WMT"]


def test_shorts_are_still_covered(mgr) -> None:
    """Flatten closes in both directions; only the long-only INVARIANT is
    one-sided."""
    out = mgr.force_flatten_orders([_position("VST", -20)], ts=TS)
    assert len(out) == 1
    assert out[0].side == Side.BUY and out[0].quantity == 20
