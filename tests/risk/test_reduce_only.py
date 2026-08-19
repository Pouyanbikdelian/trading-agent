"""The proof that a halted basket can only shrink the book.

Every test here asks one question: can this order take the account past
flat, or open something new? If the answer is "maybe", the order must be
gone. A false negative costs the desk one name in a defensive rebalance; a
false positive opens a short inside a kill switch.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
    TimeInForce,
)
from trading.risk.reduce_only import filter_reduce_only

NOW = datetime(2026, 8, 19, 14, 0, tzinfo=timezone.utc)


def inst(symbol: str, *, asset_class: AssetClass = AssetClass.EQUITY, currency: str = "USD"):
    return Instrument(
        symbol=symbol,
        asset_class=asset_class,
        currency=currency,
        exchange="SMART",
    )


def order(symbol: str, side: Side, qty: float, **kw) -> Order:
    return Order(
        client_order_id=f"{symbol}-{side.value}-{qty:g}",
        instrument=kw.pop("instrument", None) or inst(symbol, **kw),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=NOW,
    )


def pos(symbol: str, qty: float, **kw) -> Position:
    return Position(
        instrument=kw.pop("instrument", None) or inst(symbol, **kw),
        quantity=qty,
        avg_price=100.0,
    )


class TestOnlyExitsSurvive:
    def test_a_sell_against_a_long_is_kept(self) -> None:
        result = filter_reduce_only([order("AMD", Side.SELL, 50)], [pos("AMD", 100)], [])
        assert [o.instrument.symbol for o in result.kept] == ["AMD"]
        assert result.kept[0].quantity == 50
        assert not result.dropped

    def test_a_buy_against_a_long_is_dropped(self) -> None:
        result = filter_reduce_only([order("AMD", Side.BUY, 50)], [pos("AMD", 100)], [])
        assert not result.kept
        assert result.dropped == [("AMD", "would increase an existing position")]

    def test_a_buy_with_no_position_is_dropped(self) -> None:
        """The core case: a halted PM wanting GLD must not be able to buy it."""
        result = filter_reduce_only([order("GLD", Side.BUY, 10)], [pos("AMD", 100)], [])
        assert not result.kept
        assert "would open new exposure" in result.dropped[0][1]

    def test_a_sell_with_no_position_is_dropped_because_it_would_short(self) -> None:
        result = filter_reduce_only([order("GLD", Side.SELL, 10)], [pos("AMD", 100)], [])
        assert not result.kept
        assert "would open new exposure" in result.dropped[0][1]

    def test_a_buy_back_against_a_short_is_kept(self) -> None:
        result = filter_reduce_only([order("MU", Side.BUY, 30)], [pos("MU", -40)], [])
        assert [o.instrument.symbol for o in result.kept] == ["MU"]

    def test_a_sell_against_a_short_is_dropped(self) -> None:
        result = filter_reduce_only([order("MU", Side.SELL, 30)], [pos("MU", -40)], [])
        assert not result.kept


class TestItNeverCrossesFlat:
    def test_an_oversized_sell_is_trimmed_to_the_holding(self) -> None:
        result = filter_reduce_only([order("AMD", Side.SELL, 250)], [pos("AMD", 100)], [])
        assert result.kept[0].quantity == 100
        assert result.clamped == [("AMD", 250.0, 100.0)]

    def test_an_exact_full_exit_is_untouched(self) -> None:
        result = filter_reduce_only([order("AMD", Side.SELL, 100)], [pos("AMD", 100)], [])
        assert result.kept[0].quantity == 100
        assert not result.clamped

    def test_two_orders_on_one_name_share_the_headroom(self) -> None:
        """Should never happen upstream — but headroom is granted once, not twice."""
        result = filter_reduce_only(
            [order("AMD", Side.SELL, 80), order("AMD", Side.SELL, 80)],
            [pos("AMD", 100)],
            [],
        )
        assert [o.quantity for o in result.kept] == [80, 20]


class TestWorkingOrdersAreCountedAgainstTheHeadroom:
    def test_a_working_exit_reduces_what_may_still_be_sold(self) -> None:
        result = filter_reduce_only(
            [order("GEV", Side.SELL, 100)],
            [pos("GEV", 100)],
            [order("GEV", Side.SELL, 60)],
        )
        assert result.kept[0].quantity == 40
        assert result.clamped == [("GEV", 100.0, 40.0)]

    def test_a_fully_covered_position_yields_nothing(self) -> None:
        result = filter_reduce_only(
            [order("GEV", Side.SELL, 100)],
            [pos("GEV", 100)],
            [order("GEV", Side.SELL, 100)],
        )
        assert not result.kept
        assert "already fully covered" in result.dropped[0][1]

    def test_a_working_buy_on_a_long_blocks_the_name(self) -> None:
        """A queued buy can refill what we sell; we cannot prove the net."""
        result = filter_reduce_only(
            [order("OUST", Side.SELL, 100)],
            [pos("OUST", 100)],
            [order("OUST", Side.BUY, 5)],
        )
        assert not result.kept
        assert "cancel it first" in result.dropped[0][1]

    def test_working_exits_beyond_the_position_reject_the_name(self) -> None:
        result = filter_reduce_only(
            [order("GEV", Side.SELL, 10)],
            [pos("GEV", 100)],
            [order("GEV", Side.SELL, 150)],
        )
        assert not result.kept
        assert "exceed the live position" in result.dropped[0][1]

    def test_a_working_order_on_an_unrelated_name_is_ignored(self) -> None:
        result = filter_reduce_only(
            [order("AMD", Side.SELL, 50)],
            [pos("AMD", 100)],
            [order("GLD", Side.BUY, 5)],
        )
        assert result.kept[0].quantity == 50


class TestContractIdentityIsConservative:
    def test_etf_and_equity_spellings_of_one_symbol_are_the_same_exposure(self) -> None:
        """IBKR reports the same fund as ETF here and EQUITY there.

        Treating them as different would let a working exit go uncounted and
        a duplicate sell through.
        """
        result = filter_reduce_only(
            [order("GLD", Side.SELL, 100, asset_class=AssetClass.EQUITY)],
            [pos("GLD", 100, asset_class=AssetClass.ETF)],
            [order("GLD", Side.SELL, 70, asset_class=AssetClass.ETF)],
        )
        assert result.kept[0].quantity == 30

    def test_a_different_currency_is_a_different_instrument(self) -> None:
        result = filter_reduce_only(
            [order("NESN", Side.SELL, 10, currency="CHF")],
            [pos("NESN", 100, currency="EUR")],
            [],
        )
        assert not result.kept

    def test_duplicate_live_positions_make_the_name_unprovable(self) -> None:
        result = filter_reduce_only(
            [order("AMD", Side.SELL, 10)],
            [pos("AMD", 100), pos("AMD", 40)],
            [],
        )
        assert not result.kept
        assert "multiple live positions" in result.dropped[0][1]


class TestMalformedInputFailsClosed:
    def test_an_unreadable_working_order_quantity_poisons_only_its_name(self) -> None:
        bad = order("AMD", Side.SELL, 10).model_copy(update={"quantity": float("nan")})
        result = filter_reduce_only(
            [order("AMD", Side.SELL, 10), order("GLD", Side.SELL, 5)],
            [pos("AMD", 100), pos("GLD", 50)],
            [bad],
        )
        assert [o.instrument.symbol for o in result.kept] == ["GLD"]
        assert result.dropped[0][0] == "AMD"

    def test_a_zero_quantity_plan_order_is_dropped_not_submitted(self) -> None:
        result = filter_reduce_only([order("AMD", Side.SELL, 0)], [pos("AMD", 100)], [])
        assert not result.kept

    def test_a_flat_position_row_is_not_a_position(self) -> None:
        result = filter_reduce_only([order("AMD", Side.SELL, 10)], [pos("AMD", 0.0)], [])
        assert not result.kept
        assert "would open new exposure" in result.dropped[0][1]


class TestTheRealisticHaltedBasket:
    def test_a_mixed_pm_basket_becomes_sells_only(self) -> None:
        """The 2026-08-18 shape: PM wants to rotate; the halt allows only exits."""
        orders = [
            order("GLD", Side.BUY, 200),  # new defensive anchor — blocked
            order("AMD", Side.SELL, 80),  # trim a live long — allowed
            order("MU", Side.SELL, 500),  # oversized vs the holding — trimmed
            order("ITA", Side.BUY, 40),  # add to a live long — blocked
            order("SNDK", Side.SELL, 25),  # full exit — allowed
        ]
        positions = [pos("AMD", 160), pos("MU", 46), pos("ITA", 400), pos("SNDK", 25)]

        result = filter_reduce_only(orders, positions, [])

        assert [(o.instrument.symbol, o.quantity) for o in result.kept] == [
            ("AMD", 80),
            ("MU", 46),
            ("SNDK", 25),
        ]
        assert {sym for sym, _ in result.dropped} == {"GLD", "ITA"}
        assert all(o.side == Side.SELL for o in result.kept)


@pytest.mark.parametrize("qty", [1e-12, -1e-12])
def test_dust_positions_are_treated_as_flat(qty: float) -> None:
    result = filter_reduce_only([order("AMD", Side.SELL, 10)], [pos("AMD", qty)], [])
    assert not result.kept
