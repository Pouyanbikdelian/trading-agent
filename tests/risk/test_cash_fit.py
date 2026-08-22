"""A basket that costs more than the wallet holds should get smaller.

Until 2026-08-21 it got deleted. The sleeve had just been raised, the PM
asked for a USD 48.5k book against USD 39.0k of cash, and the cycle
refused all eleven names — the sells included — over a USD 9,542 gap.

Two properties matter more than any individual case here:

  * The trim can only ever REMOVE exposure. Whole-share quantities are
    floored, so no arrangement of prices can make a "fit" re-breach the
    limit it was called to satisfy. `test_the_fit_never_re_breaches`
    checks that against the real no-margin check.
  * Relative conviction survives. Every buy in a short currency scales by
    the same factor, so a 12% name stays three times a 4% name.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)
from trading.risk.cash_fit import REASON_PREFIX, fit_orders_to_cash

NOW = datetime(2026, 8, 21, 20, 6, tzinfo=timezone.utc)


def inst(symbol: str, *, currency: str = "USD", asset_class: AssetClass = AssetClass.EQUITY):
    return Instrument(symbol=symbol, asset_class=asset_class, currency=currency, exchange="SMART")


def order(symbol: str, side: Side, qty: float, *, currency: str = "USD", **kw) -> Order:
    return Order(
        client_order_id=f"{symbol}-{side.value}-{qty:g}",
        instrument=inst(symbol, currency=currency, **kw),
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=NOW,
    )


def fit(orders, cash, prices, *, base="CHF", allowed_debit=0.0, safety_margin=0.0):
    """Arithmetic tests pin ``safety_margin=0`` so the expected numbers are
    readable; the buffer itself has its own class below."""
    return fit_orders_to_cash(
        orders,
        cash_by_currency=cash,
        base_currency=base,
        last_prices=prices,
        allowed_debit=allowed_debit,
        safety_margin=safety_margin,
    )


def qty_by_symbol(result) -> dict[str, float]:
    return {o.instrument.symbol: o.quantity for o in result.orders}


class TestAnAffordableBasketIsUntouched:
    def test_exact_fit_passes_whole(self) -> None:
        orders = [order("AMD", Side.BUY, 10)]
        result = fit(orders, {"USD": 1_000.0}, {"equity:SMART:AMD": 100.0})

        assert not result.changed
        assert result.orders == orders

    def test_a_currency_with_no_buys_is_never_touched(self) -> None:
        """Sells cannot overdraw anything — they are the cure, not the cause."""
        orders = [order("NESN", Side.SELL, 50, currency="CHF")]
        result = fit(orders, {"CHF": 0.0}, {"equity:SMART:NESN": 100.0})

        assert not result.changed
        assert result.orders == orders


class TestAnUnaffordableBasketShrinks:
    def test_buys_scale_by_the_shortfall_ratio(self) -> None:
        """The live case: USD 4,000 of buys against USD 3,200 of cash."""
        orders = [order("AMD", Side.BUY, 20), order("MU", Side.BUY, 20)]
        result = fit(
            orders, {"USD": 3_200.0}, {"equity:SMART:AMD": 100.0, "equity:SMART:MU": 100.0}
        )

        assert result.scaled == {"USD": pytest.approx(0.8)}
        assert qty_by_symbol(result) == {"AMD": 16.0, "MU": 16.0}

    def test_relative_weights_are_preserved(self) -> None:
        """A 3x position stays a 3x position at the smaller size."""
        orders = [order("URA", Side.BUY, 90), order("IBB", Side.BUY, 30)]
        result = fit(
            orders, {"USD": 6_000.0}, {"equity:SMART:URA": 100.0, "equity:SMART:IBB": 100.0}
        )

        sized = qty_by_symbol(result)
        assert sized["URA"] == 3 * sized["IBB"]

    def test_sells_are_left_at_full_size(self) -> None:
        """Trimming a sell would keep the cash it was about to raise."""
        orders = [order("AMD", Side.SELL, 8), order("URA", Side.BUY, 100)]
        result = fit(orders, {"USD": 0.0}, {"equity:SMART:AMD": 100.0, "equity:SMART:URA": 100.0})

        assert qty_by_symbol(result)["AMD"] == 8.0
        # 800 of proceeds funds 8 of the 100 URA shares.
        assert qty_by_symbol(result)["URA"] == 8.0

    def test_a_name_that_floors_to_zero_is_dropped_and_named(self) -> None:
        orders = [order("AMD", Side.BUY, 100), order("BRKA", Side.BUY, 1)]
        result = fit(
            orders,
            {"USD": 5_000.0},
            {"equity:SMART:AMD": 100.0, "equity:SMART:BRKA": 1_000.0},
        )

        assert "BRKA" not in qty_by_symbol(result)
        assert ("BRKA", "USD") in result.dropped
        assert "BRKA" in result.reasons()[0]

    def test_zero_budget_drops_every_buy_but_keeps_the_sells(self) -> None:
        orders = [order("AMD", Side.BUY, 10), order("MU", Side.SELL, 5)]
        result = fit(orders, {"USD": 0.0}, {"equity:SMART:MU": 0.0, "equity:SMART:AMD": 100.0})

        # MU is unpriceable, so it funds nothing — AMD has no budget at all.
        assert qty_by_symbol(result) == {"MU": 5.0}

    def test_a_negative_balance_does_not_produce_a_negative_order(self) -> None:
        orders = [order("AMD", Side.BUY, 10)]
        result = fit(orders, {"USD": -5_000.0}, {"equity:SMART:AMD": 100.0})

        assert result.orders == []
        assert result.scaled["USD"] == 0.0


class TestFlooringNeverRoundsUp:
    @pytest.mark.parametrize("cash", [199.0, 250.0, 299.0])
    def test_whole_shares_round_down(self, cash: float) -> None:
        """At 2.5 shares affordable the answer is 2, never 3."""
        result = fit([order("AMD", Side.BUY, 10)], {"USD": cash}, {"equity:SMART:AMD": 100.0})

        bought = qty_by_symbol(result)["AMD"]
        assert bought == float(int(cash // 100))
        assert bought * 100.0 <= cash

    def test_fractional_instruments_keep_their_fraction(self) -> None:
        """Crypto and FX size continuously; flooring them throws away value."""
        result = fit(
            [order("BTC", Side.BUY, 1.0, asset_class=AssetClass.CRYPTO)],
            {"USD": 5_000.0},
            {"crypto:SMART:BTC": 10_000.0},
        )

        assert qty_by_symbol(result)["BTC"] == pytest.approx(0.5)


class TestEachCurrencyIsFittedAlone:
    def test_a_usd_shortfall_does_not_shrink_an_affordable_chf_order(self) -> None:
        orders = [order("AMD", Side.BUY, 20), order("NESN", Side.BUY, 10, currency="CHF")]
        result = fit(
            orders,
            {"USD": 1_000.0, "CHF": 50_000.0},
            {"equity:SMART:AMD": 100.0, "equity:SMART:NESN": 100.0},
        )

        assert set(result.scaled) == {"USD"}
        assert qty_by_symbol(result)["NESN"] == 10.0
        assert qty_by_symbol(result)["AMD"] == 10.0

    def test_missing_currency_is_treated_as_no_cash(self) -> None:
        result = fit([order("AMD", Side.BUY, 10)], {"CHF": 90_000.0}, {"equity:SMART:AMD": 100.0})

        assert result.orders == []


class TestUnpriceableOrdersAreLeftAlone:
    def test_a_zero_price_buy_passes_through_unchanged(self) -> None:
        """The no-margin check ignores it too; guessing here would only
        change the basket without changing the verdict."""
        orders = [order("AMD", Side.BUY, 10), order("XXX", Side.BUY, 5)]
        result = fit(orders, {"USD": 500.0}, {"equity:SMART:AMD": 100.0})

        assert qty_by_symbol(result)["XXX"] == 5.0
        assert qty_by_symbol(result)["AMD"] == 5.0


class TestTheReportedReason:
    def test_it_carries_the_prefix_the_cycle_matches_on(self) -> None:
        result = fit([order("AMD", Side.BUY, 10)], {"USD": 500.0}, {"equity:SMART:AMD": 100.0})

        assert result.reasons()[0].startswith(REASON_PREFIX)
        assert "50%" in result.reasons()[0]

    def test_one_line_per_trimmed_currency(self) -> None:
        orders = [order("AMD", Side.BUY, 20), order("NESN", Side.BUY, 20, currency="CHF")]
        result = fit(
            orders,
            {"USD": 1_000.0, "CHF": 1_000.0},
            {"equity:SMART:AMD": 100.0, "equity:SMART:NESN": 100.0},
        )

        assert len(result.reasons()) == 2


class TestThroughTheRiskManager:
    """The unit above is only useful if the manager actually reaches it."""

    def _manager(self, tmp_path, **limit_kw):
        from trading.risk.limits import RiskLimits
        from trading.risk.manager import RiskManager

        return RiskManager(
            RiskLimits(
                max_position_pct=1.0,
                max_gross_exposure=1.0,
                max_margin_borrowing_pct=0.0,
                **limit_kw,
            ),
            halt_state_path=tmp_path / "halt.json",
        )

    def _account(self) -> AccountSnapshot:
        return AccountSnapshot(
            ts=NOW,
            cash=56_000.0,
            equity=84_312.0,
            positions={},
            base_currency="CHF",
            cash_by_currency={"CHF": 25_011.0, "USD": 39_010.0},
        )

    def test_the_fit_never_re_breaches(self, tmp_path) -> None:
        """Property check: whatever the fit returns must pass the very
        check that rejected the original."""
        manager = self._manager(tmp_path)
        prices = {"equity:SMART:AMD": 137.0, "equity:SMART:URA": 38.7}
        orders = [order("AMD", Side.BUY, 300), order("URA", Side.BUY, 400)]

        fitted, decisions = manager._fit_to_cash(orders, self._account(), prices)

        assert manager._check_no_margin(fitted, self._account(), prices) is None
        assert any(d.action == "scale" for d in decisions)

    def test_it_is_off_when_the_limit_says_so(self, tmp_path) -> None:
        manager = self._manager(tmp_path, fit_orders_to_cash=False)

        assert manager.limits.fit_orders_to_cash is False

    def test_a_basket_that_reduces_an_existing_overdraft_is_allowed(self, tmp_path) -> None:
        """An already-negative balance used to refuse every basket, including
        the sells that would have dug it out. That is a trap with no exit."""
        manager = self._manager(tmp_path)
        account = AccountSnapshot(
            ts=NOW,
            cash=10_000.0,
            equity=50_000.0,
            positions={},
            base_currency="CHF",
            cash_by_currency={"CHF": 20_000.0, "USD": -8_000.0},
        )
        sells = [order("AMD", Side.SELL, 10)]

        assert manager._check_no_margin(sells, account, {"equity:SMART:AMD": 100.0}) is None

    def test_a_basket_that_deepens_an_existing_overdraft_is_still_refused(self, tmp_path) -> None:
        manager = self._manager(tmp_path)
        account = AccountSnapshot(
            ts=NOW,
            cash=10_000.0,
            equity=50_000.0,
            positions={},
            base_currency="CHF",
            cash_by_currency={"CHF": 20_000.0, "USD": -8_000.0},
        )
        buys = [order("AMD", Side.BUY, 10)]

        assert manager._check_no_margin(buys, account, {"equity:SMART:AMD": 100.0}) is not None


class TestTheWholePathFromSignalToOrders:
    """The regression that started this: eleven names, one 20% shortfall,
    and an empty basket. `signal_to_orders` is what the cycle calls."""

    def _manager(self, tmp_path, **limit_kw):
        from trading.risk.limits import RiskLimits
        from trading.risk.manager import RiskManager

        return RiskManager(
            RiskLimits(
                max_position_pct=0.5,
                max_gross_exposure=1.0,
                max_margin_borrowing_pct=0.0,
                **limit_kw,
            ),
            halt_state_path=tmp_path / "halt.json",
        )

    def _inputs(self):
        from trading.core.types import Signal

        instruments = {i.key: i for i in (inst("AMD"), inst("URA"))}
        prices = {inst("AMD").key: 100.0, inst("URA").key: 40.0}
        account = AccountSnapshot(
            ts=NOW,
            cash=50_000.0,
            equity=80_000.0,
            positions={},
            base_currency="CHF",
            cash_by_currency={"CHF": 40_000.0, "USD": 20_000.0},
        )
        # 50% of an 80k account = CHF 40k of USD stock against USD 20k cash.
        signal = Signal(
            ts=NOW,
            strategy="test",
            target_weights={inst("AMD").key: 0.25, inst("URA").key: 0.25},
        )
        return signal, account, prices, instruments

    def test_the_basket_arrives_trimmed_instead_of_empty(self, tmp_path) -> None:
        signal, account, prices, instruments = self._inputs()

        orders, decisions = self._manager(tmp_path).signal_to_orders(
            signal,
            account=account,
            last_prices=prices,
            instruments=instruments,
            fx_rates={"USD": 1.0},
        )

        assert orders, "a fundable subset existed; the basket should not be empty"
        assert {o.instrument.symbol for o in orders} == {"AMD", "URA"}
        spent = sum(o.quantity * prices[o.instrument.key] for o in orders)
        assert spent <= 20_000.0
        assert any(d.action == "scale" and REASON_PREFIX in d.reason for d in decisions)

    def test_with_the_fit_disabled_it_is_still_refused_outright(self, tmp_path) -> None:
        signal, account, prices, instruments = self._inputs()

        orders, decisions = self._manager(tmp_path, fit_orders_to_cash=False).signal_to_orders(
            signal,
            account=account,
            last_prices=prices,
            instruments=instruments,
            fx_rates={"USD": 1.0},
        )

        assert orders == []
        assert any(d.action == "reject" and "no-margin" in d.reason for d in decisions)


class TestTheSafetyBuffer:
    """Sizing to the last franc is how a fitted basket still overdraws.

    The trim happens at plan time; the market order may not reach the
    broker until the operator has finished reading the approval card.
    """

    def test_the_default_leaves_some_cash_unspent(self) -> None:
        from trading.risk.cash_fit import CASH_SAFETY_MARGIN

        result = fit(
            [order("AMD", Side.BUY, 100)],
            {"USD": 10_000.0},
            {inst("AMD").key: 1.0},
            safety_margin=CASH_SAFETY_MARGIN,
        )

        # 100 shares at $1 is affordable outright, so nothing is trimmed.
        assert not result.changed

    def test_it_bites_only_on_a_basket_that_was_already_short(self) -> None:
        from trading.risk.cash_fit import CASH_SAFETY_MARGIN

        result = fit(
            [order("AMD", Side.BUY, 1_000)],
            {"USD": 1_000.0},
            {inst("AMD").key: 1.0},
            safety_margin=CASH_SAFETY_MARGIN,
        )

        bought = qty_by_symbol(result)["AMD"]
        assert bought == 990.0
        assert bought < 1_000.0, "a full-cash fit leaves no room for a tick up"

    def test_a_nonsense_margin_cannot_invert_the_basket(self) -> None:
        result = fit(
            [order("AMD", Side.BUY, 10)],
            {"USD": 500.0},
            {inst("AMD").key: 100.0},
            safety_margin=5.0,
        )

        assert result.orders == []
