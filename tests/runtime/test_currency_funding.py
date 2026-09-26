"""Pre-cycle funding check: enough USD to actually buy US stocks.

The account is CHF-based. US equities settle in USD. Buying them from a
CHF balance creates a USD debit — borrowing — and
``max_margin_borrowing_pct = 0.0`` makes the risk manager reject any order
that pushes a currency below zero.

So the failure is not an error. It is a cycle that runs cleanly, proposes
a full basket, has every order refused, and completes having bought
nothing. In a position report that is identical to a strategy that saw no
opportunities. Paper never surfaced it because paper accumulated USD over
months of trading; the live account starts from CHF.

The second-order requirement is that this check must not cry wolf. It
fires an hour before every cycle, so a false alarm trains the operator to
ignore the one that matters — hence every "cannot tell" path degrades to
silence rather than to a warning.
"""

from __future__ import annotations

import pytest

from trading.runtime.broker_ready import (
    FUNDING_BUFFER,
    check_trade_currency_funding,
    format_funding_alert,
)


class _Snap:
    def __init__(self, equity: float) -> None:
        self.equity = equity


class _Broker:
    """CHF-based account. get_fx_rates quotes one unit of X in base."""

    def __init__(
        self,
        *,
        equity: float = 88_000.0,
        balances: dict[str, float] | None = None,
        rates: dict[str, float] | None = None,
        balances_exc: Exception | None = None,
        account_exc: Exception | None = None,
    ) -> None:
        self._equity = equity
        self._balances = {"CHF": 88_000.0} if balances is None else balances
        self._rates = {"CHF": 1.0, "USD": 0.808054} if rates is None else rates
        self._balances_exc = balances_exc
        self._account_exc = account_exc

    def get_account(self) -> _Snap:
        if self._account_exc:
            raise self._account_exc
        return _Snap(self._equity)

    def get_balances(self) -> dict[str, float]:
        if self._balances_exc:
            raise self._balances_exc
        return self._balances

    def get_fx_rates(self) -> dict[str, float]:
        return self._rates


class TestTheFailureItExistsToCatch:
    def test_a_chf_only_account_is_flagged(self) -> None:
        """The live-day scenario: funded in CHF, about to buy US stocks."""
        r = check_trade_currency_funding(
            _Broker(balances={"CHF": 88_000.0, "USD": 0.0}), gross_exposure_pct=0.11
        )
        assert r["ok"] is False
        assert r["shortfall"] > 0

    def test_ample_usd_passes_quietly(self) -> None:
        r = check_trade_currency_funding(
            _Broker(balances={"CHF": 70_000.0, "USD": 25_000.0}), gross_exposure_pct=0.11
        )
        assert r["ok"] is True and r["shortfall"] == 0

    def test_the_requirement_is_converted_into_the_quote_currency(self) -> None:
        """88k CHF x 11% x buffer, expressed in USD at 0.808 CHF per USD."""
        r = check_trade_currency_funding(
            _Broker(balances={"CHF": 88_000.0, "USD": 0.0}), gross_exposure_pct=0.11
        )
        expected = 88_000.0 * 0.11 * FUNDING_BUFFER / 0.808054
        assert r["required"] == pytest.approx(expected, rel=1e-6)

    def test_a_buffer_is_applied(self) -> None:
        """Sizing to exactly the cash on hand leaves no room for the last
        order once prices move between the check and the cycle."""
        exact = check_trade_currency_funding(
            _Broker(balances={"USD": 88_000.0 * 0.11 / 0.808054}, rates={"USD": 0.808054}),
            gross_exposure_pct=0.11,
        )
        assert exact["ok"] is False  # would have passed without the buffer

    def test_a_bigger_sleeve_needs_more_usd(self) -> None:
        # 88k CHF at 0.808 CHF/USD: 11% gross needs ~12.6k USD, 50% ~57.2k.
        small = check_trade_currency_funding(
            _Broker(balances={"USD": 15_000.0}), gross_exposure_pct=0.11
        )
        large = check_trade_currency_funding(
            _Broker(balances={"USD": 15_000.0}), gross_exposure_pct=0.50
        )
        assert small["ok"] is True and large["ok"] is False


class TestDoesNotCryWolf:
    """It runs before every cycle. A false alarm is expensive — it teaches
    the operator to ignore the true one."""

    def test_unavailable_balances_degrade_to_silence(self) -> None:
        r = check_trade_currency_funding(
            _Broker(balances_exc=ConnectionError("no")), gross_exposure_pct=0.11
        )
        assert r["ok"] is True and "skipped" in r["reason"]

    def test_unavailable_account_degrades_to_silence(self) -> None:
        r = check_trade_currency_funding(
            _Broker(account_exc=TimeoutError("no")), gross_exposure_pct=0.11
        )
        assert r["ok"] is True and "skipped" in r["reason"]

    def test_a_broker_with_no_usd_line_is_not_an_alarm(self) -> None:
        """No USD balance line at all is a different thing from a zero
        balance — most likely a broker that does not report per-currency
        cash, and guessing would produce a weekly false alarm."""
        r = check_trade_currency_funding(
            _Broker(balances={"CHF": 88_000.0}), gross_exposure_pct=0.11
        )
        assert r["ok"] is True and "no USD balance line" in r["reason"]

    def test_zero_equity_is_skipped(self) -> None:
        r = check_trade_currency_funding(_Broker(equity=0.0), gross_exposure_pct=0.11)
        assert r["ok"] is True and "skipped" in r["reason"]

    def test_missing_fx_rate_assumes_parity_rather_than_failing(self) -> None:
        r = check_trade_currency_funding(
            _Broker(balances={"USD": 0.0}, rates={}), gross_exposure_pct=0.11
        )
        assert r["required"] == pytest.approx(88_000.0 * 0.11 * FUNDING_BUFFER)

    def test_never_raises(self) -> None:
        class Hostile:
            def get_account(self):
                raise BaseExceptionGroup("weird", [ValueError("x")])  # noqa: F821

        assert check_trade_currency_funding(Hostile(), gross_exposure_pct=0.11)["ok"] is True


class TestTheAlert:
    def test_it_names_the_amount_to_convert(self) -> None:
        """'Insufficient USD' without a number leaves the operator doing
        arithmetic against a deadline."""
        r = check_trade_currency_funding(
            _Broker(balances={"CHF": 88_000.0, "USD": 1_000.0}), gross_exposure_pct=0.11
        )
        msg = format_funding_alert(r, minutes_to_cycle=60)
        assert "cycle in 60 min" in msg
        assert "short" in msg and "Client Portal" in msg
        assert "MAX_MARGIN_BORROWING_PCT" in msg

    def test_it_explains_the_silent_symptom(self) -> None:
        """The operator has to recognise the failure if it happens anyway."""
        r = check_trade_currency_funding(_Broker(balances={"USD": 0.0}), gross_exposure_pct=0.11)
        assert "bought nothing" in format_funding_alert(r, minutes_to_cycle=60)


# ------------------------------------------- the 2026-09-25 false alarm


class _LiveBroker(_Broker):
    """The live book on 2026-09-25: 85k CHF NetLiq, of which ~31k CHF is the
    operator's pinned NVDA; the desk is cash, 42,740 of it in USD."""

    RATE = 0.818

    def get_account(self):
        from datetime import datetime, timezone

        from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position

        nvda = Instrument(symbol="NVDA", asset_class=AssetClass.EQUITY, currency="USD")
        pos = Position(instrument=nvda, quantity=200, avg_price=150.0, unrealized_pnl=200 * 40.0)
        pinned_chf = 200 * 190.0 * self.RATE  # 31,084 CHF
        cash_chf = 42_740 * self.RATE + 18_900
        return AccountSnapshot(
            ts=datetime(2026, 9, 25, 18, tzinfo=timezone.utc),
            cash=cash_chf,
            equity=cash_chf + pinned_chf,
            positions={nvda.key: pos},
            base_currency="CHF",
            fx_rates={"USD": self.RATE},
        )


def _live(**kw):
    b = _LiveBroker(balances={"CHF": 18_900.0, "USD": 42_740.0}, rates={"USD": 0.818})
    return check_trade_currency_funding(b, gross_exposure_pct=0.70, **kw)


class TestItSizesTheDeskNotTheAccount:
    def test_pinned_holdings_are_not_money_the_desk_can_spend(self) -> None:
        account = _live()
        desk = _live(held_symbols={"NVDA"})
        assert account["book"] == "account" and desk["book"] == "desk"
        assert desk["book_equity"] == pytest.approx(account["equity"] - 200 * 190.0 * 0.818)
        assert desk["required"] < account["required"]
        # 53.9k CHF desk x 70% x buffer at 0.818: ~48.4k USD, not ~76k.
        assert desk["required"] == pytest.approx(
            desk["book_equity"] * 0.70 * FUNDING_BUFFER / 0.818, rel=1e-9
        )

    def test_a_desk_that_cannot_be_valued_falls_back_to_the_account_and_says_so(self) -> None:
        r = check_trade_currency_funding(
            _Broker(balances={"USD": 0.0}), gross_exposure_pct=0.11, held_symbols={"NVDA"}
        )
        assert r["book"].startswith("account (desk valuation unavailable")
        assert r["required"] == pytest.approx(88_000.0 * 0.11 * FUNDING_BUFFER / 0.808054)


class TestTheAmountIsInTheCurrencyYouConvertFrom:
    def test_the_usd_gap_is_priced_in_francs(self) -> None:
        """'convert about 34,247 CHF' was the USD shortfall x 1.02."""
        r = _live(held_symbols={"NVDA"})
        assert r["shortfall_base"] == pytest.approx(r["shortfall"] * 0.818)
        msg = format_funding_alert(r, minutes_to_cycle=60)
        assert f"`{r['shortfall_base'] * 1.02:,.0f}` CHF to USD" in msg
        assert f"`{r['shortfall'] * 1.02:,.0f}` CHF" not in msg


class TestWithFitToCashTheBasketShrinksItIsNotRefused:
    def test_the_alert_says_the_basket_is_scaled_not_rejected(self) -> None:
        r = _live(held_symbols={"NVDA"}, fit_to_cash=True)
        msg = format_funding_alert(r, minutes_to_cycle=60)
        assert "scaled down by the same factor" in msg
        assert "bought nothing" not in msg and "REJECT" not in msg
        assert "Only if you want a full basket" in msg
        assert "your pinned holdings excluded" in msg

    def test_it_says_how_much_of_the_desk_the_usd_can_buy(self) -> None:
        r = _live(held_symbols={"NVDA"}, fit_to_cash=True)
        assert r["coverage"] == pytest.approx(42_740 * 0.818 / r["book_equity"])
        assert f"about {r['coverage']:.0%} of the desk" in format_funding_alert(
            r, minutes_to_cycle=60
        )
