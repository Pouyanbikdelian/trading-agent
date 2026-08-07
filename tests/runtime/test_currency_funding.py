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
