"""The worked examples on an approval card must use that card's symbols.

Reported 2026-08-19: a basket of AMD and MU rendered
``Edit: /approve only AMD DD · /approve all except XLV``. Neither DD nor
XLV was in the plan, and the approval path refuses a symbol that isn't —
so the instruction the card gave could not be followed.
"""

from __future__ import annotations

from trading.runner.approval_view import format_trade_review

PENDING = {
    "id": "20260819abcdef",
    "plan": {
        "base_currency": "CHF",
        "order_count": 2,
        "gross_turnover": 4402.0,
        "turnover_pct": 5.2,
        "net_cash_change": -4402.0,
        "orders": [
            {"symbol": "AMD", "side": "SELL", "quantity": 6, "notional_base": 2371.0},
            {"symbol": "MU", "side": "SELL", "quantity": 3, "notional_base": 2030.0},
        ],
    },
}


def _one_order_pending() -> dict:
    single = {**PENDING, "plan": {**PENDING["plan"], "orders": PENDING["plan"]["orders"][:1]}}
    return single


class TestTheEditExamplesComeFromThePlan:
    def test_it_names_symbols_that_are_actually_in_the_basket(self) -> None:
        text = format_trade_review(PENDING, include_controls=True)

        assert "/approve only AMD" in text
        assert "/approve all except MU" in text

    def test_it_never_names_a_symbol_outside_the_basket(self) -> None:
        """`/approve only DD` is refused outright — never suggest it."""
        text = format_trade_review(PENDING, include_controls=True)

        assert " DD" not in text
        assert "XLV" not in text

    def test_a_single_change_offers_scaling_instead_of_filtering(self) -> None:
        """With one name, "only X" is the whole plan and "except X" is empty."""
        text = format_trade_review(_one_order_pending(), include_controls=True)

        assert "/approve only" not in text
        assert "/approve all except" not in text
        assert "/approve 80" in text

    def test_controls_are_absent_when_not_requested(self) -> None:
        text = format_trade_review(PENDING)

        assert "Approve as shown" not in text
