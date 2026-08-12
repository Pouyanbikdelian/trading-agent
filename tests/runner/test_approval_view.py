"""The approval card must describe the order plan, not mislead about exposure."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
    TimeInForce,
)
from trading.runner.approval_view import build_plan, format_candidates, format_trade_review

TS = datetime(2026, 8, 12, 17, 40, tzinfo=timezone.utc)


def _order(symbol: str, side: Side, quantity: float) -> Order:
    return Order(
        client_order_id=f"approval-{symbol}-{side.value}",
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY, currency="USD"),
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=TS,
    )


def _account() -> AccountSnapshot:
    cnc = Instrument(symbol="CNC", asset_class=AssetClass.EQUITY, currency="USD")
    ura = Instrument(symbol="URA", asset_class=AssetClass.EQUITY, currency="USD")
    dell = Instrument(symbol="DELL", asset_class=AssetClass.EQUITY, currency="USD")
    return AccountSnapshot(
        ts=TS,
        cash=81_900.0,
        equity=87_900.0,
        base_currency="CHF",
        positions={
            cnc.key: Position(instrument=cnc, quantity=12, avg_price=66.72),
            ura.key: Position(instrument=ura, quantity=15, avg_price=44.97),
            dell.key: Position(instrument=dell, quantity=2, avg_price=439.30),
        },
    )


def test_plan_is_base_currency_correct_and_labels_turnover() -> None:
    orders = [_order("AMD", Side.BUY, 1), _order("URA", Side.BUY, 9), _order("CNC", Side.SELL, 12)]
    prices = {"equity:AMD": 500.0, "equity:URA": 40.0, "equity:CNC": 65.0}

    plan = build_plan(orders, _account(), prices, fx_rates={"USD": 0.8})

    assert plan["gross_turnover"] == pytest.approx(1_312.0)
    assert plan["net_cash_change"] == pytest.approx(64.0)
    assert plan["orders"][0]["context"] == "new position"
    assert plan["orders"][1]["context"] == "adds to 15 held"
    assert plan["orders"][2]["context"] == "closes 12 held"
    assert plan["unchanged_symbols"] == ["DELL"]

    text = format_trade_review({"id": "20260812-abcd", "plan": plan}, include_controls=True)

    assert "CHF 1,312 turnover" in text
    assert "not additional exposure" in text
    assert "*Buy / increase*" in text and "*Sell / reduce*" in text
    assert "new position" in text and "closes 12 held" in text
    assert "Unchanged: DELL" in text


def test_candidate_ladder_is_explicitly_not_the_proposed_trade() -> None:
    text = format_candidates(
        {
            "can_pick": True,
            "candidates": [
                {"symbol": "SNDK", "score": 5.39, "in_basket": False},
                {"symbol": "AMD", "score": 1.34, "in_basket": True},
            ],
        }
    )

    assert "alternatives, not the proposed trade" in text
    assert "SNDK" in text and "AMD" in text
    assert "/pick 1 3 5" in text
