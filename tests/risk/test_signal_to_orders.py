"""Tests for the pre-trade path: Signal target weights -> Orders.

Each test pins one scaling rule with a hand-set weight + price + equity
combination so we can assert specific quantities and decisions.
"""

from __future__ import annotations

import pytest

from tests.risk.conftest import instruments_dict, signal_from
from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position, Side


def test_per_position_cap_scales_only_offenders(mgr, aapl, msft, account_100k, t0) -> None:
    sig = signal_from(t0, {"equity:AAPL": 0.25, "equity:MSFT": 0.05})
    prices = {"equity:AAPL": 100.0, "equity:MSFT": 100.0}
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices=prices,
        instruments=instruments_dict(aapl, msft),
    )
    by_sym = {o.instrument.symbol: o for o in orders}
    # AAPL capped from 25% -> 10%: $10k / $100 = 100 shares.
    assert by_sym["AAPL"].quantity == pytest.approx(100.0)
    # MSFT under cap stays at 5%: $5k / $100 = 50 shares.
    assert by_sym["MSFT"].quantity == pytest.approx(50.0)
    # Exactly one scale decision for AAPL.
    scale_decisions = [d for d in decisions if d.action == "scale"]
    assert len(scale_decisions) == 1
    assert scale_decisions[0].scale_factor == pytest.approx(0.10 / 0.25)


def test_gross_exposure_scales_everything(mgr, aapl, msft, account_100k, t0) -> None:
    # Each position at 10% (cap), but six of them = 60% gross... well, our
    # gross cap default is 1.0. Use 12 positions at 10% each = 120% gross,
    # but we only have 2 fixtures. Use a smaller gross cap instead.
    mgr.limits = mgr.limits.model_copy(update={"max_gross_exposure": 0.10})
    sig = signal_from(t0, {"equity:AAPL": 0.10, "equity:MSFT": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices={"equity:AAPL": 100.0, "equity:MSFT": 100.0},
        instruments=instruments_dict(aapl, msft),
    )
    # Gross before scaling: 0.15; after: 0.10. scale = 0.10 / 0.15.
    # AAPL fractional target = 66.666… shares; truncated to 66 because IBKR
    # rejects fractional EQUITY orders via API.
    by_sym = {o.instrument.symbol: o for o in orders}
    assert by_sym["AAPL"].quantity == 66.0
    assert any("gross" in d.reason for d in decisions if d.action == "scale")


def test_net_exposure_scales_everything(mgr, aapl, msft, account_100k, t0) -> None:
    mgr.limits = mgr.limits.model_copy(
        update={
            "max_position_pct": 1.0,
            "max_gross_exposure": 10.0,
            "max_net_exposure": 0.20,
        }
    )
    sig = signal_from(t0, {"equity:AAPL": 0.30, "equity:MSFT": 0.30})  # net 0.60
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices={"equity:AAPL": 100.0, "equity:MSFT": 100.0},
        instruments=instruments_dict(aapl, msft),
    )
    # After scaling by 0.20/0.60, each position is 0.10 weight.
    by_sym = {o.instrument.symbol: o for o in orders}
    assert by_sym["AAPL"].quantity == pytest.approx(100.0)
    assert by_sym["MSFT"].quantity == pytest.approx(100.0)
    assert any("net" in d.reason for d in decisions if d.action == "scale")


def test_sector_cap_scales_only_that_sector(mgr, aapl, msft, xom, account_100k, t0) -> None:
    mgr.limits = mgr.limits.model_copy(update={"max_position_pct": 1.0})
    sig = signal_from(t0, {"equity:AAPL": 0.20, "equity:MSFT": 0.20, "equity:XOM": 0.10})
    sector_map = {"equity:AAPL": "tech", "equity:MSFT": "tech", "equity:XOM": "energy"}
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices={"equity:AAPL": 100.0, "equity:MSFT": 100.0, "equity:XOM": 100.0},
        instruments=instruments_dict(aapl, msft, xom),
        sector_map=sector_map,
    )
    by_sym = {o.instrument.symbol: o for o in orders}
    # Tech sector exposure = 0.40, capped at 0.30 → scale = 0.75.
    # AAPL/MSFT each become 0.15. Energy untouched at 0.10.
    assert by_sym["AAPL"].quantity == pytest.approx(150.0)
    assert by_sym["MSFT"].quantity == pytest.approx(150.0)
    assert by_sym["XOM"].quantity == pytest.approx(100.0)
    sector_decisions = [d for d in decisions if d.action == "scale" and "sector" in d.reason]
    assert len(sector_decisions) == 1


def test_delta_against_existing_position(mgr, aapl, account_100k, t0) -> None:
    # Already long 50 shares; target weight asks for 100 shares.
    pos = Position(instrument=aapl, quantity=50.0, avg_price=100.0)
    account = account_100k.model_copy(update={"positions": {"equity:AAPL": pos}})
    sig = signal_from(t0, {"equity:AAPL": 0.10})  # 100 shares at $100 = $10k
    orders, _ = mgr.signal_to_orders(
        sig,
        account=account,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert len(orders) == 1
    assert orders[0].side == Side.BUY
    assert orders[0].quantity == pytest.approx(50.0)


def test_delta_flips_long_to_short(mgr, aapl, account_100k, t0) -> None:
    pos = Position(instrument=aapl, quantity=100.0, avg_price=100.0)
    account = account_100k.model_copy(update={"positions": {"equity:AAPL": pos}})
    sig = signal_from(t0, {"equity:AAPL": -0.05})  # -50 shares
    orders, _ = mgr.signal_to_orders(
        sig,
        account=account,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    # Must sell 150 shares: close 100 long + open 50 short.
    assert orders[0].side == Side.SELL
    assert orders[0].quantity == pytest.approx(150.0)


def test_no_trade_when_already_at_target(mgr, aapl, account_100k, t0) -> None:
    pos = Position(instrument=aapl, quantity=100.0, avg_price=100.0)
    account = account_100k.model_copy(update={"positions": {"equity:AAPL": pos}})
    sig = signal_from(t0, {"equity:AAPL": 0.10})
    orders, _ = mgr.signal_to_orders(
        sig,
        account=account,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []


def test_missing_instrument_rejected(mgr, aapl, account_100k, t0) -> None:
    sig = signal_from(t0, {"equity:UNKNOWN": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices={"equity:UNKNOWN": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert any(d.action == "reject" for d in decisions)


def test_missing_or_zero_price_rejected(mgr, aapl, account_100k, t0) -> None:
    sig = signal_from(t0, {"equity:AAPL": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices={"equity:AAPL": 0.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert any(d.action == "reject" for d in decisions)


def test_non_positive_equity_rejected(mgr, aapl, t0) -> None:
    account = AccountSnapshot(ts=t0, cash=0.0, equity=0.0)
    sig = signal_from(t0, {"equity:AAPL": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert any(d.action == "reject" for d in decisions)


def test_halted_manager_produces_no_orders(mgr, aapl, account_100k, t0) -> None:
    mgr.halt("manual test halt")
    sig = signal_from(t0, {"equity:AAPL": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=account_100k,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert decisions[0].action == "halt"


# ---------------------------------------------------------------------------
# No-margin enforcement (max_margin_borrowing_pct = 0.0)
#
# These tests prove that a basket which would push any currency cash
# negative is rejected wholesale — the live deployment requires this so
# IBKR doesn't auto-loan USD against a CHF base.
# ---------------------------------------------------------------------------


def test_no_margin_rejects_basket_that_would_overdraw_cash(mgr, aapl, msft, t0) -> None:
    # Start with $50k cash, all in USD. Raise per-position cap so the
    # basket can actually push USD negative; w/o raising it, the per-
    # position scaler trims each name to 10% of equity and the basket
    # fits in cash.
    mgr.limits = mgr.limits.model_copy(
        update={
            "max_position_pct": 0.50,
            "max_gross_exposure": 1.00,
            "max_margin_borrowing_pct": 0.0,
        }
    )
    snap = AccountSnapshot(
        ts=t0,
        cash=50_000.0,
        equity=100_000.0,  # rest is in CHF cash, not represented here
        base_currency="USD",
        cash_by_currency={"USD": 50_000.0, "CHF": 50_000.0},
    )
    # 0.40 + 0.40 = 0.80 gross of equity = $80k in USD orders; only
    # $50k USD cash available → $30k overdraw.
    sig = signal_from(t0, {"equity:AAPL": 0.40, "equity:MSFT": 0.40})
    prices = {"equity:AAPL": 100.0, "equity:MSFT": 100.0}
    orders, decisions = mgr.signal_to_orders(
        sig, account=snap, last_prices=prices, instruments=instruments_dict(aapl, msft)
    )
    assert orders == [], "basket must be rejected wholesale when it would overdraw"
    reject = [d for d in decisions if d.action == "reject"]
    assert reject, "expected a reject decision"
    assert "no-margin" in reject[0].reason.lower()
    assert "USD" in reject[0].reason


def test_no_margin_allows_basket_when_cash_sufficient(mgr, aapl, msft, t0) -> None:
    mgr.limits = mgr.limits.model_copy(update={"max_margin_borrowing_pct": 0.0})
    snap = AccountSnapshot(
        ts=t0,
        cash=100_000.0,
        equity=100_000.0,
        base_currency="USD",
        cash_by_currency={"USD": 100_000.0},
    )
    sig = signal_from(t0, {"equity:AAPL": 0.05, "equity:MSFT": 0.05})  # 10% gross
    orders, _ = mgr.signal_to_orders(
        sig,
        account=snap,
        last_prices={"equity:AAPL": 100.0, "equity:MSFT": 100.0},
        instruments=instruments_dict(aapl, msft),
    )
    assert len(orders) == 2, "well-funded basket should pass margin check"


def test_margin_budget_allows_partial_overdraw(mgr, aapl, t0) -> None:
    """With MAX_MARGIN_BORROWING_PCT=0.50, USD can go to -50k against
    100k equity. Basket that lands at -30k USD must be allowed."""
    mgr.limits = mgr.limits.model_copy(
        update={"max_margin_borrowing_pct": 0.50, "max_gross_exposure": 1.50}
    )
    snap = AccountSnapshot(
        ts=t0,
        cash=20_000.0,
        equity=100_000.0,
        base_currency="USD",
        cash_by_currency={"USD": 20_000.0},
    )
    # Buying ~50k of stock with only 20k cash → USD ends at -30k. With
    # the 50% budget that's allowed.
    sig = signal_from(t0, {"equity:AAPL": 0.10})
    orders, _ = mgr.signal_to_orders(
        sig,
        account=snap,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert len(orders) == 1
    assert orders[0].quantity > 0


def test_cross_currency_overdraw_called_out_explicitly(mgr, t0) -> None:
    """A CHF-base account buying USD stocks with no USD cash must be
    rejected — reproduces today's prod scenario (CHF +840k, USD 0).
    The rejection message should name USD so the operator knows to FX."""
    mgr.limits = mgr.limits.model_copy(
        update={"max_position_pct": 1.0, "max_gross_exposure": 1.0, "max_margin_borrowing_pct": 0.0}
    )
    usd_stock = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY, currency="USD")
    snap = AccountSnapshot(
        ts=t0,
        cash=840_000.0,
        equity=840_000.0,
        base_currency="CHF",
        cash_by_currency={"CHF": 840_000.0},  # no USD at all
    )
    sig = signal_from(t0, {"equity:AAPL": 0.50})
    orders, decisions = mgr.signal_to_orders(
        sig,
        account=snap,
        last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": usd_stock},
    )
    assert orders == []
    reject = next((d for d in decisions if d.action == "reject"), None)
    assert reject is not None
    assert "USD" in reject.reason, "rejection should name the deficit currency"
