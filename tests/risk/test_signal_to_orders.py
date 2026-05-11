"""Tests for the pre-trade path: Signal target weights -> Orders.

Each test pins one scaling rule with a hand-set weight + price + equity
combination so we can assert specific quantities and decisions.
"""

from __future__ import annotations

import pytest

from trading.core.types import AccountSnapshot, Instrument, Position, Side
from tests.risk.conftest import instruments_dict, signal_from


def test_per_position_cap_scales_only_offenders(mgr, aapl, msft, account_100k, t0) -> None:
    sig = signal_from(t0, {"equity:AAPL": 0.25, "equity:MSFT": 0.05})
    prices = {"equity:AAPL": 100.0, "equity:MSFT": 100.0}
    orders, decisions = mgr.signal_to_orders(
        sig, account=account_100k, last_prices=prices,
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
        sig, account=account_100k,
        last_prices={"equity:AAPL": 100.0, "equity:MSFT": 100.0},
        instruments=instruments_dict(aapl, msft),
    )
    # Gross before scaling: 0.15; after: 0.10. scale = 0.10 / 0.15.
    by_sym = {o.instrument.symbol: o for o in orders}
    assert by_sym["AAPL"].quantity == pytest.approx((0.10 * 0.10 / 0.15 * 100_000) / 100, rel=1e-9)
    assert any("gross" in d.reason for d in decisions if d.action == "scale")


def test_net_exposure_scales_everything(mgr, aapl, msft, account_100k, t0) -> None:
    mgr.limits = mgr.limits.model_copy(update={
        "max_position_pct": 1.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 0.20,
    })
    sig = signal_from(t0, {"equity:AAPL": 0.30, "equity:MSFT": 0.30})  # net 0.60
    orders, decisions = mgr.signal_to_orders(
        sig, account=account_100k,
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
        sig, account=account_100k,
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
    sig = signal_from(t0, {"equity:AAPL": 0.10})   # 100 shares at $100 = $10k
    orders, _ = mgr.signal_to_orders(
        sig, account=account, last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert len(orders) == 1
    assert orders[0].side == Side.BUY
    assert orders[0].quantity == pytest.approx(50.0)


def test_delta_flips_long_to_short(mgr, aapl, account_100k, t0) -> None:
    pos = Position(instrument=aapl, quantity=100.0, avg_price=100.0)
    account = account_100k.model_copy(update={"positions": {"equity:AAPL": pos}})
    sig = signal_from(t0, {"equity:AAPL": -0.05})   # -50 shares
    orders, _ = mgr.signal_to_orders(
        sig, account=account, last_prices={"equity:AAPL": 100.0},
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
        sig, account=account, last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []


def test_missing_instrument_rejected(mgr, aapl, account_100k, t0) -> None:
    sig = signal_from(t0, {"equity:UNKNOWN": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig, account=account_100k, last_prices={"equity:UNKNOWN": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert any(d.action == "reject" for d in decisions)


def test_missing_or_zero_price_rejected(mgr, aapl, account_100k, t0) -> None:
    sig = signal_from(t0, {"equity:AAPL": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig, account=account_100k, last_prices={"equity:AAPL": 0.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert any(d.action == "reject" for d in decisions)


def test_non_positive_equity_rejected(mgr, aapl, t0) -> None:
    account = AccountSnapshot(ts=t0, cash=0.0, equity=0.0)
    sig = signal_from(t0, {"equity:AAPL": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig, account=account, last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert any(d.action == "reject" for d in decisions)


def test_halted_manager_produces_no_orders(mgr, aapl, account_100k, t0) -> None:
    mgr.halt("manual test halt")
    sig = signal_from(t0, {"equity:AAPL": 0.05})
    orders, decisions = mgr.signal_to_orders(
        sig, account=account_100k, last_prices={"equity:AAPL": 100.0},
        instruments={"equity:AAPL": aapl},
    )
    assert orders == []
    assert decisions[0].action == "halt"
