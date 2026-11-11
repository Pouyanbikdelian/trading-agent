"""The no-trade band: drift is not a decision (2026-11-11)."""

from __future__ import annotations

import pytest

from tests.risk.conftest import instruments_dict, signal_from
from trading.core.types import AccountSnapshot, Position, Side
from trading.risk import RiskLimits, RiskManager


def _account(t0, positions):
    return AccountSnapshot(
        ts=t0,
        cash=100_000.0,
        equity=100_000.0,
        positions={f"equity:{p.instrument.symbol}": p for p in positions},
    )


def _mgr(tmp_path, band, **limits):
    return RiskManager(
        RiskLimits(min_rebalance_pct=band, **limits), halt_state_path=tmp_path / "halt.json"
    )


def _plan(mgr, t0, targets, account, *instruments):
    return mgr.signal_to_orders(
        signal_from(t0, targets),
        account=account,
        last_prices={f"equity:{i.symbol}": 100.0 for i in instruments},
        instruments=instruments_dict(*instruments),
    )


def test_off_by_default_every_share_of_drift_still_trades(tmp_path, aapl, t0) -> None:
    account = _account(t0, [Position(instrument=aapl, quantity=48, avg_price=100.0)])
    orders, decisions = _plan(_mgr(tmp_path, 0.0), t0, {"equity:AAPL": 0.05}, account, aapl)
    assert [(o.side, o.quantity) for o in orders] == [(Side.BUY, 2.0)]
    assert not any("no-trade band" in d.reason for d in decisions)


@pytest.mark.parametrize("held", [48, 52])
def test_a_small_adjustment_either_way_is_left_alone_and_said_so(tmp_path, aapl, t0, held) -> None:
    """Two shares of 100 is 0.2% of the desk; the band is 0.5%."""
    account = _account(t0, [Position(instrument=aapl, quantity=held, avg_price=100.0)])
    orders, decisions = _plan(_mgr(tmp_path, 0.005), t0, {"equity:AAPL": 0.05}, account, aapl)
    assert orders == []
    notes = [d.reason for d in decisions if "no-trade band" in d.reason]
    assert len(notes) == 1 and "AAPL" in notes[0]


def test_an_adjustment_outside_the_band_trades_in_full(tmp_path, aapl, t0) -> None:
    account = _account(t0, [Position(instrument=aapl, quantity=40, avg_price=100.0)])
    orders, _ = _plan(_mgr(tmp_path, 0.005), t0, {"equity:AAPL": 0.05}, account, aapl)
    assert [(o.side, o.quantity) for o in orders] == [(Side.BUY, 10.0)]


def test_entries_and_exits_are_decisions_not_drift(tmp_path, aapl, msft, t0) -> None:
    """A 0.3% starter position and the sale of a 0.2% remnant are both under
    the band, and both are what the book was asked to do."""
    account = _account(t0, [Position(instrument=msft, quantity=2, avg_price=100.0)])
    orders, _ = _plan(
        _mgr(tmp_path, 0.005), t0, {"equity:AAPL": 0.003, "equity:MSFT": 0.0}, account, aapl, msft
    )
    assert {(o.instrument.symbol, o.side, o.quantity) for o in orders} == {
        ("AAPL", Side.BUY, 3.0),
        ("MSFT", Side.SELL, 2.0),
    }


def test_a_position_over_its_cap_is_always_brought_back(tmp_path, aapl, t0) -> None:
    """Held 10.3% against a 10% cap: the trim is 0.3%, inside the band,
    and a limit is not left breached to save a commission."""
    account = _account(t0, [Position(instrument=aapl, quantity=103, avg_price=100.0)])
    orders, _ = _plan(_mgr(tmp_path, 0.005), t0, {"equity:AAPL": 0.10}, account, aapl)
    assert [(o.side, o.quantity) for o in orders] == [(Side.SELL, 3.0)]


def test_a_cap_that_scaled_the_basket_exempts_every_sell(tmp_path, aapl, msft, t0) -> None:
    """Gross is capped at 9.8%, so both targets are scaled down. The
    resulting one-share trims are tiny and they are exactly what the cap
    requires."""
    account = _account(
        t0,
        [
            Position(instrument=aapl, quantity=50, avg_price=100.0),
            Position(instrument=msft, quantity=50, avg_price=100.0),
        ],
    )
    mgr = _mgr(tmp_path, 0.005, max_gross_exposure=0.098)
    orders, _ = _plan(mgr, t0, {"equity:AAPL": 0.05, "equity:MSFT": 0.05}, account, aapl, msft)
    assert {(o.instrument.symbol, o.side, o.quantity) for o in orders} == {
        ("AAPL", Side.SELL, 1.0),
        ("MSFT", Side.SELL, 1.0),
    }


def test_the_band_cannot_be_set_wide_enough_to_freeze_the_book() -> None:
    with pytest.raises(ValueError):
        RiskLimits(min_rebalance_pct=0.2)
