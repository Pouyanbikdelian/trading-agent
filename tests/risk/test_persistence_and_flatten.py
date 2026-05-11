"""Halt-state persistence + force_flatten tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.core.types import AssetClass, Instrument, Position, Side
from trading.risk import RiskLimits, RiskManager


def test_halt_state_persists_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "halt.json"
    m1 = RiskManager(RiskLimits(), halt_state_path=path)
    m1.halt("oops")
    # Construct a fresh manager pointing at the same file — it must come up halted.
    m2 = RiskManager(RiskLimits(), halt_state_path=path)
    assert m2.is_halted()
    assert m2.state.reason == "oops"


def test_unhalt_clears_persisted_state(tmp_path: Path) -> None:
    path = tmp_path / "halt.json"
    m1 = RiskManager(RiskLimits(), halt_state_path=path)
    m1.halt("oops")
    m1.unhalt()
    m2 = RiskManager(RiskLimits(), halt_state_path=path)
    assert not m2.is_halted()


def test_halt_state_persists_high_watermark(tmp_path: Path) -> None:
    """A restart in the middle of the day must remember peak equity, else
    the drawdown kill switch resets and we lose protection."""
    from trading.core.types import AccountSnapshot
    path = tmp_path / "halt.json"
    m1 = RiskManager(RiskLimits(max_daily_loss_pct=1.0), halt_state_path=path)
    high = AccountSnapshot(ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                            cash=200_000.0, equity=200_000.0)
    m1.start_of_day(high)
    m1.evaluate_intraday(high)
    assert m1.state.equity_high_watermark == 200_000.0
    m2 = RiskManager(RiskLimits(max_daily_loss_pct=1.0), halt_state_path=path)
    assert m2.state.equity_high_watermark == 200_000.0


def test_force_flatten_generates_closing_orders(mgr) -> None:
    aapl = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    msft = Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY)
    positions = [
        Position(instrument=aapl, quantity=100.0, avg_price=150.0),
        Position(instrument=msft, quantity=-50.0, avg_price=300.0),
    ]
    orders = mgr.force_flatten_orders(positions)
    assert len(orders) == 2
    # Long position -> SELL to close.
    aapl_order = next(o for o in orders if o.instrument.symbol == "AAPL")
    assert aapl_order.side == Side.SELL
    assert aapl_order.quantity == 100.0
    # Short position -> BUY to close.
    msft_order = next(o for o in orders if o.instrument.symbol == "MSFT")
    assert msft_order.side == Side.BUY
    assert msft_order.quantity == 50.0


def test_force_flatten_ignores_zero_positions(mgr) -> None:
    aapl = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    positions = [Position(instrument=aapl, quantity=0.0, avg_price=100.0)]
    assert mgr.force_flatten_orders(positions) == []


def test_force_flatten_unique_order_ids(mgr) -> None:
    aapl = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    msft = Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY)
    positions = [
        Position(instrument=aapl, quantity=10, avg_price=100.0),
        Position(instrument=msft, quantity=10, avg_price=100.0),
    ]
    orders = mgr.force_flatten_orders(positions)
    assert orders[0].client_order_id != orders[1].client_order_id
