"""Tests for the in-memory Simulator broker."""

from __future__ import annotations

import pytest

from trading.core.types import (
    AssetClass,
    Instrument,
    OrderStatus,
    OrderType,
    Side,
)
from trading.execution import BrokerError, Simulator
from tests.execution.conftest import make_bar, make_order


def test_protocol_isinstance() -> None:
    """Simulator satisfies the Broker Protocol at runtime."""
    from trading.execution.base import Broker
    s = Simulator()
    assert isinstance(s, Broker)


def test_requires_connect(aapl, t0) -> None:
    s = Simulator()
    order = make_order(aapl, Side.BUY, 10, created_at=t0)
    with pytest.raises(BrokerError, match="not connected"):
        s.submit_order(order)


def test_market_buy_fills_at_open_plus_slippage(sim, aapl, t0, t1) -> None:
    order = make_order(aapl, Side.BUY, 10, created_at=t0)
    sim.submit_order(order)
    fills = sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=102.0)})
    assert len(fills) == 1
    # 100 * (1 + 2bps) = 100.02
    assert fills[0].price == pytest.approx(100.02, rel=1e-9)
    assert fills[0].quantity == 10
    assert fills[0].commission == pytest.approx(100.02 * 10 * 1e-4, rel=1e-9)


def test_market_sell_fills_at_open_minus_slippage(sim, aapl, t0, t1) -> None:
    order = make_order(aapl, Side.SELL, 10, created_at=t0)
    sim.submit_order(order)
    fills = sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=102.0)})
    assert fills[0].price == pytest.approx(99.98, rel=1e-9)
    assert fills[0].quantity == -10   # negative for sells


def test_position_updates_after_fill(sim, aapl, t0, t1) -> None:
    sim.submit_order(make_order(aapl, Side.BUY, 10, created_at=t0))
    sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=102.0)})
    positions = sim.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 10
    assert positions[0].avg_price == pytest.approx(100.02, rel=1e-9)
    assert positions[0].unrealized_pnl == pytest.approx((102.0 - 100.02) * 10, rel=1e-6)


def test_cash_decreases_on_buy(sim, aapl, t0, t1) -> None:
    sim.submit_order(make_order(aapl, Side.BUY, 10, created_at=t0))
    sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=102.0)})
    acct = sim.get_account()
    # 10 shares at 100.02 + commission.
    expected = 100_000.0 - 10 * 100.02 - 100.02 * 10 * 1e-4
    assert acct.cash == pytest.approx(expected, rel=1e-9)


def test_equity_includes_unrealized_pnl(sim, aapl, t0, t1) -> None:
    sim.submit_order(make_order(aapl, Side.BUY, 10, created_at=t0))
    sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=102.0)})
    acct = sim.get_account()
    # Equity = cash + 10 * 102.
    assert acct.equity == pytest.approx(acct.cash + 10 * 102.0, rel=1e-9)


def test_closing_position_realizes_pnl(sim, aapl, t0, t1, t2) -> None:
    sim.submit_order(make_order(aapl, Side.BUY, 10, created_at=t0))
    sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=100.0)})
    sim.submit_order(make_order(aapl, Side.SELL, 10, created_at=t1))
    sim.step(t2, {"AAPL": make_bar(t2, open=110.0, close=110.0)})
    positions = sim.get_positions()
    assert positions[0].quantity == 0
    # PnL on the closing 10 shares: (sell_price - buy_avg) * quantity.
    expected_realized = (110.0 * (1 - 2e-4) - 100.0 * (1 + 2e-4)) * 10
    assert positions[0].realized_pnl == pytest.approx(expected_realized, rel=1e-6)


def test_flipping_long_to_short(sim, aapl, t0, t1, t2) -> None:
    sim.submit_order(make_order(aapl, Side.BUY, 5, created_at=t0))
    sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=100.0)})
    # Sell 10 — closes 5 long + opens 5 short.
    sim.submit_order(make_order(aapl, Side.SELL, 10, created_at=t1))
    sim.step(t2, {"AAPL": make_bar(t2, open=110.0, close=110.0)})
    positions = sim.get_positions()
    assert positions[0].quantity == pytest.approx(-5.0)
    # The residual short's avg_price is the fill price of the sell.
    assert positions[0].avg_price == pytest.approx(110.0 * (1 - 2e-4), rel=1e-9)


def test_duplicate_order_id_rejected(sim, aapl, t0) -> None:
    o = make_order(aapl, Side.BUY, 1, created_at=t0)
    sim.submit_order(o)
    with pytest.raises(BrokerError, match="duplicate"):
        sim.submit_order(o)


def test_cancel_pending_order(sim, aapl, t0, t1) -> None:
    o = make_order(aapl, Side.BUY, 10, created_at=t0)
    sim.submit_order(o)
    sim.cancel_order(o.client_order_id)
    fills = sim.step(t1, {"AAPL": make_bar(t1, open=100.0)})
    assert fills == []
    assert sim.get_order_status(o.client_order_id) == OrderStatus.CANCELLED


def test_cancel_unknown_order_raises(sim) -> None:
    with pytest.raises(BrokerError, match="unknown"):
        sim.cancel_order("never-existed")


def test_non_market_order_stays_pending(sim, aapl, t0, t1) -> None:
    o = make_order(aapl, Side.BUY, 10, created_at=t0,
                   order_type=OrderType.LIMIT, limit_price=99.0)
    sim.submit_order(o)
    fills = sim.step(t1, {"AAPL": make_bar(t1, open=100.0, close=100.0)})
    assert fills == []
    assert sim.get_order_status(o.client_order_id) == OrderStatus.SUBMITTED


def test_get_account_before_step_raises(sim) -> None:
    with pytest.raises(BrokerError, match="step"):
        sim.get_account()


def test_get_fills_since_filters(sim, aapl, t0, t1, t2) -> None:
    sim.submit_order(make_order(aapl, Side.BUY, 5, created_at=t0))
    sim.step(t1, {"AAPL": make_bar(t1, open=100.0)})
    sim.submit_order(make_order(aapl, Side.BUY, 5, created_at=t1))
    sim.step(t2, {"AAPL": make_bar(t2, open=110.0)})
    after_t2 = sim.get_fills(since=t2)
    assert len(after_t2) == 1
    assert after_t2[0].ts == t2
