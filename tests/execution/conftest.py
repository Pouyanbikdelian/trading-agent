"""Shared fixtures for execution tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AssetClass,
    Bar,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)
from trading.execution import Simulator, new_client_order_id


@pytest.fixture
def aapl() -> Instrument:
    return Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def t0() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def t1() -> datetime:
    return datetime(2024, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def t2() -> datetime:
    return datetime(2024, 1, 3, tzinfo=timezone.utc)


@pytest.fixture
def sim() -> Simulator:
    s = Simulator(initial_cash=100_000.0, slippage_bps=2.0, commission_bps=1.0)
    s.connect()
    return s


def make_order(
    instrument: Instrument,
    side: Side,
    quantity: float,
    *,
    created_at: datetime,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
) -> Order:
    return Order(
        client_order_id=new_client_order_id(),
        instrument=instrument,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        tif=TimeInForce.DAY,
        created_at=created_at,
    )


def make_bar(ts: datetime, *, open: float, close: float | None = None,
             high: float | None = None, low: float | None = None,
             volume: float = 1_000.0) -> Bar:
    return Bar(
        ts=ts,
        open=open,
        high=high if high is not None else max(open, close or open) + 0.5,
        low=low if low is not None else min(open, close or open) - 0.5,
        close=close if close is not None else open,
        volume=volume,
    )
