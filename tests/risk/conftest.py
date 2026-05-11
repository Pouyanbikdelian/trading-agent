"""Shared fixtures for risk-manager tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Position,
    Signal,
)
from trading.risk import RiskLimits, RiskManager


@pytest.fixture
def t0() -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def t1() -> datetime:
    return datetime(2024, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def aapl() -> Instrument:
    return Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def msft() -> Instrument:
    return Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY)


@pytest.fixture
def xom() -> Instrument:
    return Instrument(symbol="XOM", asset_class=AssetClass.EQUITY)


@pytest.fixture
def limits() -> RiskLimits:
    return RiskLimits(
        max_position_pct=0.10,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_sector_exposure=0.30,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.15,
    )


@pytest.fixture
def mgr(limits: RiskLimits, tmp_path) -> RiskManager:
    """Each test gets a fresh manager with its own halt-state file."""
    return RiskManager(limits, halt_state_path=tmp_path / "halt.json")


@pytest.fixture
def account_100k(t0: datetime) -> AccountSnapshot:
    return AccountSnapshot(ts=t0, cash=100_000.0, equity=100_000.0)


def signal_from(
    ts: datetime,
    weights: dict[str, float],
    *,
    strategy: str = "test",
) -> Signal:
    return Signal(ts=ts, strategy=strategy, target_weights=weights)


def instruments_dict(*instruments: Instrument) -> dict[str, Instrument]:
    return {i.key: i for i in instruments}
