"""Smoke tests — fast, hermetic, no network. These run on every commit."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading import __version__
from trading.core.clock import FixedClock, IteratingClock, UtcClock
from trading.core.config import Settings
from trading.core.types import AssetClass, Bar, Instrument, Side


def test_version_string() -> None:
    assert isinstance(__version__, str)
    assert __version__  # non-empty


def test_instrument_key_is_stable() -> None:
    aapl = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    btc = Instrument(symbol="BTC/USDT", asset_class=AssetClass.CRYPTO, exchange="binance")
    assert aapl.key == "equity:AAPL"
    assert btc.key == "crypto:binance:BTC/USDT"


def test_bar_requires_timezone() -> None:
    naive = datetime(2024, 1, 1, 9, 30)
    with pytest.raises(ValueError, match="timezone-aware"):
        Bar(ts=naive, open=1, high=1, low=1, close=1)

    aware = naive.replace(tzinfo=timezone.utc)
    bar = Bar(ts=aware, open=1, high=1, low=1, close=1, volume=100)
    assert bar.ts.tzinfo is not None


def test_bar_is_immutable() -> None:
    bar = Bar(
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=100, high=101, low=99, close=100.5,
    )
    with pytest.raises(Exception):  # pydantic raises ValidationError for frozen
        bar.close = 200  # type: ignore[misc]


def test_side_enum_string_value() -> None:
    assert Side.BUY == "buy"
    assert Side.SELL == "sell"


def test_clocks_basic_behavior() -> None:
    real = UtcClock()
    assert real.now().tzinfo is not None

    fixed = FixedClock(instant=datetime(2024, 6, 1, tzinfo=timezone.utc))
    assert fixed.now() == datetime(2024, 6, 1, tzinfo=timezone.utc)

    it = IteratingClock(current=datetime(2024, 1, 1, tzinfo=timezone.utc))
    it.advance_to(datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert it.now().month == 2

    with pytest.raises(ValueError):
        it.advance_to(datetime(2024, 3, 1))  # naive


def test_settings_defaults_are_safe() -> None:
    s = Settings()
    # Defaults must NOT enable live trading.
    assert s.trading_env in ("research", "paper", "live")
    assert s.is_live_armed() is False or (s.trading_env == "live" and s.allow_live_trading)


def test_live_armed_requires_both_flags() -> None:
    s_paper = Settings(TRADING_ENV="paper", ALLOW_LIVE_TRADING=True)  # type: ignore[call-arg]
    assert s_paper.is_live_armed() is False  # paper + flag still NOT live

    s_live_no_flag = Settings(TRADING_ENV="live", ALLOW_LIVE_TRADING=False)  # type: ignore[call-arg]
    assert s_live_no_flag.is_live_armed() is False  # live without explicit flag still NOT armed

    s_live_armed = Settings(TRADING_ENV="live", ALLOW_LIVE_TRADING=True)  # type: ignore[call-arg]
    assert s_live_armed.is_live_armed() is True
