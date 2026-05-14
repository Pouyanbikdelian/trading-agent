"""Tests for the universe screens — pure functions, hand-built inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument
from trading.selection import (
    Fundamentals,
    ScreenConfig,
    apply_screens,
    liquidity_screen,
    quality_screen,
    sector_momentum_screen,
)


@pytest.fixture
def universe() -> list[Instrument]:
    return [
        Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY),
        Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY),
        Instrument(symbol="JNK", asset_class=AssetClass.EQUITY),  # junk
        Instrument(symbol="OIL_CO", asset_class=AssetClass.EQUITY),
    ]


@pytest.fixture
def closes_volumes() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    closes = pd.DataFrame(
        {
            "AAPL": np.linspace(180.0, 200.0, 60),
            "MSFT": np.linspace(380.0, 400.0, 60),
            "JNK": np.linspace(2.0, 2.5, 60),
            "OIL_CO": np.linspace(50.0, 55.0, 60),
        },
        index=idx,
    )
    volumes = pd.DataFrame(
        {
            "AAPL": np.full(60, 50_000_000.0),  # huge ADV ~ $10B
            "MSFT": np.full(60, 25_000_000.0),
            "JNK": np.full(60, 5_000.0),  # ADV ~ $11k
            "OIL_CO": np.full(60, 1_000_000.0),
        },
        index=idx,
    )
    return closes, volumes


# -------------------------------------------------------------- liquidity ----


def test_liquidity_drops_low_volume_names(universe, closes_volumes) -> None:
    closes, volumes = closes_volumes
    kept = liquidity_screen(
        universe,
        closes=closes,
        volumes=volumes,
        min_dollar_volume=1_000_000_000.0,  # $1B/day floor
        lookback=30,
    )
    syms = [i.symbol for i in kept]
    # AAPL and MSFT clear $1B easily; JNK and OIL_CO don't.
    assert "AAPL" in syms
    assert "MSFT" in syms
    assert "JNK" not in syms
    assert "OIL_CO" not in syms


def test_liquidity_drops_missing_symbol(universe, closes_volumes) -> None:
    closes, volumes = closes_volumes
    universe_plus = [*universe, Instrument(symbol="UNCACHED", asset_class=AssetClass.EQUITY)]
    kept = liquidity_screen(
        universe_plus,
        closes=closes,
        volumes=volumes,
        min_dollar_volume=0.0,
        lookback=30,
    )
    # UNCACHED isn't in closes/volumes → dropped, even with min=0.
    assert "UNCACHED" not in [i.symbol for i in kept]


# ----------------------------------------------------------------- quality ----


def test_quality_requires_min_roe(universe) -> None:
    funds = {
        "AAPL": Fundamentals(
            symbol="AAPL",
            sector="Technology",
            roe=0.40,
            debt_to_equity=1.5,
            free_cash_flow_positive=True,
        ),
        "MSFT": Fundamentals(
            symbol="MSFT",
            sector="Technology",
            roe=0.30,
            debt_to_equity=0.5,
            free_cash_flow_positive=True,
        ),
        "JNK": Fundamentals(
            symbol="JNK",
            sector="Industrials",
            roe=-0.05,
            debt_to_equity=4.0,
            free_cash_flow_positive=False,
        ),
        "OIL_CO": Fundamentals(
            symbol="OIL_CO",
            sector="Energy",
            roe=0.12,
            debt_to_equity=0.8,
            free_cash_flow_positive=True,
        ),
    }
    kept = quality_screen(universe, funds, min_roe=0.15)
    syms = {i.symbol for i in kept}
    assert syms == {"AAPL", "MSFT"}


def test_quality_drops_missing_fundamentals(universe) -> None:
    """A name without fundamentals must be dropped, not silently allowed."""
    funds = {"AAPL": Fundamentals(symbol="AAPL", roe=0.40)}
    kept = quality_screen(universe, funds, min_roe=0.10)
    assert [i.symbol for i in kept] == ["AAPL"]


def test_quality_max_debt_to_equity(universe) -> None:
    funds = {
        "AAPL": Fundamentals(symbol="AAPL", debt_to_equity=0.5),
        "MSFT": Fundamentals(symbol="MSFT", debt_to_equity=2.5),
        "JNK": Fundamentals(symbol="JNK", debt_to_equity=10.0),
        "OIL_CO": Fundamentals(symbol="OIL_CO", debt_to_equity=1.0),
    }
    kept = quality_screen(universe, funds, max_debt_to_equity=1.5)
    assert {i.symbol for i in kept} == {"AAPL", "OIL_CO"}


def test_quality_require_positive_fcf(universe) -> None:
    funds = {
        "AAPL": Fundamentals(symbol="AAPL", free_cash_flow_positive=True),
        "MSFT": Fundamentals(symbol="MSFT", free_cash_flow_positive=False),
        "JNK": Fundamentals(symbol="JNK", free_cash_flow_positive=None),
        "OIL_CO": Fundamentals(symbol="OIL_CO", free_cash_flow_positive=True),
    }
    kept = quality_screen(universe, funds, require_positive_fcf=True)
    assert {i.symbol for i in kept} == {"AAPL", "OIL_CO"}


# -------------------------------------------------- sector_momentum_screen ----


def test_sector_momentum_keeps_top_sectors(universe) -> None:
    idx = pd.date_range("2024-01-01", periods=63, freq="1D", tz="UTC")
    sectors = pd.DataFrame(
        {
            "Technology": np.linspace(100.0, 130.0, 63),  # +30% return
            "Energy": np.linspace(100.0, 110.0, 63),  # +10%
            "Industrials": np.linspace(100.0, 90.0, 63),  # -10%
        },
        index=idx,
    )
    funds = {
        "AAPL": Fundamentals(symbol="AAPL", sector="Technology"),
        "MSFT": Fundamentals(symbol="MSFT", sector="Technology"),
        "JNK": Fundamentals(symbol="JNK", sector="Industrials"),
        "OIL_CO": Fundamentals(symbol="OIL_CO", sector="Energy"),
    }
    kept = sector_momentum_screen(
        universe,
        funds,
        sector_prices=sectors,
        top_n_sectors=2,
        lookback=63,
    )
    # Top 2 sectors are Tech (+30%) and Energy (+10%); Industrials drops out.
    assert {i.symbol for i in kept} == {"AAPL", "MSFT", "OIL_CO"}
    assert "JNK" not in {i.symbol for i in kept}


def test_sector_momentum_empty_prices_pass_through(universe) -> None:
    funds = {i.symbol: Fundamentals(symbol=i.symbol, sector="Technology") for i in universe}
    kept = sector_momentum_screen(
        universe,
        funds,
        sector_prices=pd.DataFrame(),
        top_n_sectors=2,
    )
    assert len(kept) == len(universe)


# ------------------------------------------------------------ apply_screens ----


def test_apply_screens_chains_filters(universe, closes_volumes) -> None:
    closes, volumes = closes_volumes
    funds = {
        "AAPL": Fundamentals(
            symbol="AAPL",
            sector="Technology",
            roe=0.40,
            debt_to_equity=1.5,
            free_cash_flow_positive=True,
        ),
        "MSFT": Fundamentals(
            symbol="MSFT",
            sector="Technology",
            roe=0.30,
            debt_to_equity=0.5,
            free_cash_flow_positive=True,
        ),
        "OIL_CO": Fundamentals(
            symbol="OIL_CO",
            sector="Energy",
            roe=0.12,
            debt_to_equity=0.8,
            free_cash_flow_positive=True,
        ),
    }
    idx = pd.date_range("2024-01-01", periods=63, freq="1D", tz="UTC")
    sectors = pd.DataFrame(
        {
            "Technology": np.linspace(100.0, 130.0, 63),
            "Energy": np.linspace(100.0, 95.0, 63),
        },
        index=idx,
    )
    cfg = ScreenConfig(
        min_dollar_volume=1_000_000_000.0,
        min_roe=0.20,
        top_n_sectors=1,
        sector_momentum_lookback=63,
    )
    kept = apply_screens(
        universe,
        cfg,
        closes=closes,
        volumes=volumes,
        fundamentals=funds,
        sector_prices=sectors,
    )
    # AAPL survives all 3: liquid, ROE 40%, in top-1 sector.
    # MSFT survives liquidity + quality, in top sector.
    # JNK fails liquidity. OIL_CO fails ROE.
    assert {i.symbol for i in kept} == {"AAPL", "MSFT"}


def test_apply_screens_no_op_when_all_disabled(universe) -> None:
    cfg = ScreenConfig()  # everything None / False
    out = apply_screens(universe, cfg)
    assert out == universe


def test_apply_screens_missing_data_disables_screen(universe, closes_volumes) -> None:
    """If you ask for a quality screen but provide no fundamentals, the
    screen quietly skips. Same for sector momentum."""
    closes, volumes = closes_volumes
    cfg = ScreenConfig(
        min_dollar_volume=1_000_000_000.0,
        min_roe=0.5,
        top_n_sectors=1,
    )
    kept = apply_screens(
        universe,
        cfg,
        closes=closes,
        volumes=volumes,
        fundamentals=None,
        sector_prices=None,
    )
    # Quality + sector momentum were configured but had no data → only
    # liquidity actually fired. AAPL + MSFT survive on liquidity alone.
    assert {i.symbol for i in kept} == {"AAPL", "MSFT"}
