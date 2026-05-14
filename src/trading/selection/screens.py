"""Universe pre-filters — strip the obviously-bad names *before* the
strategy layer ever sees them.

A strategy ranked over 500 S&P names spends most of its degrees of freedom
on companies that are too illiquid to trade, fundamentally broken, or
sitting in a structurally-declining sector. Pre-filtering tightens the
candidate set first; the strategy then picks among names that already
clear the basic-hygiene bar.

Three screens here:

1. ``liquidity_screen``     — drop names below an average dollar-volume floor.
2. ``quality_screen``       — drop names failing fundamental thresholds
                              (ROE, debt-to-equity, free-cash flow positive).
3. ``sector_momentum_screen`` — drop names whose sector ETF is in the
                                bottom half of recent sector returns.

All three are pure functions: ``list[Instrument] + criteria -> list[Instrument]``.
They don't talk to the network — they consume frames the caller pre-fetched.
This makes them trivial to test and lets the runner cache fundamentals and
sector returns on whatever cadence makes sense (weekly is plenty).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading.core.types import Instrument


class Fundamentals(BaseModel):
    """The fundamental fields the quality screen reads.

    Frozen so a dict[symbol, Fundamentals] can be passed around without
    aliasing. ``sector`` is also used by ``sector_momentum_screen``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    symbol: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    roe: float | None = None
    """Return on equity (TTM). 0.15 = 15%."""
    debt_to_equity: float | None = None
    """Debt / equity. 1.0 = 1:1; > 2 = highly levered."""
    profit_margin: float | None = None
    free_cash_flow_positive: bool | None = None


class ScreenConfig(BaseModel):
    """Knobs the runner / CLI use to dial each screen in or out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # liquidity
    min_dollar_volume: float | None = Field(default=None, ge=0.0)
    """Average daily dollar-volume floor over ``liquidity_lookback``.
    None disables the screen."""
    liquidity_lookback: int = Field(default=30, ge=2)

    # quality
    min_roe: float | None = None
    """0.10 = 10% ROE floor. None disables."""
    max_debt_to_equity: float | None = Field(default=None, ge=0.0)
    """2.0 = max 2:1 debt/equity. None disables."""
    require_positive_fcf: bool = False

    # sector momentum
    top_n_sectors: int | None = Field(default=None, ge=1)
    """If set, drop names whose sector is outside the top N by recent return.
    None disables."""
    sector_momentum_lookback: int = Field(default=63, ge=5)


# -------------------------------------------------------------- liquidity ----


def liquidity_screen(
    instruments: Iterable[Instrument],
    *,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    min_dollar_volume: float,
    lookback: int = 30,
) -> list[Instrument]:
    """Keep instruments whose average daily dollar-volume over the last
    ``lookback`` bars meets ``min_dollar_volume``. Symbols missing from
    ``closes``/``volumes`` are dropped (can't price them = can't trade them).
    """
    out: list[Instrument] = []
    for ins in instruments:
        sym = ins.symbol
        if sym not in closes.columns or sym not in volumes.columns:
            continue
        c = closes[sym].iloc[-lookback:]
        v = volumes[sym].iloc[-lookback:]
        if c.empty or v.empty:
            continue
        adv = float((c * v).dropna().mean())
        if adv >= min_dollar_volume:
            out.append(ins)
    return out


# ---------------------------------------------------------------- quality ----


def quality_screen(
    instruments: Iterable[Instrument],
    fundamentals: dict[str, Fundamentals],
    *,
    min_roe: float | None = None,
    max_debt_to_equity: float | None = None,
    require_positive_fcf: bool = False,
) -> list[Instrument]:
    """Filter by fundamental thresholds. A name with missing fundamentals
    is **dropped** — we can't verify it, so we don't trust it."""
    out: list[Instrument] = []
    for ins in instruments:
        f = fundamentals.get(ins.symbol)
        if f is None:
            continue
        if min_roe is not None and (f.roe is None or f.roe < min_roe):
            continue
        if max_debt_to_equity is not None and (
            f.debt_to_equity is None or f.debt_to_equity > max_debt_to_equity
        ):
            continue
        if require_positive_fcf and f.free_cash_flow_positive is not True:
            continue
        out.append(ins)
    return out


# ------------------------------------------------------- sector momentum ----


def sector_momentum_screen(
    instruments: Iterable[Instrument],
    fundamentals: dict[str, Fundamentals],
    *,
    sector_prices: pd.DataFrame,
    top_n_sectors: int,
    lookback: int = 63,
) -> list[Instrument]:
    """Keep only names whose sector is in the top N by recent return.

    ``sector_prices`` is a wide-format DataFrame whose columns are sector
    *names* (matching ``Fundamentals.sector``), values are sector-level
    prices (e.g. sector ETF closes). The screen ranks sectors by their
    ``lookback``-bar return and keeps only members of the top N.
    """
    if sector_prices.empty:
        return list(instruments)

    # Sector return = pct change over the lookback window.
    window = sector_prices.iloc[-lookback:]
    if len(window) < 2:
        return list(instruments)
    returns = (window.iloc[-1] / window.iloc[0]) - 1.0
    ranked = returns.sort_values(ascending=False)
    keep_sectors = set(ranked.head(top_n_sectors).index)

    out: list[Instrument] = []
    for ins in instruments:
        f = fundamentals.get(ins.symbol)
        if f is None or f.sector is None:
            continue
        if f.sector in keep_sectors:
            out.append(ins)
    return out


# --------------------------------------------------------- composite apply ----


def apply_screens(
    instruments: Iterable[Instrument],
    cfg: ScreenConfig,
    *,
    closes: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    fundamentals: dict[str, Fundamentals] | None = None,
    sector_prices: pd.DataFrame | None = None,
) -> list[Instrument]:
    """Apply every enabled screen in ``cfg`` in order. Each screen sees the
    output of the previous one, so a name needs to pass all of them to
    survive. Caller is responsible for providing the data each screen needs;
    missing data quietly disables the screen rather than raising."""
    out = list(instruments)

    if cfg.min_dollar_volume is not None and closes is not None and volumes is not None:
        out = liquidity_screen(
            out,
            closes=closes,
            volumes=volumes,
            min_dollar_volume=cfg.min_dollar_volume,
            lookback=cfg.liquidity_lookback,
        )

    if (
        any(x is not None for x in (cfg.min_roe, cfg.max_debt_to_equity))
        or cfg.require_positive_fcf
    ) and fundamentals is not None:
        out = quality_screen(
            out,
            fundamentals,
            min_roe=cfg.min_roe,
            max_debt_to_equity=cfg.max_debt_to_equity,
            require_positive_fcf=cfg.require_positive_fcf,
        )

    if cfg.top_n_sectors is not None and fundamentals is not None and sector_prices is not None:
        out = sector_momentum_screen(
            out,
            fundamentals,
            sector_prices=sector_prices,
            top_n_sectors=cfg.top_n_sectors,
            lookback=cfg.sector_momentum_lookback,
        )

    return out
