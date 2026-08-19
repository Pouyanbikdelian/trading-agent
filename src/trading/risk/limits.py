"""Risk-manager configuration + halt-state DTOs.

``RiskLimits`` is a frozen pydantic model with the per-instrument and
portfolio-level caps the manager enforces. ``HaltState`` is the manager's
small persistent state — it tracks the daily-open equity, the high-water
mark, and whether we're currently halted.

Halt state lives in a JSON file under ``settings.state_dir`` so it
survives process restarts. That's deliberate: a crash mid-day must not
forget that we hit the kill switch.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trading.core.config import Settings


class RiskLimits(BaseModel):
    """Caps applied by the risk manager. Values are fractions of equity
    (or fractions of the daily/peak equity for kill switches)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position_pct: float = Field(default=0.10, gt=0.0, le=1.0)
    """Per-instrument cap. |position_value| <= max_position_pct * equity."""

    allow_short: bool = Field(default=False)
    """Invariant: when False (default — this is a long-only system), no
    generated order may take a position below zero, PERIOD. Negative
    target weights are clamped to 0 and sell quantities are clamped to
    the effective current position (holdings net of working orders).
    Added 2026-07-15 after stacked after-hours orders shorted two names
    on paper: fixing the stacking addressed the cause; this enforces the
    invariant regardless of upstream confusion. Strategies that
    genuinely short must opt in explicitly."""

    max_gross_exposure: float = Field(default=1.0, gt=0.0)
    """Sum of |weights| across positions. >1.0 means leverage."""

    max_net_exposure: float = Field(default=1.0, gt=0.0)
    """Sum of signed weights. Net long if positive, net short if negative."""

    max_sector_exposure: float = Field(default=0.30, gt=0.0, le=1.0)
    """Cap on gross exposure within a single sector (sector map is per-call)."""

    max_daily_loss_pct: float = Field(default=0.02, gt=0.0, le=1.0)
    """Halt when day's PnL <= -max_daily_loss_pct * day_open_equity."""

    max_drawdown_pct: float = Field(default=0.15, gt=0.0, le=1.0)
    """Halt when equity <= (1 - max_drawdown_pct) * equity_high_watermark."""

    max_margin_borrowing_pct: float = Field(default=0.0, ge=0.0, le=10.0)
    """Hard cap on per-currency cash going negative. 0.0 = cash-account
    behavior (no margin); orders that would push any currency cash below
    -max_margin_borrowing_pct * equity are rejected pre-submit. CHF-base
    accounts buying USD stocks need either a pre-trade FX or this limit
    > 0 — otherwise IBKR's auto-loan kicks in and we're on margin."""

    baseline_sanity_divergence_pct: float = Field(default=0.5, gt=0.0, le=10.0)
    """How far today's equity may sit from the stored daily-open baseline
    before the baseline itself is judged to be the broken thing.

    The kill switches only mean anything if the number they compare
    against describes the same account. On 2026-08-07 it did not: a paper
    cycle had stamped CHF 1,068,862 into a live state directory, and the
    live session — 87,413 of equity — halted instantly at "daily loss
    -91.82%". A real intraday move of this size does not happen; a
    baseline from the wrong account, a currency mix-up, or a large
    deposit or withdrawal all do. Every one of those calls for
    re-stamping the baseline and telling the operator, not for halting.

    Set generously (50%) so it can only ever catch the absurd. Below it
    the kill switches behave exactly as before."""

    @classmethod
    def from_settings(cls, settings: Settings) -> RiskLimits:
        """Default factory honoring values from ``.env``."""
        return cls(
            max_position_pct=settings.max_position_pct,
            max_gross_exposure=settings.max_gross_exposure,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            max_margin_borrowing_pct=settings.max_margin_borrowing_pct,
            baseline_sanity_divergence_pct=settings.baseline_sanity_divergence_pct,
        )


class HaltState(BaseModel):
    """The manager's persistent state — daily-PnL tracking + halt flag.

    Frozen on purpose: we mutate by replacing the whole record via
    ``model_copy(update=...)`` and persisting the new value. That makes the
    write-then-read-back-in-tests pattern trivial and avoids partially-
    persisted state if a save mid-update crashes.
    """

    model_config = ConfigDict(frozen=True)

    halted: bool = False
    reason: str = ""
    halted_at: datetime | None = None
    equity_high_watermark: float = 0.0
    daily_equity_open: float = 0.0
    last_day: date | None = None
    # A daily-loss baseline is execution-safe only when it was captured at
    # the opening of the actual NYSE session.  ``last_day`` remains for
    # backwards-compatible historical state, while these fields let the live
    # monitor reject a stale/mid-session/foreign-currency baseline rather
    # than silently treating it as today's open.
    daily_baseline_session: date | None = None
    daily_baseline_captured_at: datetime | None = None
    daily_baseline_source: str | None = None
    daily_baseline_currency: str | None = None

    def replace(self, **fields: Any) -> HaltState:
        return self.model_copy(update=fields)
