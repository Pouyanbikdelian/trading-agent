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

    @classmethod
    def from_settings(cls, settings: Settings) -> "RiskLimits":
        """Default factory honoring values from ``.env``."""
        return cls(
            max_position_pct=settings.max_position_pct,
            max_gross_exposure=settings.max_gross_exposure,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
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

    def replace(self, **fields: Any) -> "HaltState":
        return self.model_copy(update=fields)
