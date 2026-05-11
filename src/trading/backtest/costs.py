"""Cost model for the vectorized backtester.

A deliberately simple model: a single bps figure for commission, a single bps
figure for slippage. Both are applied to per-bar turnover (sum of absolute
weight changes). Real-world broker costs are messier — per-share fees, tiered
discounts, exchange rebates, minimum tickets — but those depend on the venue
and only matter at sized live positions. Phase 6 (IBKR execution) will record
realized fills and reconcile against this approximation.

Defaults are conservative for IBKR retail equities:
* Commission: 1 bps (~0.5 bp tiered + buffer).
* Slippage:   2 bps round-trip on liquid US equities at end-of-day bars.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CostModel(BaseModel):
    """Per-trade cost in basis points. Frozen so it can be hashed / shared."""

    model_config = ConfigDict(frozen=True)

    commission_bps: float = Field(default=1.0, ge=0.0)
    slippage_bps: float = Field(default=2.0, ge=0.0)

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    @property
    def fractional(self) -> float:
        """Total cost as a fraction (e.g. 3 bps -> 0.0003)."""
        return self.total_bps / 1e4


ZERO_COSTS = CostModel(commission_bps=0.0, slippage_bps=0.0)
"""Sentinel for tests and pure-signal analyses."""
