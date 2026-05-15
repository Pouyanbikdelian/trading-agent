r"""Core-satellite portfolio framework.

Splits the book into two sleeves:

*   The **core** is a fixed list of long-term strategic holdings the
    algorithm never touches except to drift them back to their target
    weights on a slow cadence. Use it for high-conviction themes you
    want to hold for years (nuclear, defense, utilities, a personal
    favourite).
*   The **satellite** is whatever the active strategy (top-K momentum,
    combiner, etc.) wants to do, *but only within the satellite share
    of total equity*.

Configuration
-------------
Define core holdings in ``config/portfolio.yaml``::

    core_allocation: 0.50              # 50% of equity in the core sleeve
    core_rebalance_days: 63            # quarterly drift correction
    core_holdings:
      nuclear:
        weight: 0.20                    # 20% of core (= 10% of total equity)
        symbols: [URA, CCJ, NLR]
      utilities:
        weight: 0.20
        symbols: [XLU, NEE, DUK]
      defense:
        weight: 0.10
        symbols: [ITA, LMT, RTX]
      cash:
        weight: 0.50                    # remaining 50% of core in cash

    # ``cash`` is a special key — it sits as a zero-weight allocation,
    # i.e. unused capital stays in the broker's cash balance.

Within each theme the symbols are equal-weighted. Theme weights must
sum to <= 1.0; any unused fraction stays in cash.

Composition with the algorithm
------------------------------
The runner calls :func:`apply_core_satellite` once per cycle::

    core_weights = build_core_weights(spec)
    final = apply_core_satellite(satellite_weights, core_weights, spec)

``final`` has the algorithm's signals scaled down to occupy only the
satellite fraction of equity, plus the static core sleeve laid on top.
Gross exposure remains <= 1.0 by construction (provided the satellite
weights respect the satellite fraction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field


class CoreTheme(BaseModel):
    """One named bundle of long-term holdings (e.g. 'nuclear')."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weight: float = Field(ge=0.0, le=1.0)
    """Weight as a fraction of the *core* sleeve (not total equity)."""

    symbols: list[str] = Field(default_factory=list)
    """Equal-weighted within the theme. May be empty for the special 'cash' theme."""


class CoreSpec(BaseModel):
    """Whole core-satellite specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    core_allocation: float = Field(default=0.50, ge=0.0, le=1.0)
    """Fraction of total equity dedicated to the core sleeve. Satellite
    gets ``1 - core_allocation``."""

    core_rebalance_days: int = Field(default=63, ge=1)
    """Bars between core-sleeve drift corrections. 63 ≈ quarterly."""

    core_holdings: dict[str, CoreTheme] = Field(default_factory=dict)

    def validate_theme_weights(self) -> None:
        """Refuse if theme weights sum to > 1.0."""
        total = sum(t.weight for t in self.core_holdings.values())
        if total > 1.0001:
            raise ValueError(f"core theme weights sum to {total:.3f} > 1.0; would imply leverage")


def load_core_spec(path: str | Path) -> CoreSpec:
    """Read a ``portfolio.yaml`` into a validated :class:`CoreSpec`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"core spec not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    spec = CoreSpec.model_validate(raw)
    spec.validate_theme_weights()
    return spec


def build_core_weights(spec: CoreSpec, columns: list[str]) -> pd.Series:
    r"""Compute the *static* per-symbol weights of the core sleeve,
    expressed as fractions of TOTAL equity.

    Returns a :class:`pd.Series` indexed by ``columns`` (the universe
    the satellite trades over). Symbols not in any theme get 0; symbols
    in a theme get
    :math:`\frac{\text{core\_allocation} \cdot \text{theme.weight}}
    {|\text{theme.symbols}|}`.
    """
    out = pd.Series(0.0, index=columns, dtype=float)
    for theme in spec.core_holdings.values():
        if not theme.symbols or theme.weight <= 0:
            continue
        per_name = spec.core_allocation * theme.weight / len(theme.symbols)
        for sym in theme.symbols:
            if sym in out.index:
                out.loc[sym] = out.loc[sym] + per_name
    return out


def apply_core_satellite(
    satellite_weights: pd.DataFrame,
    core_weights: pd.Series,
    spec: CoreSpec,
) -> pd.DataFrame:
    r"""Compose the static core sleeve with the algorithm's satellite
    weights, holding total gross exposure at most 1.0.

    The satellite signals are scaled down to occupy the ``1 -
    core_allocation`` slice of equity, then summed with the static core
    weights repeated across every row. If a satellite name overlaps a
    core name (you trade a satellite position in a stock that's also in
    the core), the weights *add* — the satellite is sized on top of the
    core position. Per-position caps remain the risk manager's job.
    """
    if not 0.0 <= spec.core_allocation <= 1.0:
        raise ValueError("core_allocation must be in [0, 1]")
    satellite_share = 1.0 - spec.core_allocation

    # Align satellite to the same columns; missing core names are zero
    # in the satellite, missing satellite names are zero in the core.
    all_cols = sorted(set(satellite_weights.columns).union(core_weights.index))
    sat = satellite_weights.reindex(columns=all_cols, fill_value=0.0)
    core = core_weights.reindex(index=all_cols, fill_value=0.0)

    # Scale satellite to its allocation. The satellite weights are
    # already at gross <= 1; multiplying by satellite_share lands them
    # at gross <= satellite_share.
    scaled_sat = sat * satellite_share

    # Broadcast core across rows.
    core_frame = pd.DataFrame(
        np.tile(core.values, (len(sat), 1)),
        index=sat.index,
        columns=all_cols,
    )

    return scaled_sat + core_frame
