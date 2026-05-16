"""Strategy selection and portfolio combination.

Public surface::

    from trading.selection import (
        rank_strategies,                # DSR-ranked leaderboard
        equal_weight, inverse_vol, min_variance,
        vol_target,
    )
"""

from __future__ import annotations

from trading.selection.combine import (
    dsr_weighted,
    equal_weight,
    inverse_vol,
    min_variance,
    sharpe_weighted,
)
from trading.selection.dip_buy_overlay import dip_buy
from trading.selection.ewma_vol_target import ewma_vol_target
from trading.selection.hedge_overlay import beta_hedge
from trading.selection.mode_overlay import (
    DEFAULT_DEFENSIVE_SLEEVE,
    ModePolicy,
    apply_mode,
    estimate_mode_impact,
)
from trading.selection.overlay import vol_target
from trading.selection.rank import rank_strategies
from trading.selection.regime_derisk import regime_derisk
from trading.selection.scores import (
    annualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    per_period_sharpe,
    probabilistic_sharpe,
)
from trading.selection.screens import (
    Fundamentals,
    ScreenConfig,
    apply_screens,
    liquidity_screen,
    quality_screen,
    sector_momentum_screen,
)

__all__ = [
    "DEFAULT_DEFENSIVE_SLEEVE",
    "Fundamentals",
    "ModePolicy",
    "ScreenConfig",
    "annualize_sharpe",
    "apply_mode",
    "apply_screens",
    "beta_hedge",
    "deflated_sharpe",
    "dip_buy",
    "dsr_weighted",
    "equal_weight",
    "estimate_mode_impact",
    "ewma_vol_target",
    "expected_max_sharpe",
    "inverse_vol",
    "liquidity_screen",
    "min_variance",
    "moments",
    "per_period_sharpe",
    "probabilistic_sharpe",
    "quality_screen",
    "rank_strategies",
    "regime_derisk",
    "sector_momentum_screen",
    "sharpe_weighted",
    "vol_target",
]
