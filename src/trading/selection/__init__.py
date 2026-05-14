"""Strategy selection and portfolio combination.

Public surface::

    from trading.selection import (
        rank_strategies,                # DSR-ranked leaderboard
        equal_weight, inverse_vol, min_variance,
        vol_target,
    )
"""

from __future__ import annotations

from trading.selection.combine import dsr_weighted, equal_weight, inverse_vol, min_variance
from trading.selection.overlay import vol_target
from trading.selection.rank import rank_strategies
from trading.selection.scores import (
    annualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    per_period_sharpe,
    probabilistic_sharpe,
)

__all__ = [
    "annualize_sharpe",
    "deflated_sharpe",
    "dsr_weighted",
    "equal_weight",
    "expected_max_sharpe",
    "inverse_vol",
    "min_variance",
    "moments",
    "per_period_sharpe",
    "probabilistic_sharpe",
    "rank_strategies",
    "vol_target",
]
