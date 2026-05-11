"""Strategy library.

Importing this package registers every built-in strategy in
``trading.strategies.base.STRATEGY_REGISTRY``. The CLI looks up strategies
by name there.

Built-in strategies::

    donchian          Channel breakout, classic Turtle.
    ema_cross         Fast/slow EMA crossover trend filter.
    xsec_momentum     12-1 cross-sectional momentum.
    rsi2              Larry Connors' RSI(2) mean reversion.
    zscore_meanrev    Rolling z-score mean reversion.
    risk_parity       Inverse-vol weighting.
    pairs             Cointegration-screened pairs trade via spread z-score.
"""

from __future__ import annotations

from trading.strategies.base import (
    STRATEGY_REGISTRY,
    Strategy,
    StrategyParams,
    available_strategies,
    get_strategy,
    register,
)

# Side-effect imports: each module registers its class on import.
from trading.strategies import donchian as _donchian       # noqa: F401
from trading.strategies import ema_cross as _ema_cross     # noqa: F401
from trading.strategies import xsec_momentum as _xsec      # noqa: F401
from trading.strategies import rsi2 as _rsi2               # noqa: F401
from trading.strategies import zscore_meanrev as _zscore   # noqa: F401
from trading.strategies import risk_parity as _rp          # noqa: F401
from trading.strategies import pairs as _pairs             # noqa: F401

from trading.strategies.donchian import Donchian, DonchianParams
from trading.strategies.ema_cross import EmaCross, EmaCrossParams
from trading.strategies.pairs import Pairs, PairsParams
from trading.strategies.risk_parity import RiskParity, RiskParityParams
from trading.strategies.rsi2 import Rsi2, Rsi2Params
from trading.strategies.xsec_momentum import XSecMomentum, XSecMomentumParams
from trading.strategies.zscore_meanrev import ZScoreMeanRev, ZScoreMeanRevParams

__all__ = [
    "STRATEGY_REGISTRY",
    "Strategy",
    "StrategyParams",
    "Donchian",
    "DonchianParams",
    "EmaCross",
    "EmaCrossParams",
    "Pairs",
    "PairsParams",
    "RiskParity",
    "RiskParityParams",
    "Rsi2",
    "Rsi2Params",
    "XSecMomentum",
    "XSecMomentumParams",
    "ZScoreMeanRev",
    "ZScoreMeanRevParams",
    "available_strategies",
    "get_strategy",
    "register",
]
