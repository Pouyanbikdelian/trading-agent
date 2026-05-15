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
    kalman_pairs      Dynamic-hedge-ratio pairs trade via Kalman filter.
    pca_residual      Statistical-arbitrage on rolling-PCA residuals
                      (Avellaneda-Lee 2010).
    top_k_momentum    Antonacci dual-momentum: rank by trailing return,
                      gate by absolute return, inverse-vol weight top K.
"""

from __future__ import annotations

# Side-effect imports: each module registers its class on import.
from trading.strategies import donchian as _donchian  # noqa: F401
from trading.strategies import ema_cross as _ema_cross  # noqa: F401
from trading.strategies import long_term_momentum as _long_term_momentum  # noqa: F401
from trading.strategies import pairs as _pairs  # noqa: F401
from trading.strategies import pairs_kalman as _pairs_kalman  # noqa: F401
from trading.strategies import pca_residual as _pca_residual  # noqa: F401
from trading.strategies import risk_parity as _rp  # noqa: F401
from trading.strategies import rsi2 as _rsi2  # noqa: F401
from trading.strategies import top_k_momentum as _top_k_momentum  # noqa: F401
from trading.strategies import xsec_momentum as _xsec  # noqa: F401
from trading.strategies import zscore_meanrev as _zscore  # noqa: F401
from trading.strategies.base import (
    STRATEGY_REGISTRY,
    Strategy,
    StrategyParams,
    available_strategies,
    get_strategy,
    register,
)
from trading.strategies.donchian import Donchian, DonchianParams
from trading.strategies.ema_cross import EmaCross, EmaCrossParams
from trading.strategies.long_term_momentum import (
    LongTermMomentum,
    LongTermMomentumParams,
)
from trading.strategies.pairs import Pairs, PairsParams
from trading.strategies.pairs_kalman import KalmanPairs, KalmanPairsParams
from trading.strategies.pca_residual import PCAResidual, PCAResidualParams
from trading.strategies.risk_parity import RiskParity, RiskParityParams
from trading.strategies.rsi2 import Rsi2, Rsi2Params
from trading.strategies.top_k_momentum import TopKMomentum, TopKMomentumParams
from trading.strategies.xsec_momentum import XSecMomentum, XSecMomentumParams
from trading.strategies.zscore_meanrev import ZScoreMeanRev, ZScoreMeanRevParams

__all__ = [
    "STRATEGY_REGISTRY",
    "Donchian",
    "DonchianParams",
    "EmaCross",
    "EmaCrossParams",
    "KalmanPairs",
    "KalmanPairsParams",
    "LongTermMomentum",
    "LongTermMomentumParams",
    "PCAResidual",
    "PCAResidualParams",
    "Pairs",
    "PairsParams",
    "RiskParity",
    "RiskParityParams",
    "Rsi2",
    "Rsi2Params",
    "Strategy",
    "StrategyParams",
    "TopKMomentum",
    "TopKMomentumParams",
    "XSecMomentum",
    "XSecMomentumParams",
    "ZScoreMeanRev",
    "ZScoreMeanRevParams",
    "available_strategies",
    "get_strategy",
    "register",
]
