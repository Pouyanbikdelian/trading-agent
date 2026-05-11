"""Regime classification: realized-vol buckets and Gaussian HMM.

The public surface is small — strategies don't construct classifiers; the
runner does, and pipes the regime series through ``Strategy.modulate``::

    from trading.regime import HmmRegime, RealizedVolRegime, regime_scale

    classifier = RealizedVolRegime(window=20, n_states=3).fit(train_returns)
    regime = classifier.predict(all_returns)
    scaled_weights = regime_scale(weights, regime, {0: 1.0, 1: 1.0, 2: 0.0})
"""

from __future__ import annotations

from trading.regime.base import RegimeClassifier, regime_scale
from trading.regime.hmm import HmmParams, HmmRegime
from trading.regime.realized_vol import RealizedVolParams, RealizedVolRegime

__all__ = [
    "HmmParams",
    "HmmRegime",
    "RealizedVolParams",
    "RealizedVolRegime",
    "RegimeClassifier",
    "regime_scale",
]
