"""Regime classifier interface.

A regime classifier maps a returns series to a per-bar integer state label
(``0``, ``1``, ``2``, …). The labels themselves carry no semantics — they
are just buckets. Concrete classifiers stamp a *meaning* on the labels by
ordering them consistently (e.g. realized-vol: 0 = low vol, 2 = high vol).

Fit / predict separation
------------------------
Like sklearn estimators, classifiers split ``fit(train_returns)`` from
``predict(returns)``. ``fit`` learns parameters (quantile edges, HMM
transition + emission matrices, …) from in-sample data; ``predict`` applies
those parameters to a possibly-different out-of-sample series. Strategies
that need an online regime should call ``fit`` on a warm-up window and then
``predict`` on the full history — this is what the walk-forward harness
does for everything else, kept identical here so the patterns line up.

Why a Protocol, not an ABC
--------------------------
Same reasoning as ``DataSource``: third parties (and tests) can implement
the contract without inheriting. The two built-in classifiers do inherit a
small shared base for convenience but it isn't required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class RegimeClassifier(Protocol):
    """Anything that learns N discrete regimes from returns."""

    n_states: int

    def fit(self, returns: pd.Series) -> "RegimeClassifier": ...

    def predict(self, returns: pd.Series) -> pd.Series:
        """Return an integer ``Series`` aligned to ``returns.index`` with
        values in ``[0, n_states)``. Bars too early for a stable inference
        (e.g. inside the rolling-vol warm-up) get ``-1``."""
        ...


def regime_scale(
    weights: pd.DataFrame,
    regime: pd.Series,
    scale_by_state: dict[int, float],
    *,
    unknown_scale: float = 1.0,
) -> pd.DataFrame:
    """Scale a weight frame row-by-row using a ``regime -> multiplier`` map.

    Common patterns this captures cleanly::

        # Risk-off in high-vol regime, full size everywhere else.
        scaled = regime_scale(w, regime, {0: 1.0, 1: 1.0, 2: 0.0})

        # Half size while we don't know the regime yet (warm-up).
        scaled = regime_scale(w, regime, {0: 1.0, 1: 1.0}, unknown_scale=0.5)

    Bars with a state not in ``scale_by_state`` (including the warm-up
    sentinel ``-1``) get ``unknown_scale``.
    """
    aligned_regime = regime.reindex(weights.index).fillna(-1).astype(int)
    multiplier = aligned_regime.map(scale_by_state).fillna(unknown_scale).astype(float)
    return weights.mul(multiplier, axis=0)


def _ensure_returns(series: pd.Series) -> pd.Series:
    """Cast / sanity-check a returns series.

    Returns must be finite and indexed; we drop the first NaN row that a
    naive ``prices.pct_change()`` always produces but leave any others to
    surface as errors rather than silently masking them.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("returns must be a pandas Series")
    s = series.astype(float)
    # Drop a leading NaN, the most common pct_change artifact.
    if len(s) > 0 and pd.isna(s.iloc[0]):
        s = s.iloc[1:]
    if not np.isfinite(s.values).all():
        raise ValueError("returns contains NaN/inf after the leading bar")
    return s
