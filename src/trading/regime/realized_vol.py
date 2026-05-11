"""Realized-volatility regime classifier.

Compute the rolling realized volatility of a returns series, learn quantile
edges from the in-sample window, and assign each out-of-sample bar to a
volatility bucket (0 = low, 1 = mid, 2 = high for the default 3-state).

Why quantiles and not absolute thresholds
-----------------------------------------
Vol levels are wildly different across markets (an FX major's 0.5% daily
vol vs. crypto's 4%+). Quantiles make the classifier portable. The trade-off
is that "high vol" in a calm regime is a much lower number than "high vol"
in a stressed one — but that's the right behavior for relative-vol gating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading.regime.base import RegimeClassifier, _ensure_returns


class RealizedVolParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    window: int = Field(default=20, ge=2)
    n_states: int = Field(default=3, ge=2, le=10)


class RealizedVolRegime:
    """Quantile-bucketed rolling vol classifier."""

    def __init__(self, params: RealizedVolParams | None = None, **kwargs: object) -> None:
        if params is None:
            params = RealizedVolParams(**kwargs)  # type: ignore[arg-type]
        self.params = params
        self.n_states = params.n_states
        self._edges: np.ndarray | None = None

    def fit(self, returns: pd.Series) -> "RealizedVolRegime":
        r = _ensure_returns(returns)
        rv = r.rolling(self.params.window, min_periods=self.params.window).std(ddof=1)
        rv = rv.dropna()
        # Heuristic: at least 4 observations per state. Estimating quantile
        # edges from one or two values per bucket is meaningless and produces
        # near-degenerate edges that misclassify everything OOS.
        min_obs = self.params.n_states * 4
        if len(rv) < min_obs:
            raise ValueError(
                f"need at least {min_obs} realized-vol observations to fit "
                f"({self.params.n_states} states x 4); got {len(rv)}"
            )
        # Even-quantile edges. ``np.quantile`` is robust to sample size.
        qs = np.linspace(0.0, 1.0, self.params.n_states + 1)[1:-1]
        self._edges = np.quantile(rv.values, qs)
        return self

    def predict(self, returns: pd.Series) -> pd.Series:
        if self._edges is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        r = _ensure_returns(returns)
        rv = r.rolling(self.params.window, min_periods=self.params.window).std(ddof=1)
        labels = np.full(len(rv), -1, dtype=int)
        mask = rv.notna().values
        # digitize is half-open on the right — values equal to an edge go to
        # the higher bucket, which matches the intent (low = below first edge).
        labels[mask] = np.digitize(rv.values[mask], self._edges)
        return pd.Series(labels, index=rv.index, name="regime")
