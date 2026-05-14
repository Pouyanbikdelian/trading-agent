r"""Multi-feature regime classifier — Gaussian HMM on cross-asset signals.

References
----------
Hamilton, J. D. (1989). *A new approach to the economic analysis of
nonstationary time series and the business cycle.* Econometrica, 57(2),
357-384.

Ang, A. and Timmermann, A. (2012). *Regime changes and financial
markets.* Annual Review of Financial Economics, 4, 313-337.

Model
-----
Daily observations :math:`\mathbf{x}_t \in \mathbb{R}^d` are driven by a
discrete latent state :math:`s_t \in \{1, \ldots, K\}` that follows a
homogeneous Markov chain with transition matrix :math:`A` and stationary
emission

.. math::

   \mathbf{x}_t \mid s_t = k  \;\sim\;
   \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).

Parameters :math:`(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, A)` are
estimated by Baum-Welch (EM); the posterior :math:`P(s_t \mid
\mathbf{x}_{1:T})` and the Viterbi MAP sequence
:math:`\arg\max_{s_{1:T}} P(s_{1:T}, \mathbf{x}_{1:T})` are recovered in
closed form on the fitted model.

Feature vector
--------------
We extend the univariate :class:`HmmRegime` from
:mod:`trading.regime.hmm` to a four-dimensional cross-asset signal:

1. :math:`r_t^{\text{mkt}}` — daily log return of the broad market
   (SPY by default).
2. :math:`\log \widehat\sigma_t^{\text{mkt}}` — rolling-20 standard
   deviation of (1), in logs so the HMM emission stays approximately
   Gaussian.
3. :math:`r_t^{\text{TLT}} - r_t^{\text{IEF}}` — long-duration minus
   intermediate-duration Treasury return. A proxy for term-premium
   shocks: positive in flight-to-quality, negative when yields rise.
4. :math:`r_t^{\text{HYG}} - r_t^{\text{LQD}}` — high-yield minus
   investment-grade credit return. A proxy for the credit-spread
   factor: positive in risk-on, negative in credit stress.

This four-feature set deliberately uses only *price-derived* inputs from
ETFs available through free data sources (yfinance), so the classifier
runs without macroeconomic releases.  The choice mirrors the
factor-zoo distillation in Cochrane (2011, *Discount Rates*) where
yield-curve and credit factors are the two most-cited drivers of
expected returns beyond the market.

States are returned sorted by their emission mean on the *return*
dimension, so :math:`s = 0` is the lowest-mean (bear-like) state and
:math:`s = K-1` the highest-mean (bull-like) state.  This makes the
labels stable across re-fits and machines (hmmlearn's internal ordering
is random).

Practical notes
---------------
* EM is sensitive to initialisation.  ``random_state`` is held fixed
  for reproducibility; large universes / longer histories warrant
  multi-start fitting.
* The ``covariance_type='diag'`` default is sufficient for these
  features (cross-feature correlations are small).  Switch to
  ``'full'`` if the term-spread / credit-spread channel becomes
  important in your application.
* Warm-up: the first ``vol_window`` bars carry insufficient information
  to compute feature (2); :meth:`predict` emits ``-1`` for those bars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading.regime.base import _ensure_returns

if TYPE_CHECKING:  # pragma: no cover
    from hmmlearn.hmm import GaussianHMM  # noqa: F401


class HmmMacroParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_states: int = Field(default=3, ge=2, le=5)
    vol_window: int = Field(default=20, ge=5)
    covariance_type: str = "diag"
    n_iter: int = Field(default=300, ge=10)
    random_state: int = 42
    tol: float = Field(default=1e-3, gt=0.0)


def build_features(
    market: pd.Series,
    *,
    tlt: pd.Series | None = None,
    ief: pd.Series | None = None,
    hyg: pd.Series | None = None,
    lqd: pd.Series | None = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    r"""Assemble the four-feature observation matrix.

    Inputs are *price* series (close levels), inner-joined and converted
    to log returns.  Missing optional series (e.g. an FX user without
    bond / credit ETFs) produce a column of zeros so the HMM sees the
    full feature vector but learns it has no information.
    """
    m = _ensure_returns(market.pct_change()).rename("mkt_ret")
    log_vol = np.log(m.rolling(vol_window, min_periods=vol_window).std(ddof=1)).rename("log_vol")

    def _diff(a: pd.Series | None, b: pd.Series | None) -> pd.Series:
        if a is None or b is None:
            return pd.Series(0.0, index=m.index, dtype=float)
        ar = a.pct_change()
        br = b.pct_change()
        return (ar - br).reindex(m.index).fillna(0.0)

    term = _diff(tlt, ief).rename("term_spread_proxy")
    credit = _diff(hyg, lqd).rename("credit_spread_proxy")

    return pd.concat([m, log_vol, term, credit], axis=1).dropna()


class HmmMacroRegime:
    r"""Multi-feature Gaussian HMM regime classifier.

    Fit/predict shape matches :class:`trading.regime.RegimeClassifier`.
    State IDs are remapped post-fit so :math:`s = 0` is the most-bearish
    (lowest mean return) state and :math:`s = K-1` the most-bullish.
    """

    def __init__(self, params: HmmMacroParams | None = None, **kwargs: object) -> None:
        if params is None:
            params = HmmMacroParams(**kwargs)  # type: ignore[arg-type]
        self.params = params
        self.n_states = params.n_states
        self._model: Any = None
        self._state_order: np.ndarray | None = None

    def fit(self, features: pd.DataFrame) -> HmmMacroRegime:
        from hmmlearn.hmm import GaussianHMM  # lazy import — heavy

        x = features.dropna().to_numpy(dtype=float)
        if x.shape[0] < self.params.n_states * 10:
            raise ValueError(
                f"need >= {self.params.n_states * 10} observations to fit "
                f"a {self.params.n_states}-state HMM; got {x.shape[0]}"
            )
        self._model = GaussianHMM(
            n_components=self.params.n_states,
            covariance_type=self.params.covariance_type,
            n_iter=self.params.n_iter,
            random_state=self.params.random_state,
            tol=self.params.tol,
        )
        self._model.fit(x)
        self._state_order = _rank_states_by_first_mean(self._model)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self._model is None or self._state_order is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        usable = features.dropna()
        if usable.empty:
            return pd.Series(dtype=int, name="macro_regime")
        raw = self._model.predict(usable.to_numpy(dtype=float))
        mapped = self._state_order[raw]
        out = pd.Series(-1, index=features.index, name="macro_regime", dtype=int)
        out.loc[usable.index] = mapped
        return out

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        r"""Return the smoothed posterior :math:`P(s_t = k \mid \mathbf{x}_{1:T})`
        for every bar.  Columns are the sorted state IDs."""
        if self._model is None or self._state_order is None:
            raise RuntimeError("call .fit(...) before .predict_proba(...)")
        usable = features.dropna()
        if usable.empty:
            return pd.DataFrame(columns=range(self.n_states))
        prob_raw = self._model.predict_proba(usable.to_numpy(dtype=float))
        # Re-order columns to match the sorted state IDs.
        inv_order = np.argsort(self._state_order)
        prob_sorted = prob_raw[:, inv_order]
        return pd.DataFrame(prob_sorted, index=usable.index, columns=list(range(self.n_states)))


def _rank_states_by_first_mean(model: Any) -> np.ndarray:
    r"""Return ``order`` such that ``order[hmm_state_i]`` is the
    sorted-rank of state :math:`i` by its emission mean on the first
    feature dimension (market return).  Lowest mean -> rank 0."""
    means = np.asarray(model.means_)[:, 0]
    sorted_idx = np.argsort(means)
    order = np.empty_like(sorted_idx)
    for rank, raw in enumerate(sorted_idx):
        order[raw] = rank
    return order
