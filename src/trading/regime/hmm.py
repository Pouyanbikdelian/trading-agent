"""Gaussian Hidden Markov Model regime classifier.

Backed by ``hmmlearn.hmm.GaussianHMM`` with a 1-D emission (returns). The
HMM states themselves are arbitrary IDs; we sort them by emission mean
after fitting so ``state=0`` is always the most-bearish (lowest mean) and
``state = n_states-1`` is the most-bullish. That makes the labels stable
across re-fits and across machines (hmmlearn's state ordering is non-
deterministic out of the box).

Caveats
-------
* HMMs are sensitive to initialization. We use ``random_state=42`` for
  reproducibility and ``n_iter=200`` to give EM room to converge. If you
  need a different model, train your own and use ``HmmRegime.from_fitted``.
* The model is fit on raw returns. For series with very different scales
  (equities vs. FX), fit one HMM per series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trading.regime.base import _ensure_returns

if TYPE_CHECKING:  # pragma: no cover
    from hmmlearn.hmm import GaussianHMM  # noqa: F401


class HmmParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_states: int = Field(default=2, ge=2, le=6)
    covariance_type: str = Field(default="diag")
    n_iter: int = Field(default=200, ge=10)
    random_state: int = 42
    tol: float = Field(default=1e-3, gt=0.0)


class HmmRegime:
    """Gaussian-emission HMM over returns."""

    def __init__(self, params: HmmParams | None = None, **kwargs: object) -> None:
        if params is None:
            params = HmmParams(**kwargs)  # type: ignore[arg-type]
        self.params = params
        self.n_states = params.n_states
        self._model: Any = None
        # After fit: index i -> sorted-rank of HMM state i (0 = most bearish).
        self._state_order: np.ndarray | None = None

    @classmethod
    def from_fitted(cls, model: Any, n_states: int) -> "HmmRegime":
        """Wrap a pre-fit ``GaussianHMM`` (or compatible) without re-training."""
        inst = cls(HmmParams(n_states=n_states))
        inst._model = model
        inst._state_order = _rank_states_by_mean(model)
        return inst

    def fit(self, returns: pd.Series) -> "HmmRegime":
        from hmmlearn.hmm import GaussianHMM   # lazy import — heavy

        r = _ensure_returns(returns)
        x = r.values.reshape(-1, 1)
        self._model = GaussianHMM(
            n_components=self.params.n_states,
            covariance_type=self.params.covariance_type,
            n_iter=self.params.n_iter,
            random_state=self.params.random_state,
            tol=self.params.tol,
        )
        self._model.fit(x)
        self._state_order = _rank_states_by_mean(self._model)
        return self

    def predict(self, returns: pd.Series) -> pd.Series:
        if self._model is None or self._state_order is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        r = _ensure_returns(returns)
        raw = self._model.predict(r.values.reshape(-1, 1))
        # Remap raw state IDs into mean-sorted state IDs.
        mapped = self._state_order[raw]
        return pd.Series(mapped, index=r.index, name="regime")


def _rank_states_by_mean(model: Any) -> np.ndarray:
    """Return an array ``order`` such that ``order[hmm_state_i]`` is the
    mean-sorted index of that state (0 = lowest mean)."""
    means = np.asarray(model.means_).ravel()
    sorted_idx = np.argsort(means)
    order = np.empty_like(sorted_idx)
    for rank, raw in enumerate(sorted_idx):
        order[raw] = rank
    return order
