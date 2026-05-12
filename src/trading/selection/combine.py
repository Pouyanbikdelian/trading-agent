"""Strategy combiners — blend ``N`` per-strategy weight frames into one.

Each combiner takes a ``{name: weights_df}`` dict and returns a single
``DataFrame`` aligned to the input frames. ``returns_by_strategy`` (the
backtested net returns of each strategy) is required by combiners that
size by realized risk.

Combiners
---------
* ``equal_weight``: uniform average. Most robust to noisy estimates; a
  good baseline.
* ``inverse_vol``: each strategy gets weight proportional to ``1 / σ_i``,
  where σ_i is the realized vol over a fixed lookback. Strategies with
  noisier returns get less capital.
* ``min_variance``: closed-form portfolio that minimizes the in-sample
  variance under a sum-to-one constraint. Closed form via the inverse
  covariance matrix. Sensitive to the input strategies being roughly
  comparably scaled; if one is leveraged-up the optimizer will go all-in
  on it.

All three are *static* — they fit a single weight per strategy over a
training period, then apply it to the full series. For rolling combiners,
re-call the function per fold (Phase 5 callers already do this in walk-
forward setups).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align_frames(weights_by_strategy: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align every input frame to the common index and column set."""
    if not weights_by_strategy:
        raise ValueError("weights_by_strategy must not be empty")
    frames = list(weights_by_strategy.values())
    common_idx = frames[0].index
    common_cols = frames[0].columns
    for f in frames[1:]:
        common_idx = common_idx.intersection(f.index)
        common_cols = common_cols.intersection(f.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        raise ValueError("strategy weight frames have no overlap in rows or columns")
    return {
        k: v.loc[common_idx, common_cols].astype(float).fillna(0.0)
        for k, v in weights_by_strategy.items()
    }


def equal_weight(weights_by_strategy: dict[str, pd.DataFrame]) -> pd.DataFrame:
    aligned = _align_frames(weights_by_strategy)
    n = len(aligned)
    return sum(aligned.values()) / n  # type: ignore[no-any-return]


def _scalar_blend(
    aligned: dict[str, pd.DataFrame],
    scalar_weights: dict[str, float],
) -> pd.DataFrame:
    """Weighted sum of frames using a single scalar per strategy."""
    out = None
    for name, frame in aligned.items():
        contribution = frame * scalar_weights[name]
        out = contribution if out is None else out + contribution
    assert out is not None
    return out


def inverse_vol(
    weights_by_strategy: dict[str, pd.DataFrame],
    returns_by_strategy: dict[str, pd.Series],
    *,
    lookback: int | None = None,
) -> pd.DataFrame:
    """Inverse-volatility blend over the (last ``lookback`` bars of the)
    strategy returns. Lookback ``None`` uses the full series."""
    aligned = _align_frames(weights_by_strategy)
    scalars: dict[str, float] = {}
    for name in aligned:
        r = returns_by_strategy[name].dropna()
        if lookback is not None:
            r = r.iloc[-lookback:]
        vol = float(r.std(ddof=1))
        scalars[name] = 1.0 / vol if vol > 0 else 0.0
    s = sum(scalars.values())
    if s == 0:
        raise ValueError("all strategies have zero realized vol — cannot inverse-vol blend")
    scalars = {k: v / s for k, v in scalars.items()}
    return _scalar_blend(aligned, scalars)


def min_variance(
    weights_by_strategy: dict[str, pd.DataFrame],
    returns_by_strategy: dict[str, pd.Series],
    *,
    lookback: int | None = None,
    ridge: float = 1e-6,
) -> pd.DataFrame:
    """Closed-form long-only minimum-variance blend of the strategies.

    Solves ``w = Σ^-1 1 / (1' Σ^-1 1)`` then clips negatives to zero and
    re-normalizes. The ridge is added to the diagonal of Σ for numerical
    stability when strategies are highly correlated.
    """
    aligned = _align_frames(weights_by_strategy)
    names = list(aligned.keys())
    rets = pd.DataFrame({n: returns_by_strategy[n] for n in names}).dropna(how="any")
    if lookback is not None:
        rets = rets.iloc[-lookback:]
    if len(rets) < 2:
        raise ValueError("need at least 2 return observations to estimate covariance")

    sigma = rets.cov().values
    sigma = sigma + np.eye(len(names)) * ridge
    ones = np.ones(len(names))
    try:
        x = np.linalg.solve(sigma, ones)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"covariance matrix is singular: {e}") from e
    w = x / x.sum() if x.sum() != 0 else ones / len(names)
    w = np.clip(w, 0.0, None)
    w = ones / len(names) if w.sum() == 0 else w / w.sum()
    scalars = dict(zip(names, w.tolist(), strict=True))
    return _scalar_blend(aligned, scalars)
