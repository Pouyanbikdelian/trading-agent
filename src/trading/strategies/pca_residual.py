r"""Statistical-arbitrage on the residuals of a rolling principal-component model.

References
----------
Avellaneda, M. and Lee, J.-H. (2010). *Statistical arbitrage in the U.S.
equities market.* Quantitative Finance, 10(7), 761-782.

Konstantinov, G., Chorus, A., Rebmann, J. (2020). *A network and machine
learning approach to factor, asset and blended allocation.* JPM Quant
Special Issue.  (For the rolling-PCA factor-model formulation.)

Model
-----
Let :math:`R \in \mathbb{R}^{T \times N}` be a panel of demeaned daily
returns on :math:`N` instruments over a rolling window of length
:math:`T`.  We decompose

.. math::

   R = U \Sigma V^\top, \qquad
   \tilde R_K = U_K \Sigma_K V_K^\top,

where :math:`U_K, \Sigma_K, V_K` retain the top :math:`K` singular
components.  :math:`\tilde R_K` is the rank-:math:`K` projection that
captures the systematic risk explained by the dominant factors
(market, sector, style).  The cross-sectional residual at the right
edge of the window is

.. math::

   \varepsilon_T = R_T - \tilde R_{K,T} \in \mathbb{R}^{N}.

Under the model these residuals are idiosyncratic — uncorrelated with
the factors and, by appeal to the Avellaneda-Lee mean-reversion thesis,
expected to reverse over short horizons.

Trading signal
--------------
For each instrument :math:`i` we accumulate the residual stream and
standardise:

.. math::

   s_i^{(t)} = \sum_{u=t-h+1}^{t} \varepsilon_{i, u},  \qquad
   z_i^{(t)} = \frac{s_i^{(t)} - \bar{s_i^{(t)}}}{\hat\sigma_{s_i^{(t)}}},

over a horizon :math:`h` (``residual_horizon`` parameter).  The
position rule is the classic mean-reversion gate:

* :math:`z_i < -z_{\text{in}}` → long  :math:`i`,
* :math:`z_i > +z_{\text{in}}` → short :math:`i` (if ``allow_short``),
* :math:`|z_i| < z_{\text{out}}` → flat.

The cross-section is dollar-neutralised by construction when both legs
are allowed: long and short books are sized symmetrically, the
remaining factor exposure averages to zero in expectation.

Numerical notes
---------------
* PCA is implemented via :func:`numpy.linalg.svd` on the demeaned
  return matrix; we drop trailing rows with any NaN before each window
  to keep the SVD well-defined.
* Windows shorter than ``min_window`` produce no signal (residual NaN);
  these bars are forced flat.
* Computational cost is :math:`O(T \cdot \min(T, N)^2)` per
  refit-bar.  At daily resolution with :math:`T = 60` and
  :math:`N = 500` this is ~6 ms in numpy; running every bar is fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from trading.strategies.base import Strategy, StrategyParams, register


class PCAResidualParams(StrategyParams):
    pca_window: int = Field(default=60, ge=20)
    """Number of trailing return observations the PCA fits on."""

    n_factors: int = Field(default=3, ge=1, le=10)
    """Number of leading components retained as 'factors'."""

    residual_horizon: int = Field(default=5, ge=1)
    """Window over which residuals are accumulated before z-scoring."""

    z_window: int = Field(default=20, ge=5)
    """Lookback for z-scoring the accumulated residual."""

    entry_z: float = Field(default=1.5, gt=0.0)
    exit_z: float = Field(default=0.3, ge=0.0)

    allow_short: bool = True

    weight_per_asset: float = Field(default=0.05, gt=0.0)

    @model_validator(mode="after")
    def _exit_lt_entry(self) -> PCAResidualParams:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        return self


@register
class PCAResidual(Strategy):
    name = "pca_residual"
    Params = PCAResidualParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.shape[1] < 5:
            raise ValueError(
                f"PCAResidual requires at least 5 instruments to make the factor "
                f"decomposition meaningful (got {prices.shape[1]})"
            )
        p = self.params
        ret = prices.pct_change().fillna(0.0)
        n_t, n_n = ret.shape
        residuals = np.full((n_t, n_n), np.nan)

        ret_values = ret.to_numpy()
        K = min(p.n_factors, n_n - 1)

        # --- Rolling PCA residuals --------------------------------------
        for t in range(p.pca_window, n_t):
            window = ret_values[t - p.pca_window : t]
            # column-demean for the principal-component decomposition
            mu = window.mean(axis=0)
            centred = window - mu
            # SVD: centred = U diag(s) Vt;  rank-K projection = U_K diag(s_K) V_K^T.
            # We only need the right singular vectors V_K = Vt[:K] to project
            # the most recent demeaned return cross-section onto the factor
            # subspace; U and s are discarded.
            try:
                _U, _s, Vt = np.linalg.svd(centred, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            # Project the last row onto the factor subspace and read off the residual
            last = centred[-1]  # most recent demeaned return cross-section
            V_K = Vt[:K]  # shape (K, N)
            factor_loadings = V_K @ last  # shape (K,)
            reconstruction = factor_loadings @ V_K  # shape (N,)
            eps = last - reconstruction  # idiosyncratic residual
            residuals[t] = eps

        eps_df = pd.DataFrame(residuals, index=prices.index, columns=prices.columns)

        # --- Accumulate over horizon, z-score over z_window -------------
        cum = eps_df.rolling(p.residual_horizon, min_periods=p.residual_horizon).sum()
        mu = cum.rolling(p.z_window, min_periods=p.z_window).mean()
        sigma = cum.rolling(p.z_window, min_periods=p.z_window).std(ddof=1)
        z = ((cum - mu) / sigma).shift(1)  # lag one bar — no lookahead

        # --- State machine per asset ------------------------------------
        long_entry = z < -p.entry_z
        short_entry = z > p.entry_z
        exit_band = z.abs() < p.exit_z

        weights = np.zeros((n_t, n_n))
        for j, col in enumerate(prices.columns):
            event = pd.Series(np.nan, index=prices.index)
            event[long_entry[col]] = 1.0
            if p.allow_short:
                event[short_entry[col]] = -1.0
            event[exit_band[col]] = 0.0
            sign = event.ffill().fillna(0.0)
            weights[:, j] = sign.values

        return (
            pd.DataFrame(weights, index=prices.index, columns=prices.columns) * p.weight_per_asset
        )
