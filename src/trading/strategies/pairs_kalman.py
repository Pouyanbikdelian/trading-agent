r"""Dynamic-hedge-ratio pair trading via the Kalman filter.

References
----------
Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and
Techniques.* Wiley. Chs. 3-4.

Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their
Rationale.* Wiley. Ch. 3 (Kalman filter as time-varying regression).

Model
-----
For a pair :math:`(Y_t, X_t)` of price series we posit the local linear
relation

.. math::

   Y_t = \beta_t X_t + \varepsilon_t, \qquad
   \varepsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, V),

with a hidden time-varying slope :math:`\beta_t` that evolves as a
random walk:

.. math::

   \beta_t = \beta_{t-1} + \eta_t, \qquad
   \eta_t \overset{\text{iid}}{\sim} \mathcal{N}(0, W).

This is the textbook state-space dynamic linear model.  The hidden state
is the hedge ratio; conditional on the observation history
:math:`\mathcal{F}_t = \sigma(Y_{1:t}, X_{1:t})` it is Gaussian, so the
Kalman recursions deliver the posterior mean :math:`m_t = E[\beta_t \mid
\mathcal{F}_t]` and variance :math:`P_t = \mathrm{Var}(\beta_t \mid
\mathcal{F}_t)` in closed form.

Recursion (one-step predict, observe, update)::

    m_{t|t-1} = m_{t-1}                                (predict mean)
    P_{t|t-1} = P_{t-1} + W                            (predict var)
    e_t       = Y_t - m_{t|t-1} X_t                    (innovation)
    S_t       = X_t^2 P_{t|t-1} + V                    (innovation var)
    K_t       = P_{t|t-1} X_t / S_t                    (Kalman gain)
    m_t       = m_{t|t-1} + K_t e_t                    (filter mean)
    P_t       = (1 - K_t X_t) P_{t|t-1}                (filter var)

The standardised innovation :math:`z_t = e_t / \sqrt{S_t}` is, by
construction, a unit-variance white-noise sequence under the model.  We
treat its excursions as the trading signal:

* enter long  :math:`Y` / short :math:`X` when :math:`z_t < -z_{\text{in}}`,
* enter short :math:`Y` / long  :math:`X` when :math:`z_t > +z_{\text{in}}`,
* close the position when :math:`|z_t| < z_{\text{out}}`.

The hedge ratio at trade-open is :math:`m_t`; we lag by one bar in the
weights produced (see :meth:`generate`) so no future information enters
the position decision.

Hyperparameters
---------------
:math:`V`  — observation variance.  Default is the residual variance of
    an OLS fit over the first ``fit_window`` bars.  This is consistent
    with treating the OLS run as the prior anchor.

:math:`W`  — process variance of the random-walk on :math:`\beta`.  No
    closed-form estimator; tuned by ``delta`` which sets the
    *information half-life* of the filter (see Pole 2007 §3.4):
    :math:`W = (\delta / (1-\delta)) \cdot P_0`.  Larger :math:`\delta`
    (closer to 1) means :math:`\beta` adapts faster; smaller means it
    is held nearly fixed.

The defaults (delta = 1e-4, fit_window = 60) match the parameterisation
used in Chan (2013, p. 76) for liquid US equity pairs at daily resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from trading.strategies.base import Strategy, StrategyParams, register


class KalmanPairsParams(StrategyParams):
    fit_window: int = Field(default=60, ge=20)
    """Bars used for the OLS prior fit that initialises :math:`m_0, P_0, V`."""

    delta: float = Field(default=1e-4, gt=0.0, lt=1.0)
    """Information decay; sets :math:`W = (\\delta/(1-\\delta)) P_0`."""

    entry_z: float = Field(default=2.0, gt=0.0)
    exit_z: float = Field(default=0.5, ge=0.0)

    weight_per_leg: float = Field(default=0.5, gt=0.0)
    beta_hedge: bool = True
    """If ``True`` the short leg is sized as :math:`-m_t \\cdot
    \\text{weight\\_per\\_leg}` to deliver a dollar-neutral position."""

    @model_validator(mode="after")
    def _exit_lt_entry(self) -> KalmanPairsParams:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be strictly less than entry_z")
        return self


@register
class KalmanPairs(Strategy):
    r"""Dynamic-hedge-ratio pair trading via Kalman filtering of the slope.

    Input ``prices`` must contain exactly two columns, taken as
    :math:`(Y, X)` in that column order.
    """

    name = "kalman_pairs"
    Params = KalmanPairsParams

    def generate(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.shape[1] != 2:
            raise ValueError("KalmanPairs requires exactly two price columns (Y, X)")
        p = self.params
        y_name, x_name = prices.columns[0], prices.columns[1]
        y = prices[y_name].to_numpy(dtype=float)
        x = prices[x_name].to_numpy(dtype=float)
        n = len(y)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if n < p.fit_window + 2:
            return weights

        # --- 1. OLS prior on the fit window -----------------------------
        y0 = y[: p.fit_window]
        x0 = x[: p.fit_window]
        beta_hat, V = _ols_beta_and_residual_var(y0, x0)
        # Initial state variance is the *estimator* variance of beta_hat
        # under OLS:  V / sum(x^2).  This is much smaller than V itself
        # when x is in price-level units; using V here is the classic
        # textbook mis-step that makes the predicted observation variance
        # x^2 * P + V collapse onto x^2 * V, swamping the V-component and
        # standardising every innovation to ~0.
        sum_x2 = float(np.dot(x0, x0))
        P = max(V / sum_x2, 1e-12) if sum_x2 > 0 else 1e-6
        W = (p.delta / (1.0 - p.delta)) * P

        # --- 2. Kalman recursion + standardised innovations -------------
        beta = np.full(n, np.nan)
        z = np.full(n, np.nan)
        m, P_t = beta_hat, P
        for t in range(p.fit_window, n):
            # predict
            P_pred = P_t + W
            # innovation
            e = y[t] - m * x[t]
            S = x[t] * x[t] * P_pred + V
            if S <= 0:
                S = max(V, 1e-12)
            # update
            K = P_pred * x[t] / S
            m = m + K * e
            P_t = (1.0 - K * x[t]) * P_pred
            beta[t] = m
            z[t] = e / np.sqrt(S)

        beta_s = pd.Series(beta, index=prices.index)
        z_s = pd.Series(z, index=prices.index)

        # --- 3. Position state machine on the lagged z ------------------
        z_lag = z_s.shift(1)
        enter_long = z_lag < -p.entry_z
        enter_short = z_lag > p.entry_z
        exit_signal = z_lag.abs() < p.exit_z

        event = pd.Series(np.nan, index=prices.index)
        event[enter_long] = 1.0
        event[enter_short] = -1.0
        event[exit_signal] = 0.0
        sign = event.ffill().fillna(0.0).to_numpy()

        weights.loc[:, y_name] = sign * p.weight_per_leg
        if p.beta_hedge:
            beta_lag = beta_s.shift(1).fillna(0.0).to_numpy()
            weights.loc[:, x_name] = -sign * beta_lag * p.weight_per_leg
        else:
            weights.loc[:, x_name] = -sign * p.weight_per_leg
        return weights


def _ols_beta_and_residual_var(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    r"""OLS slope through the origin and the residual variance.

    Computed without the statsmodels dependency:
    :math:`\hat\beta = (X^\top X)^{-1} X^\top Y = \sum x_t y_t / \sum x_t^2`,
    :math:`\hat V = \frac{1}{T-1} \sum (y_t - \hat\beta x_t)^2`.
    """
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0, 1.0
    beta = float(np.dot(x, y) / denom)
    resid = y - beta * x
    V = float(np.var(resid, ddof=1)) if len(resid) > 1 else 1.0
    return beta, max(V, 1e-12)
