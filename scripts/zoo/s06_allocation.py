"""Step 6 — portfolio construction: inverse-vol, equal risk contribution,
minimum variance (ETFs and stocks), SPY volatility targeting.

Covariances use 252d of past daily returns with Ledoit-Wolf-style constant-
correlation shrinkage (0.5) — sample covariance on 500 names is noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from harness import ETFS, evaluate, monthly_dates, prices, returns, save, show, universe


def shrink_cov(x: np.ndarray, a: float = 0.5) -> np.ndarray:
    s = np.cov(x, rowvar=False)
    sd = np.sqrt(np.diag(s))
    c = s / np.outer(sd, sd)
    rbar = (c.sum() - len(c)) / (len(c) ** 2 - len(c))
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, sd**2)
    return a * target + (1 - a) * s


def minvar(cov: np.ndarray, cap: float) -> np.ndarray:
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w,
                   bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 200})
    return res.x if res.success else np.full(n, 1 / n)


def erc(cov: np.ndarray, iters: int = 200) -> np.ndarray:
    w = 1 / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        rc = w * (cov @ w)
        w *= (rc.mean() / rc) ** 0.5
        w /= w.sum()
    return w


def main() -> None:
    px = prices()
    r = returns()
    m = monthly_dates(px.index)
    res = []
    # ETF universes: full 11, and a risk-asset-light core
    for label, uni in [("11 ETFs", ETFS), ("core 6", ["SPY", "EFA", "IEF", "TLT", "GLD", "DBC"])]:
        w_iv = pd.DataFrame(0.0, index=m, columns=uni)
        w_erc, w_mv = w_iv.copy(), w_iv.copy()
        for d in m:
            x = r[uni].loc[:d].iloc[-252:].dropna()
            if len(x) < 200:
                continue
            cov = shrink_cov(x.values)
            iv = 1 / np.sqrt(np.diag(cov))
            w_iv.loc[d] = iv / iv.sum()
            w_erc.loc[d] = erc(cov)
            w_mv.loc[d] = minvar(cov, 0.5)
        res.append(evaluate(f"Inverse-vol {label}", "allocation", w_iv))
        res.append(evaluate(f"Risk parity (ERC) {label}", "allocation", w_erc))
        res.append(evaluate(f"Min variance {label}", "allocation", w_mv))
        # levered to 10% vol: risk parity's whole point
        port = (w_erc.reindex(px.index).ffill().shift(1) * r[uni]).sum(axis=1)
        k = (0.10 / (port.rolling(60).std() * np.sqrt(252))).clip(upper=2.5)
        res.append(evaluate(f"Risk parity {label} @10% vol", "allocation",
                            w_erc.reindex(px.index).ffill().mul(k, axis=0)))
    # stock min-variance: 150 lowest-vol names each month, capped 3%
    stocks = [s for s in universe() if s in px.columns]
    vol = r[stocks].rolling(252, min_periods=200).std()
    w_s = pd.DataFrame(0.0, index=m, columns=stocks)
    for d in m:
        v = vol.loc[d].dropna()
        if len(v) < 200:
            continue
        pick = v.nsmallest(150).index
        x = r[pick].loc[:d].iloc[-252:].fillna(0).values
        w_s.loc[d, pick] = minvar(shrink_cov(x), 0.03)
    res.append(evaluate("Min variance stocks (150 low-vol, 3% cap)", "allocation", w_s))
    # SPY vol targeting
    for tgt in [0.10, 0.15]:
        rv = r["SPY"].rolling(21).std() * np.sqrt(252)
        k = (tgt / rv).clip(upper=1.5)
        res.append(evaluate(f"SPY vol-target {int(tgt*100)}%", "allocation", pd.DataFrame({"SPY": k})))
    tab = save(res, "s06_allocation")
    show(tab)


if __name__ == "__main__":
    main()
