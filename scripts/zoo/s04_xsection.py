"""Step 4 — cross-sectional stock factors, monthly. Long-only and SPY-hedged.

Includes the desk's own mechanical config (126d lookback, 21d skip, top-25)
so the zoo says where the current system ranks.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from harness import ROOT, evaluate, monthly_dates, prices, returns, save, show, universe

_spec = importlib.util.spec_from_file_location("qsb", ROOT / "scripts/quality_short_backtest.py")
qsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qsb)  # type: ignore[union-attr]


def topn_weights(score: pd.DataFrame, n: int, hedge: str | None, beta=None) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=score.index, columns=[*score.columns, "SPY"])
    for d, row in score.iterrows():
        row = row.dropna()
        if len(row) < 100:
            continue
        top = row.nlargest(n).index
        w.loc[d, top] = 1.0 / n
        if hedge == "spy":
            w.loc[d, "SPY"] = -1.0
        elif hedge == "beta" and beta is not None:
            w.loc[d, "SPY"] = -float(beta.loc[d, top].mean())
    return w


def main() -> None:
    px = prices()
    r = returns()
    stocks = [s for s in universe() if s in px.columns]
    m = monthly_dates(px.index)
    P = px[stocks]
    pm = P.loc[m]
    spy = r["SPY"]
    beta = r[stocks].rolling(252, min_periods=200).cov(spy).div(spy.rolling(252).var(), axis=0).loc[m]
    vol = r[stocks].rolling(60, min_periods=40).std().loc[m]
    sig = {
        "Mom 12-1": pm.shift(1) / pm.shift(12) - 1,
        "Desk top-k (126/21)": (P.shift(21) / P.shift(126) - 1).loc[m],
        "Mom 6-1": pm.shift(1) / pm.shift(6) - 1,
        "52w-high proximity": (P / P.rolling(252).max()).loc[m],
        "Low volatility": -vol,
        "Low beta": -beta,
    }
    # quality/value from point-in-time EDGAR fundamentals (exfin)
    fund = pd.read_parquet(ROOT / "output/quality_short/fundamentals_pit.parquet")
    sc = qsb.score(fund, pm, pm, exfin=True)
    q = sc.pivot(index="date", columns="sym", values="Q").reindex(index=m, columns=stocks)
    v = sc.pivot(index="date", columns="sym", values="V").reindex(index=m, columns=stocks)
    zq = q.sub(q.mean(1), axis=0).div(q.std(1), axis=0)
    zv = v.sub(v.mean(1), axis=0).div(v.std(1), axis=0)
    mom = sig["Mom 12-1"]
    zm = mom.sub(mom.mean(1), axis=0).div(mom.std(1), axis=0)
    sig["Quality"] = zq
    sig["Quality+Value"] = (zq + zv) / 2
    sig["Quality+Value+Mom"] = (zq + zv + zm) / 3
    sig["Quality+Mom"] = (zq + zm) / 2

    res = []
    for name, s in sig.items():
        for n in [25, 50]:
            res.append(evaluate(f"{name} top{n} long", "xsection", topn_weights(s, n, None)))
        res.append(evaluate(f"{name} top50 beta-hedged", "xsection_hedged",
                            topn_weights(s, 50, "beta", beta)))
    tab = save(res, "s04_xsection")
    show(tab)


if __name__ == "__main__":
    main()
