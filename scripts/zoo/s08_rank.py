"""Step 8 — rank the zoo: per regime, overall, vs luck; train-only ensembles.

Survivorship flag: long-only STOCK books (families xsection, ml long, meanrev
stock books) are flattered by using today's index members. ETF/index rules
are not. Rankings are shown both ways.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew

from harness import OUT, REGIMES, TRAIN_END, regimes, sharpe

sys.path.insert(0, str(OUT.parents[1] / "src"))
from trading.selection.scores import deflated_sharpe  # noqa: E402

STOCK_LONG = ("top25 long", "top50 long", "Weekly reversal", "Bounce screen",
              "Min variance stocks")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    rets, stats = [], []
    for p in sorted(OUT.glob("ret_s0*.parquet")):
        rets.append(pd.read_parquet(p))
        stats.append(pd.read_csv(OUT / p.name.replace("ret_", "stats_").replace(".parquet", ".csv")))
    R = pd.concat(rets, axis=1)
    R = R.loc[:, ~R.columns.duplicated()].loc["2019-03-01":]  # common window, all warmed up
    S = pd.concat(stats, ignore_index=True).drop_duplicates("name")
    S["survivorship"] = S["name"].apply(lambda n: any(k in n for k in STOCK_LONG))
    return R, S


def table(R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    lab = regimes().reindex(R.index)
    rows = []
    n_trials = R.shape[1]
    srs = (R.mean() / R.std()).dropna()
    for c in R.columns:
        r = R[c].dropna()
        tr, te = r[r.index <= TRAIN_END], r[r.index > TRAIN_END]
        sr = r.mean() / r.std()
        dsr = deflated_sharpe(sr, len(r), float(skew(r)), float(kurtosis(r, fisher=False)),
                              n_trials, float(srs.std()))
        eq = (1 + r).cumprod()
        row = {"name": c, "sharpe": sharpe(r), "train": sharpe(tr), "test": sharpe(te),
               "cagr": eq.iloc[-1] ** (252 / len(r)) - 1, "maxdd": (eq / eq.cummax() - 1).min(),
               "DSR": dsr, "corr_spy": r.corr(R["SPY buy&hold"])}
        for g in REGIMES:
            row[g] = sharpe(r[lab == g])
        rows.append(row)
    T = pd.DataFrame(rows).merge(S[["name", "family", "survivorship"]], on="name")
    return T


def ensembles(R: pd.DataFrame, T: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """All weights chosen on TRAIN data only, then frozen."""
    tr = R[R.index <= TRAIN_END]
    lab = regimes().reindex(R.index)
    clean = T[~T["survivorship"] & (T["family"] != "benchmark")]
    out = []
    # A. top-8 clean strategies by train Sharpe, low mutual correlation, equal risk
    cand = clean.sort_values("train", ascending=False)["name"].tolist()
    picked: list[str] = []
    for c in cand:
        if all(abs(tr[c].corr(tr[p])) < 0.7 for p in picked):
            picked.append(c)
        if len(picked) == 8:
            break
    iv = 1 / tr[picked].std()
    w = iv / iv.sum()
    out.append(("ENSEMBLE equal-risk top-8 (train-picked, no stock longs)",
                (R[picked] * w).sum(axis=1), picked))
    # B. min-variance of those 8, weights fit on train covariance
    cov = tr[picked].cov().values
    n = len(picked)
    res = minimize(lambda x: x @ cov @ x, np.full(n, 1 / n), bounds=[(0, 0.4)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP")
    out.append(("ENSEMBLE min-variance top-8", (R[picked] * res.x).sum(axis=1), picked))
    # C. max-Sharpe on train (the overfit-prone one, for contrast)
    mu = tr[picked].mean().values
    res2 = minimize(lambda x: -(x @ mu) / math.sqrt(x @ cov @ x), np.full(n, 1 / n),
                    bounds=[(0, 0.4)] * n,
                    constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP")
    out.append(("ENSEMBLE max-Sharpe top-8 (overfit-prone)", (R[picked] * res2.x).sum(axis=1), picked))
    # D. regime switch: in each regime hold the 3 clean strategies with best TRAIN
    #    Sharpe in that regime, equal risk
    sw = pd.Series(0.0, index=R.index)
    regime_picks = {}
    for g in REGIMES:
        trg = tr[lab.reindex(tr.index) == g]
        best = trg[clean["name"]].apply(sharpe).sort_values(ascending=False).head(3).index.tolist()
        regime_picks[g] = best
        ivg = 1 / tr[best].std()
        wg = ivg / ivg.sum()
        sw[lab == g] = (R.loc[lab == g, best] * wg).sum(axis=1)
    out.append(("ENSEMBLE regime-switch (best 3 per regime, train)", sw, regime_picks))
    return out


def main() -> None:
    R, S = load()
    T = table(R, S)
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 48)
    cols = ["name", "sharpe", "train", "test", "cagr", "maxdd", "DSR", "corr_spy", *REGIMES]
    print(f"{R.shape[1]} strategies, common window {R.index[0].date()} -> {R.index[-1].date()}\n")
    print("=== OVERALL — no survivorship bias (ETF/index rules, hedged books, pairs) ===")
    print(T[~T.survivorship].sort_values("sharpe", ascending=False)[cols].head(20)
          .round(2).to_string(index=False))
    print("\n=== OVERALL — stock long books (FLATTERED by today's index members) ===")
    print(T[T.survivorship].sort_values("sharpe", ascending=False)[cols].head(8)
          .round(2).to_string(index=False))
    for g in REGIMES:
        print(f"\n=== BEST IN REGIME: {g} (clean only) ===")
        print(T[~T.survivorship].sort_values(g, ascending=False)[["name", g, "sharpe", "test"]]
              .head(6).round(2).to_string(index=False))
    print("\n=== TRAIN-ONLY ENSEMBLES (weights frozen at end-2022) ===")
    erows = []
    for name, r, picks in ensembles(R, T):
        tr, te = r[r.index <= TRAIN_END], r[r.index > TRAIN_END]
        eq = (1 + r).cumprod()
        lab = regimes().reindex(r.index)
        erows.append({"name": name, "sharpe": sharpe(r), "train": sharpe(tr), "test": sharpe(te),
                      "cagr": eq.iloc[-1] ** (252 / len(r)) - 1,
                      "maxdd": (eq / eq.cummax() - 1).min(),
                      **{g: sharpe(r[lab == g]) for g in REGIMES}})
        print(f"  {name}: {picks}")
    print(pd.DataFrame(erows).round(2).to_string(index=False))
    T.to_csv(OUT / "zoo_ranking.csv", index=False)


if __name__ == "__main__":
    main()
