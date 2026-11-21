"""Step 5 — pairs trading (Gatev-Goetzmann-Rouwenhorst distance method).

Formation 252d: normalise prices, pick the n pairs with the smallest sum of
squared spread distance. Trading 126d: open when the spread moves 2 formation
sigmas, close when it crosses zero (or the period ends). Dollar-neutral.
Variants: any pair / only pairs inside the same correlation cluster /
cointegration-filtered (Engle-Granger p<0.05).
"""

# ruff: noqa: N806
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from harness import ROOT, evaluate, prices, save, show, universe

_spec = importlib.util.spec_from_file_location("ls", ROOT / "scripts/ls_search.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)  # type: ignore[union-attr]

FORM, TRADE = 252, 126


def pick_pairs(form: pd.DataFrame, n: int, mode: str, clu: pd.Series | None):
    norm = form / form.iloc[0]
    x = norm.values
    sq = (x**2).sum(0)
    ssd = sq[:, None] + sq[None, :] - 2 * x.T @ x
    np.fill_diagonal(ssd, np.inf)
    iu = np.triu_indices_from(ssd, 1)
    order = np.argsort(ssd[iu])
    cols = norm.columns
    out = []
    from statsmodels.tsa.stattools import coint

    for k in order[:4000]:
        a, b = cols[iu[0][k]], cols[iu[1][k]]
        if mode == "cluster" and (clu is None or clu.get(a) != clu.get(b)):
            continue
        if mode == "coint" and coint(np.log(form[a].values), np.log(form[b].values))[1] > 0.05:
            continue
        sig = float((norm[a] - norm[b]).std())
        out.append((a, b, sig))
        if len(out) >= n:
            break
    return out


def run(mode: str, n: int = 20) -> pd.DataFrame:
    px = prices()
    stocks = [s for s in universe() if s in px.columns]
    P = px[stocks]
    rets = P.pct_change(fill_method=None)
    w = pd.DataFrame(0.0, index=P.index, columns=stocks)
    starts = range(FORM, len(P) - 5, TRADE)
    clu_all = ls.clusters(rets, P.index[list(starts)]) if mode == "cluster" else None
    for s in starts:
        form = P.iloc[s - FORM : s].dropna(axis=1)
        trade = P.iloc[s : s + TRADE][form.columns]
        clu = clu_all.loc[P.index[s]] if clu_all is not None else None
        pairs = pick_pairs(form, n, mode, clu)
        base = trade.iloc[0]
        for a, b, sig in pairs:
            spread = trade[a] / base[a] - trade[b] / base[b]
            pos = 0.0
            ps = []
            for v in spread.values:
                if np.isnan(v):
                    pos = 0.0
                elif pos == 0 and v > 2 * sig:
                    pos = -1.0
                elif pos == 0 and v < -2 * sig:
                    pos = 1.0
                elif (pos == -1 and v <= 0) or (pos == 1 and v >= 0):
                    pos = 0.0
                ps.append(pos)
            ps = pd.Series(ps, index=trade.index)
            ps.iloc[-1] = 0.0
            w.loc[trade.index, a] += ps / n
            w.loc[trade.index, b] -= ps / n
    return w


def main() -> None:
    res = []
    for mode in ["any", "cluster", "coint"]:
        res.append(evaluate(f"Pairs distance ({mode}), 20 pairs", "pairs", run(mode)))
        print(f"done {mode}", flush=True)
    tab = save(res, "s05_pairs")
    show(tab)


if __name__ == "__main__":
    main()
