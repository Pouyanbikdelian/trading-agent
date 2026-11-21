"""Tactical relative-value long/short — an HONEST combination search.

Follow-up to quality_short_backtest.py (2026-11-20), where every
long-quality / short-junk variant lost money 2018-2026. Yan: "retry
different combos until you succeed". Searching until something wins is how
you manufacture a backtest that dies live, so the search here is disciplined:

* ~190 combinations of signal x neutralisation x short leg x size x weighting.
* Winners are SELECTED on 2018-2022 only, then JUDGED on 2023-2026, which
  the selection never saw.
* Deflated Sharpe (selection/scores.py) charges the winner for the number
  of trials. A winner that cannot clear it is luck.

New structures vs the first pass:
* Relative value: scores demeaned WITHIN clusters of co-moving stocks
  (hierarchical clustering on trailing 1y daily correlations, refit each
  January on past data only). Long the best of a group, short the worst of
  the SAME group — the "pair trade" idea at portfolio scale.
* Beta-neutral weighting: junk is high-beta, so a dollar-neutral book is
  secretly net-short the market and bleeds in rallies. Short notional is
  scaled by beta_long / beta_short (clipped 0.5-2).
* SPY hedge as the short leg: stock-picking edge without single-name
  squeeze risk.
* Signals: quality, value, 12-1 momentum, 1-month reversal, low volatility.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import kurtosis, skew, spearmanr

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qsb", ROOT / "scripts/quality_short_backtest.py")
qsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qsb)  # type: ignore[union-attr]

TRAIN_END = pd.Timestamp("2022-12-31")
COST_BP = 10.0
BORROW_SINGLE = 0.02
BORROW_SPY = 0.003
N_CLUSTERS = 20


def clusters(rets: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """{date x sym} cluster id, refit each January on the prior 252 days."""
    out = {}
    current: pd.Series | None = None
    last_year = None
    for d in dates:
        if d.year != last_year:
            win = rets.loc[:d].iloc[-252:]
            win = win.loc[:, win.notna().mean() > 0.9].fillna(0.0)
            if win.shape[1] > N_CLUSTERS:
                c = np.corrcoef(win.values.T)
                dist = np.clip(1 - c, 0, 2)
                iu = np.triu_indices_from(dist, 1)
                z = linkage(dist[iu], method="average")
                current = pd.Series(fcluster(z, N_CLUSTERS, "maxclust"), index=win.columns)
            last_year = d.year
        out[d] = current
    return pd.DataFrame(out).T


def panel() -> pd.DataFrame:
    px = qsb.load_prices()
    rets = px.pct_change()
    month_ends = px.resample("ME").last().index
    dates = pd.DatetimeIndex([px.index[px.index <= d][-1] for d in month_ends])
    m = px.loc[dates]
    fund = pd.read_parquet(ROOT / "output/quality_short/fundamentals_pit.parquet")
    sc = qsb.score(fund, m, m, exfin=True)[["date", "sym", "Q", "V"]]
    spy = rets["SPY"]
    beta = rets.rolling(252, min_periods=200).cov(spy).div(spy.rolling(252).var(), axis=0)
    vol = rets.rolling(60, min_periods=40).std()
    feats = {
        "mom": m.shift(1) / m.shift(12) - 1,
        "rev": -(m / m.shift(1) - 1),
        "lowvol": -vol.loc[dates],
        "beta": beta.loc[dates],
        "fwd": m.shift(-1) / m - 1,
        "clu": clusters(rets, dates),
    }
    p = sc
    for k, w in feats.items():
        w = w.copy()
        w.index = dates
        p = p.merge(qsb._long(w, k), on=["date", "sym"], how="left")
    spy_fwd = (m["SPY"].shift(-1) / m["SPY"] - 1).rename("spy_fwd")
    spy_fwd.index = dates
    p = p.merge(spy_fwd, left_on="date", right_index=True)
    return p.dropna(subset=["fwd", "spy_fwd", "beta", "clu"])


def z(s: pd.Series) -> pd.Series:
    s = s.clip(s.quantile(0.02), s.quantile(0.98))
    return (s - s.mean()) / s.std()


SIGNALS = {
    "Q": ["Q"],
    "QV": ["Q", "V"],
    "QM": ["Q", "mom"],
    "QVM": ["Q", "V", "mom"],
    "M": ["mom"],
    "REV": ["rev"],
    "LV": ["lowvol"],
    "Q+LV": ["Q", "lowvol"],
}


def signal_scores(p: pd.DataFrame) -> pd.DataFrame:
    """Per date: z of each raw input, composite, and a cluster-demeaned copy."""
    out = []
    for _, g in p.groupby("date"):
        g = g.copy()
        for col in ("Q", "V", "mom", "rev", "lowvol"):
            g["z_" + col] = z(g[col])
        for name, cols in SIGNALS.items():
            s = g[["z_" + c for c in cols]].mean(axis=1, skipna=False)
            g["raw_" + name] = s
            g["rel_" + name] = s - s.groupby(g["clu"]).transform("mean")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def simulate(sp: pd.DataFrame, sig: str, neut: str, short: str, frac: float, wt: str):
    col = f"{neut}_{sig}"
    prev_l: set = set()
    prev_s: set = set()
    recs = []
    for d, g in sp.groupby("date"):
        g = g.dropna(subset=[col])
        n = max(int(len(g) * frac), 5)
        if len(g) < 4 * n:
            continue
        top = g.nlargest(n, col)
        r_l, b_l = top["fwd"].mean(), top["beta"].mean()
        longs = set(top["sym"])
        if short == "spy":
            r_s, b_s, shorts, borrow = g["spy_fwd"].iloc[0], 1.0, {"SPY"}, BORROW_SPY
        else:
            pool = g if short == "sym" else g[g["mom"] < 0]
            bot = pool.nsmallest(n, col)
            if len(bot) < 5:
                continue
            r_s, b_s, shorts, borrow = (
                bot["fwd"].mean(),
                bot["beta"].mean(),
                set(bot["sym"]),
                BORROW_SINGLE,
            )
        w_s = 1.0 if wt == "dollar" else float(np.clip(b_l / b_s, 0.5, 2.0)) if b_s > 0 else 1.0
        turn_l = len(longs ^ prev_l) / max(len(longs), 1)
        turn_s = 0.0 if short == "spy" else len(shorts ^ prev_s) / max(len(shorts), 1)
        cost = (turn_l + turn_s * w_s) * COST_BP / 1e4 + w_s * borrow / 12
        recs.append((d, r_l - w_s * r_s - cost, g["spy_fwd"].iloc[0]))
        prev_l, prev_s = longs, shorts
    return pd.DataFrame(recs, columns=["date", "ret", "spy"]).set_index("date")


def stats(r: pd.Series) -> dict:
    if len(r) < 12:
        return {"sharpe": np.nan, "ann": np.nan, "dd": np.nan}
    eq = (1 + r).cumprod()
    ann = eq.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    return {"sharpe": ann / vol if vol else np.nan, "ann": ann, "dd": (eq / eq.cummax() - 1).min()}


def main() -> None:
    p = panel()
    sp = signal_scores(p)
    print(
        f"panel: {sp['date'].nunique()} months, median {sp.groupby('date').size().median():.0f} names"
    )
    grid = list(
        itertools.product(
            SIGNALS, ["raw", "rel"], ["sym", "falling", "spy"], [0.1, 0.2], ["dollar", "beta"]
        )
    )
    rows, series = [], {}
    for sig, neut, short, frac, wt in grid:
        m = simulate(sp, sig, neut, short, frac, wt)
        if m.empty:
            continue
        key = f"{sig}|{neut}|{short}|{frac}|{wt}"
        series[key] = m["ret"]
        tr, te = m[m.index <= TRAIN_END]["ret"], m[m.index > TRAIN_END]["ret"]
        st_tr, st_te, st_all = stats(tr), stats(te), stats(m["ret"])
        rows.append(
            {
                "combo": key,
                "train_sharpe": st_tr["sharpe"],
                "train_ann": st_tr["ann"],
                "test_sharpe": st_te["sharpe"],
                "test_ann": st_te["ann"],
                "test_dd": st_te["dd"],
                "all_ann": st_all["ann"],
                "all_dd": st_all["dd"],
                "corr_spy": m["ret"].corr(m["spy"]),
            }
        )
    res = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False)
    n_trials = len(res)
    rho = spearmanr(res["train_sharpe"], res["test_sharpe"], nan_policy="omit").statistic
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(
        f"\n{n_trials} combos. Rank corr train->test Sharpe: {rho:.2f} "
        "(~0 = in-sample ranking predicts nothing)"
    )
    print(f"share of combos with test Sharpe > 0: {(res['test_sharpe'] > 0).mean():.0%}")
    print("\nTOP 12 BY TRAIN (2018-22) — and what they did on unseen 2023-26:")
    print(res.head(12).round(3).to_string(index=False))
    print("\nTOP 8 BY TEST (hindsight — NOT a valid selection, shown for context):")
    print(res.sort_values("test_sharpe", ascending=False).head(8).round(3).to_string(index=False))

    from trading.selection.scores import deflated_sharpe

    best = res.iloc[0]["combo"]
    r = series[best]
    sr_m = r.mean() / r.std()
    sr_std = float(np.nanstd([series[k].mean() / series[k].std() for k in series]))
    dsr = deflated_sharpe(
        sr_m, len(r), float(skew(r)), float(kurtosis(r, fisher=False)), n_trials, sr_std
    )
    print(
        f"\nTrain-selected winner {best}: full-period deflated Sharpe prob = {dsr:.2f} "
        "(>0.95 = real edge after charging for the search)"
    )
    out = ROOT / "output/quality_short"
    res.to_csv(out / "ls_search.csv", index=False)
    pd.DataFrame(series).to_csv(out / "ls_search_curves.csv")
    print(f"wrote {out}/ls_search.csv")


if __name__ == "__main__":
    main()
