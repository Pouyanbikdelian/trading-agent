"""Step 7 — machine learning, strictly walk-forward.

Stock selection: features per stock-month (momentum 12-1/6-1, 1m reversal,
60d vol, beta, distance to 52w high, point-in-time quality & value). Target:
next-month return minus the cross-sectional mean. Models retrained every
January on ALL data available before that month (expanding window, first
fit needs 24 months). Long the top 25/50 predicted; also beta-hedged.

Market timing: logistic model on SPY trend/vol/drawdown/breadth features
predicting next-month SPY up/down; hold SPY when P(up) > 0.5, else IEF.
"""

# ruff: noqa: N806
from __future__ import annotations

import importlib.util
import warnings

import pandas as pd
from harness import ROOT, evaluate, monthly_dates, prices, returns, save, show, universe
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("qsb", ROOT / "scripts/quality_short_backtest.py")
qsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qsb)  # type: ignore[union-attr]
FEATS = ["mom12", "mom6", "rev1", "vol60", "beta", "d52", "Q", "V"]


def build_panel() -> pd.DataFrame:
    px = prices()
    r = returns()
    stocks = [s for s in universe() if s in px.columns]
    m = monthly_dates(px.index)
    P = px[stocks]
    pm = P.loc[m]
    spy = r["SPY"]
    f = {
        "mom12": pm.shift(1) / pm.shift(12) - 1,
        "mom6": pm.shift(1) / pm.shift(6) - 1,
        "rev1": pm / pm.shift(1) - 1,
        "vol60": r[stocks].rolling(60, min_periods=40).std().loc[m],
        "beta": r[stocks]
        .rolling(252, min_periods=200)
        .cov(spy)
        .div(spy.rolling(252).var(), axis=0)
        .loc[m],
        "d52": (P / P.rolling(252).max()).loc[m],
        "fwd": pm.shift(-1) / pm - 1,
    }
    fund = pd.read_parquet(ROOT / "output/quality_short/fundamentals_pit.parquet")
    sc = qsb.score(fund, pm, pm, exfin=False)
    f["Q"] = sc.pivot(index="date", columns="sym", values="Q").reindex(index=m, columns=stocks)
    f["V"] = sc.pivot(index="date", columns="sym", values="V").reindex(index=m, columns=stocks)
    long = None
    for k, w in f.items():
        s = w.rename_axis(index="date", columns="sym").stack(future_stack=True).rename(k)
        long = s.to_frame() if long is None else long.join(s, how="outer")
    long = long.reset_index()
    # cross-sectional rank-normalise features each month (robust, scale-free)
    for c in FEATS:
        long[c] = long.groupby("date")[c].rank(pct=True) - 0.5
    long["y"] = long["fwd"] - long.groupby("date")["fwd"].transform("mean")
    return long


def walk_forward(panel: pd.DataFrame, model_fn) -> pd.DataFrame:
    dates = sorted(panel["date"].unique())
    preds = []
    model = None
    for i, d in enumerate(dates):
        if i < 24:
            continue
        d = pd.Timestamp(d)
        if model is None or d.month == 1:
            # train on months whose forward return is fully known before d
            tr = panel[(panel["date"] < dates[i - 1])].dropna(subset=["y"])
            Xtr = tr[FEATS].fillna(0).values
            model = model_fn()
            model.fit(Xtr, tr["y"].values)
        cur = panel[panel["date"] == d]
        p = model.predict(cur[FEATS].fillna(0).values)
        preds.append(pd.DataFrame({"date": d, "sym": cur["sym"].values, "pred": p}))
    return pd.concat(preds)


def to_weights(pred: pd.DataFrame, n: int, beta: pd.DataFrame | None) -> pd.DataFrame:
    piv = pred.pivot(index="date", columns="sym", values="pred")
    w = pd.DataFrame(0.0, index=piv.index, columns=[*piv.columns, "SPY"])
    for d, row in piv.iterrows():
        top = row.dropna().nlargest(n).index
        w.loc[d, top] = 1.0 / n
        if beta is not None:
            w.loc[d, "SPY"] = -float(beta.loc[d, top].mean())
    return w


def main() -> None:
    panel = build_panel()
    px = prices()
    r = returns()
    stocks = [s for s in universe() if s in px.columns]
    m = monthly_dates(px.index)
    beta = (
        r[stocks]
        .rolling(252, min_periods=200)
        .cov(r["SPY"])
        .div(r["SPY"].rolling(252).var(), axis=0)
        .loc[m]
    )
    models = {
        "Ridge": lambda: Ridge(alpha=10.0),
        "GBM": lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200
        ),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=150, max_depth=5, min_samples_leaf=200, n_jobs=-1, random_state=0
        ),
    }
    res = []
    for name, fn in models.items():
        pred = walk_forward(panel, fn)
        for n in [25, 50]:
            res.append(evaluate(f"ML {name} top{n} long", "ml", to_weights(pred, n, None)))
        res.append(evaluate(f"ML {name} top50 beta-hedged", "ml", to_weights(pred, 50, beta)))
        print(f"done {name}", flush=True)

    # market timing
    spy = px["SPY"]
    rs = r["SPY"]
    stocks_px = px[stocks]
    breadth = (stocks_px > stocks_px.rolling(200).mean()).mean(axis=1)
    X = pd.DataFrame(
        {
            "trend200": spy / spy.rolling(200).mean() - 1,
            "trend50": spy / spy.rolling(50).mean() - 1,
            "mom1": spy / spy.shift(21) - 1,
            "mom12": spy / spy.shift(252) - 1,
            "vol21": rs.rolling(21).std(),
            "dd": spy / spy.rolling(252).max() - 1,
            "breadth": breadth,
        }
    ).loc[m]
    y = (spy.loc[m].shift(-1) / spy.loc[m] - 1 > 0).astype(float)
    w = pd.DataFrame(0.0, index=m, columns=["SPY", "IEF"])
    for i, d in enumerate(m):
        if i < 24:
            continue
        tr = X.iloc[: i - 1].dropna()
        ytr = y.loc[tr.index]
        if ytr.nunique() < 2:
            continue
        mu, sd = tr.mean(), tr.std()
        clf = LogisticRegression(C=0.5).fit((tr - mu) / sd, ytr)
        xd = X.loc[[d]]
        if xd.isna().any(axis=None):
            continue
        p = clf.predict_proba((xd - mu) / sd)[0, 1]
        w.loc[d, "SPY" if p > 0.5 else "IEF"] = 1.0
    res.append(evaluate("ML logistic SPY timing (else IEF)", "ml", w))
    tab = save(res, "s07_ml")
    show(tab)


if __name__ == "__main__":
    main()
