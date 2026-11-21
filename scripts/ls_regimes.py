"""When does long/short work? Regime-gated test of the ls_search combos.

Yan (2026-11-20): run the long/short sleeve only in the periods it works.
Hindsight will always find "the" period, so every gate here is a rule
computable at the rebalance date from past prices only, the gate is chosen
on 2018-2022, and judged on 2023-2026. When the gate is off the sleeve sits
in cash.

Gates (all known at month-end t):
  trend_up     SPY above its 200d MA
  trend_down   SPY below its 200d MA
  calm / stressed   SPY 21d realized vol below / above its expanding median
  drawdown     SPY > 10% below its 252d high
  disp_high / disp_low  cross-sectional dispersion of 1m stock returns above /
               below its expanding median (relative value needs dispersion)
  rebound      SPY 1m > +5% after a negative 12m (momentum-crash setup)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qsb", ROOT / "scripts/quality_short_backtest.py")
qsb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qsb)  # type: ignore[union-attr]
TRAIN_END = pd.Timestamp("2022-12-31")


def gates(dates: pd.DatetimeIndex) -> pd.DataFrame:
    px = qsb.load_prices()
    spy = px["SPY"]
    r = spy.pct_change()
    rv = r.rolling(21).std()
    m = px.loc[dates]
    m1 = m / m.shift(1) - 1
    disp = m1.drop(columns="SPY").std(axis=1)
    g = pd.DataFrame(index=dates)
    above = spy > spy.rolling(200).mean()
    g["trend_up"] = above.loc[dates].values
    g["trend_down"] = ~g["trend_up"]
    rvm = rv.loc[dates]
    g["stressed"] = (rvm > rvm.expanding(12).median()).values
    g["calm"] = ~g["stressed"]
    g["drawdown"] = (spy / spy.rolling(252).max() - 1).loc[dates].values < -0.10
    dmed = disp.expanding(12).median()
    g["disp_high"] = (disp > dmed).values
    g["disp_low"] = ~g["disp_high"]
    s12 = (m["SPY"] / m["SPY"].shift(12) - 1).values
    s1 = m1["SPY"].values
    g["rebound"] = (s1 > 0.05) & (s12 < 0)
    g["not_rebound"] = ~g["rebound"]
    g["always"] = True
    return g


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(12)) if x.std() > 0 else np.nan


def main() -> None:
    curves = pd.read_csv(
        ROOT / "output/quality_short/ls_search_curves.csv", index_col=0, parse_dates=True
    )
    g = gates(curves.index)
    rows = []
    for combo in curves.columns:
        r = curves[combo].dropna()
        gg = g.loc[r.index]
        for gate in g.columns:
            on = gg[gate].astype(bool)
            gated = r.where(on, 0.0)
            tr, te = gated[gated.index <= TRAIN_END], gated[gated.index > TRAIN_END]
            rows.append(
                {
                    "combo": combo,
                    "gate": gate,
                    "on%": on.mean(),
                    "train_sharpe": sharpe(tr),
                    "test_sharpe": sharpe(te),
                    "ann_all": (1 + gated).prod() ** (12 / len(gated)) - 1,
                    "ret_when_on/mo": r[on].mean(),
                    "ret_when_off/mo": r[~on].mean(),
                    "n_on": int(on.sum()),
                }
            )
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 20)

    # 1. Which gates help ACROSS combos? (robust = helps many combos, both halves)
    base = res[res.gate == "always"].set_index("combo")
    res["d_train"] = res["train_sharpe"] - res["combo"].map(base["train_sharpe"])
    res["d_test"] = res["test_sharpe"] - res["combo"].map(base["test_sharpe"])
    fam = res.assign(short=res.combo.str.split("|").str[2])
    print("Median Sharpe change from gating, by gate x short-leg type:")
    print(
        fam[fam.gate != "always"]
        .groupby(["short", "gate"])[["d_train", "d_test", "on%"]]
        .median()
        .round(2)
        .unstack(0)
        .to_string()
    )

    # 2. Per combo: pick the best gate on TRAIN, report TEST.
    pick = res.sort_values("train_sharpe", ascending=False).groupby("combo").head(1)
    pick = pick.merge(
        base[["train_sharpe", "test_sharpe"]],
        left_on="combo",
        right_index=True,
        suffixes=("", "_ungated"),
    )
    print("\nBest gate chosen on 2018-22, judged on 2023-26 (top 15 by train):")
    cols = [
        "combo",
        "gate",
        "on%",
        "train_sharpe",
        "test_sharpe",
        "train_sharpe_ungated",
        "test_sharpe_ungated",
        "ann_all",
    ]
    print(
        pick.sort_values("train_sharpe", ascending=False)[cols]
        .head(15)
        .round(2)
        .to_string(index=False)
    )
    better = (pick["test_sharpe"] > pick["test_sharpe_ungated"]).mean()
    print(f"\nTrain-chosen gate improved the unseen-period Sharpe for {better:.0%} of combos")

    # 3. Single-stock shorts specifically: conditional monthly returns.
    ss = res[res.combo.str.contains(r"\|sym\||\|falling\|") & (res.gate != "always")]
    print("\nSingle-stock-short combos — avg monthly return when gate ON vs OFF (all years):")
    print(
        ss.groupby("gate")[["ret_when_on/mo", "ret_when_off/mo", "n_on"]]
        .median()
        .round(4)
        .sort_values("ret_when_on/mo", ascending=False)
        .to_string()
    )
    res.to_csv(ROOT / "output/quality_short/ls_regimes.csv", index=False)


if __name__ == "__main__":
    main()
