"""Step 10 — backtest the EXACT live code (strategies/regime_switch.py) on the
desk's real cadence (every other Friday), vs weekly/monthly/daily and SPY."""

from __future__ import annotations

import sys

import pandas as pd

from harness import OUT, TRAIN_END, evaluate, prices, regimes, sharpe, stats

sys.path.insert(0, str(OUT.parents[1] / "src"))
from trading.strategies.regime_switch import ALL_SYMBOLS, RegimeSwitch  # noqa: E402


def main() -> None:
    px = prices()[list(ALL_SYMBOLS)]
    w = RegimeSwitch().generate(px)
    idx = w.index
    fridays = idx[idx.weekday == 4]
    cadences = {
        "daily": idx,
        "weekly (Fri)": fridays,
        "every other Fri (desk)": fridays[::2],
        "monthly": idx[pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("max").values == idx],
    }
    res = []
    for name, dates in cadences.items():
        ww = w.loc[w.index.isin(dates)].reindex(idx).ffill().fillna(0.0)
        res.append(evaluate(f"LIVE regime_switch — {name}", "live", ww))
    spy = pd.DataFrame({"SPY": 1.0}, index=idx)
    res.append(evaluate("SPY buy&hold", "benchmark", spy))
    rows = []
    for r in res:
        s = stats(r)
        x = r.ret.loc["2019-03-01":]
        s.update({"sharpe": sharpe(x), "train": sharpe(x[x.index <= TRAIN_END]),
                  "test": sharpe(x[x.index > TRAIN_END])})
        rows.append(s)
    pd.set_option("display.width", 220)
    t = pd.DataFrame(rows)
    print(t[["name", "sharpe", "train", "test", "cagr", "vol", "maxdd", "bull/calm",
             "bull/stressed", "bear"]].round(2).to_string(index=False))
    desk = res[2].ret.loc["2019-03-01":]
    s = res[-1].ret.reindex(desk.index)
    print("\nYear by year (desk cadence vs SPY):")
    print(pd.DataFrame({"sleeve": desk, "SPY": s}).groupby(desk.index.year)
          .apply(lambda x: (1 + x).prod() - 1).round(3).to_string())
    lab = regimes().reindex(w.index)
    print("\nAvg allocation by regime (desk cadence):")
    ww = w.loc[w.index.isin(fridays[::2])]
    print(ww.groupby(lab.reindex(ww.index)).mean().round(2).loc[:, lambda d: d.max() > 0.01]
          .to_string())


if __name__ == "__main__":
    main()
