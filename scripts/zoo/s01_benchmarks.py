"""Step 1 — benchmarks. Every later strategy has to beat these."""

from __future__ import annotations

import pandas as pd

from harness import evaluate, monthly_dates, prices, save, show, universe


def main() -> None:
    px = prices()
    idx = px.index
    res = []
    w = pd.DataFrame(0.0, index=idx, columns=["SPY"]); w["SPY"] = 1.0
    res.append(evaluate("SPY buy&hold", "benchmark", w))
    w = pd.DataFrame(0.0, index=idx, columns=["QQQ"]); w["QQQ"] = 1.0
    res.append(evaluate("QQQ buy&hold", "benchmark", w))
    w = pd.DataFrame({"SPY": 0.6, "IEF": 0.4}, index=idx)
    res.append(evaluate("60/40 SPY/IEF", "benchmark", w))
    stocks = [s for s in universe() if s in px.columns]
    m = monthly_dates(idx)
    avail = px.loc[m, stocks].notna()
    w = avail.div(avail.sum(axis=1), axis=0)
    res.append(evaluate("Equal-weight stocks (monthly)", "benchmark", w))
    tab = save(res, "s01_benchmarks")
    show(tab)


if __name__ == "__main__":
    main()
