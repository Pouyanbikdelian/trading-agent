"""Step 2 — trend following (ETFs). Classic, published rules only; no tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness import ETFS, evaluate, monthly_dates, prices, returns, save, show


def vol_target(w: pd.DataFrame, target: float = 0.10, lb: int = 60) -> pd.DataFrame:
    """Scale the whole book so its trailing realized vol ~= target (cap 2x)."""
    r = returns()[w.columns].fillna(0)
    port = (w.shift(1).fillna(0) * r).sum(axis=1)
    rv = port.rolling(lb).std() * np.sqrt(252)
    k = (target / rv).clip(upper=2.0).fillna(0)
    return w.mul(k, axis=0)


def main() -> None:
    px = prices()[ETFS]
    r = px.pct_change()
    idx = px.index
    m = monthly_dates(idx)
    res = []

    # 1. SPY 200d MA timing: in SPY above, else IEF / else cash
    above = px["SPY"] > px["SPY"].rolling(200).mean()
    res.append(
        evaluate(
            "SPY>200d else IEF", "trend", pd.DataFrame({"SPY": above * 1.0, "IEF": (~above) * 1.0})
        )
    )
    res.append(evaluate("SPY>200d else cash", "trend", pd.DataFrame({"SPY": above * 1.0})))
    # 2. Faber 10-month SMA, monthly, on SPY
    sma10 = px["SPY"].loc[m].rolling(10).mean()
    sig = (px["SPY"].loc[m] > sma10).astype(float)
    res.append(evaluate("SPY 10m SMA (Faber)", "trend", pd.DataFrame({"SPY": sig})))
    # 3. EMA 50/200 cross on SPY and QQQ
    for t in ["SPY", "QQQ"]:
        e = (px[t].ewm(span=50).mean() > px[t].ewm(span=200).mean()).astype(float)
        res.append(evaluate(f"{t} EMA50/200", "trend", pd.DataFrame({t: e})))
    # 4. GTAA-5 (Faber): 5 asset classes, each 20% if above its 10m SMA
    g5 = ["SPY", "EFA", "IEF", "DBC", "GLD"]
    pm = px.loc[m, g5]
    w = (pm > pm.rolling(10).mean()).astype(float) * 0.2
    res.append(evaluate("GTAA-5 (Faber)", "trend", w))
    # 5. Dual momentum GEM: best of SPY/EFA on 12m; if it loses to IEF -> IEF
    r12 = pm.reindex(columns=["SPY", "EFA", "IEF"]).pipe(lambda x: x)
    r12 = px.loc[m, ["SPY", "EFA", "IEF"]] / px.loc[m, ["SPY", "EFA", "IEF"]].shift(12) - 1
    w = pd.DataFrame(0.0, index=m, columns=["SPY", "EFA", "IEF"])
    for d, row in r12.dropna().iterrows():
        best = "SPY" if row["SPY"] >= row["EFA"] else "EFA"
        w.loc[d, best if row[best] > row["IEF"] else "IEF"] = 1.0
    res.append(evaluate("Dual momentum (GEM)", "trend", w))
    # 6. TSMOM 12m on all 11 ETFs, inverse-vol, long/flat and long/short, 10% vol target
    r12 = px.loc[m] / px.loc[m].shift(12) - 1
    iv = 1 / (r.rolling(60).std().loc[m])
    for name, sgn in [("long/flat", r12.gt(0).astype(float)), ("long/short", np.sign(r12))]:
        w = (sgn * iv).div(iv.where(r12.notna()).sum(axis=1), axis=0)
        w = w.reindex(idx).ffill()
        res.append(evaluate(f"TSMOM-12m ETFs {name}, 10% vol", "trend", vol_target(w)))
    # 7. Donchian 100d breakout per ETF, long/flat, equal risk
    hi = px.rolling(100).max().shift(1)
    lo = px.rolling(50).min().shift(1)
    state = pd.DataFrame(np.nan, index=idx, columns=ETFS)
    state[px >= hi] = 1.0
    state[px <= lo] = 0.0
    state = state.ffill().fillna(0)
    ivd = 1 / r.rolling(60).std()
    w = (state * ivd).div((ivd.where(state.notna())).sum(axis=1), axis=0)
    res.append(evaluate("Donchian 100/50 ETFs, 10% vol", "trend", vol_target(w)))

    tab = save(res, "s02_trend")
    show(tab)


if __name__ == "__main__":
    main()
