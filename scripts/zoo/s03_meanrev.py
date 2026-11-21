"""Step 3 — mean reversion: index swing rules + cross-sectional stock reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness import ETFS, evaluate, prices, returns, save, show, universe, weekly_dates


def rsi(s: pd.Series | pd.DataFrame, n: int):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def hold_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    st = pd.Series(np.nan, index=entry.index)
    st[exit_] = 0.0
    st[entry] = 1.0
    return st.ffill().fillna(0.0)


def main() -> None:
    px = prices()
    hi, lo, cl = prices("high"), prices("low"), prices("close")
    r = returns()
    idx = px.index
    res = []
    # --- index swing rules (Connors-style), long only
    for t in ["SPY", "QQQ", "IWM"]:
        p = px[t]
        up = p > p.rolling(200).mean()
        r2 = rsi(p, 2)
        st = hold_state((r2 < 10) & up, (r2 > 70) | (p > p.rolling(5).mean()))
        res.append(evaluate(f"{t} RSI2<10 in uptrend", "meanrev", pd.DataFrame({t: st})))
        ibs = (cl[t] - lo[t]) / (hi[t] - lo[t]).replace(0, np.nan)
        st = hold_state(ibs < 0.2, ibs > 0.8)
        res.append(evaluate(f"{t} IBS<0.2", "meanrev", pd.DataFrame({t: st})))
        z = (p - p.rolling(20).mean()) / p.rolling(20).std()
        st = hold_state(z < -2, z > 0)
        res.append(evaluate(f"{t} Bollinger z<-2", "meanrev", pd.DataFrame({t: st})))

    # --- cross-sectional stock reversal (weekly)
    stocks = [s for s in universe() if s in px.columns]
    wk = weekly_dates(idx)
    r5 = (px[stocks] / px[stocks].shift(5) - 1).loc[wk]
    above = (px[stocks] > px[stocks].rolling(200).mean()).loc[wk]
    n = 50
    def book(mask_long, mask_short=None, hedge=False):
        w = pd.DataFrame(0.0, index=wk, columns=[*stocks, "SPY"])
        for d in wk:
            row = r5.loc[d].dropna()
            if len(row) < 200:
                continue
            L = row[mask_long.loc[d].reindex(row.index).fillna(False)] if mask_long is not None else row
            L = L.nsmallest(n).index
            if len(L) == 0:
                continue
            w.loc[d, L] = 1.0 / len(L)
            if mask_short is not None:
                S = row.nlargest(n).index
                w.loc[d, S] = -1.0 / len(S)
            if hedge:
                w.loc[d, "SPY"] = -1.0
        return w
    allm = above.astype(bool) | True
    res.append(evaluate("Weekly reversal: 50 losers, long", "meanrev", book(allm)))
    res.append(evaluate("Weekly reversal: 50 losers, long, SPY hedge", "meanrev", book(allm, hedge=True)))
    res.append(evaluate("Weekly reversal: losers-winners L/S", "meanrev", book(allm, mask_short=True)))
    res.append(evaluate("Weekly reversal: 50 losers in uptrend", "meanrev", book(above)))
    res.append(evaluate("Weekly reversal: losers in uptrend, SPY hedge", "meanrev", book(above, hedge=True)))

    # --- the bounce screen as a portfolio: 5d drop <= -10% & above 200d, hold 5 days
    r5d = px[stocks] / px[stocks].shift(5) - 1
    up200 = px[stocks] > px[stocks].rolling(200).mean()
    sig = ((r5d <= -0.10) & up200).astype(float)
    held = sig.rolling(5, min_periods=1).max()  # stays held 5 sessions after signal
    w = held.div(held.sum(axis=1).clip(lower=10), axis=0)  # cap 10% per name
    res.append(evaluate("Bounce screen: -10% 5d & >200d, hold 5d", "meanrev", w))

    tab = save(res, "s03_meanrev")
    show(tab)


if __name__ == "__main__":
    main()
