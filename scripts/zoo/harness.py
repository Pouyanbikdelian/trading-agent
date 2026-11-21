"""Strategy zoo harness — one evaluator for every strategy family.

Yan (2026-11-20): "go through all possible algos... find the best trading,
1. in each regime, 2. overall, 3. realistically better than a coin flip."

Rules every strategy obeys, enforced HERE, not by each strategy:
* A strategy returns TARGET WEIGHTS decided at the close of day t
  (dates x assets). The harness applies them to day t+1's return
  (``w.shift(1)``) — no strategy can trade on the close it just saw.
* Costs: ``COST_BP`` per unit of turnover (one-way), plus borrow on shorts.
* Train 2018-2022, test 2023-2026-05. Selection may only look at train.
* Regime of a day = SPY trend (vs 200d MA) x SPY 21d vol (vs expanding
  median), both measured at t-1, so the label is knowable when trading.
* Data: data/parquet cache, today's S&P 500 + Nasdaq-100 (survivorship-
  biased toward long stock books) + 11 ETFs. Stated, not fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/zoo"
TRAIN_END = pd.Timestamp("2022-12-31")
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC"]
COST_BP_STOCK = 10.0
COST_BP_ETF = 3.0
BORROW = 0.02
# Below the 200d MA the tape is almost never calm (16 of 1,830 days), so the
# bear regime is not split by volatility.
REGIMES = ["bull/calm", "bull/stressed", "bear"]


def universe() -> list[str]:
    u = yaml.safe_load((ROOT / "config/universes.generated.yaml").read_text())["universes"]
    return sorted(set(u["sp500"]["symbols"]) | set(u["nasdaq100"]["symbols"]))


_CACHE: dict[str, pd.DataFrame] = {}


def prices(field: str = "adj_close") -> pd.DataFrame:
    """Daily panel, stocks + ETFs, tz-naive index."""
    if field in _CACHE:
        return _CACHE[field]
    cols = {}
    for sym in [*universe(), *ETFS]:
        for ac in ("equity", "etf"):
            p = ROOT / f"data/parquet/{ac}/{sym}/1d.parquet"
            if p.exists():
                d = pd.read_parquet(p)
                if field == "adj_open":
                    cols[sym] = d["open"] * d["adj_close"] / d["close"]
                else:
                    cols[sym] = d[field]
                break
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated()]
    _CACHE[field] = px
    return px


def returns() -> pd.DataFrame:
    if "ret" not in _CACHE:
        _CACHE["ret"] = prices().pct_change(fill_method=None)
    return _CACHE["ret"]


def regimes() -> pd.Series:
    """Label per day, known at t-1: e.g. 'up/calm', 'down/stressed'."""
    spy = prices()["SPY"]
    r = spy.pct_change()
    trend = np.where(spy > spy.rolling(200).mean(), "up", "down")
    rv = r.rolling(21).std()
    stress = np.where(rv > rv.expanding(252).median(), "stressed", "calm")
    lab = pd.Series(
        [("bear" if a == "down" else f"bull/{b}") for a, b in zip(trend, stress)],
        index=spy.index,
    )
    lab[spy.rolling(200).mean().isna() | rv.expanding(252).median().isna()] = np.nan
    return lab.shift(1)


@dataclass
class Result:
    name: str
    family: str
    ret: pd.Series  # daily net returns


def evaluate(name: str, family: str, w: pd.DataFrame, cost_bp: float | None = None) -> Result:
    """Weights decided at close t -> earn day t+1. Costs on turnover."""
    r = returns().reindex(columns=w.columns)
    w = w.reindex(r.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * r.fillna(0.0)).sum(axis=1)
    if cost_bp is None:
        is_etf = pd.Series({c: c in ETFS for c in w.columns})
        bp = is_etf.map({True: COST_BP_ETF, False: COST_BP_STOCK})
        turn = (w.diff().abs() * bp).sum(axis=1) / 1e4
    else:
        turn = w.diff().abs().sum(axis=1) * cost_bp / 1e4
    borrow = w.clip(upper=0).abs().sum(axis=1) * BORROW / 252
    net = (gross - turn.shift(1).fillna(0) - borrow.shift(1).fillna(0)).loc["2018-03-01":]
    # start once the strategy has actually held something
    first = held.abs().sum(axis=1).loc[net.index]
    first = first[first > 0]
    if not first.empty:
        net = net.loc[first.index[0] :]
    return Result(name, family, net.rename(name))


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else np.nan


def stats(res: Result) -> dict:
    r = res.ret.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    tr, te = r[r.index <= TRAIN_END], r[r.index > TRAIN_END]
    out = {
        "name": res.name, "family": res.family,
        "sharpe": sharpe(r), "train": sharpe(tr), "test": sharpe(te),
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
        "vol": r.std() * np.sqrt(252), "maxdd": (eq / eq.cummax() - 1).min(),
        "start": r.index[0].date(),
    }
    lab = regimes().reindex(r.index)
    for reg in REGIMES:
        out[reg] = sharpe(r[lab == reg])
    return out


def monthly_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def weekly_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    iso = idx.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).max().values)


def save(results: list[Result], tag: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([r.ret for r in results], axis=1).to_parquet(OUT / f"ret_{tag}.parquet")
    tab = pd.DataFrame([stats(r) for r in results])
    tab.to_csv(OUT / f"stats_{tag}.csv", index=False)
    return tab


def show(tab: pd.DataFrame) -> None:
    pd.set_option("display.width", 260, "display.max_columns", 30)
    cols = ["name", "sharpe", "train", "test", "cagr", "vol", "maxdd", *REGIMES]
    print(tab[cols].sort_values("sharpe", ascending=False).round(2).to_string(index=False))
