"""Bounce screen: which S&P 500 / Nasdaq-100 names are most likely to bounce
over the next week, with probabilities measured on history.

Yan's thesis (2026-11-20): the market keeps falling, but there is a tradeable
relief bounce first. That is a short-horizon mean-reversion question, so the
test is: after an oversold signal on day t, buy the next OPEN, sell the close
5 sessions later. Count how often that made money and how often it beat SPY.

Honesty built in:
* Probabilities are DATE-CLUSTERED. Sell-offs make 80 names oversold on the
  same day; counting them as 80 independent trials overstates confidence.
  Each date counts once (mean over its names), then dates are averaged, and
  a bootstrap over dates gives the 90% interval.
* Split 2018-22 vs 2023+: an edge that only lived in the old half is dead
  (see quality_rebound_study: the dip-buy edge decayed after 2023).
* "stressed" regime = SPY below its 50d MA AND down >3% over 10 days — the
  tape Yan is describing. Probabilities are shown for it separately.
* Survivorship bias: universe = TODAY's index members, so names that crashed
  out are missing. Absolute hit rates are flattered; compare against the
  baseline row, which carries the same bias.

Usage:
  uv run python scripts/bounce_screen.py            # backtest on data/parquet cache
  uv run python scripts/bounce_screen.py --fetch    # refresh 10y via yfinance, then rank NOW
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
HOLD = 5  # sessions
STRESS_10D = -0.03


def universe() -> list[str]:
    u = yaml.safe_load((ROOT / "config/universes.generated.yaml").read_text())["universes"]
    syms = set(u["sp500"]["symbols"]) | set(u["nasdaq100"]["symbols"])
    return sorted(s.replace(".", "-") for s in syms)


def load_cache(syms: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for s in [*syms, "SPY"]:
        for ac in ("equity", "etf"):
            p = ROOT / f"data/parquet/{ac}/{s}/1d.parquet"
            if p.exists():
                out[s] = pd.read_parquet(p)
                break
    return out


def load_fetch(syms: list[str], period: str = "10y") -> dict[str, pd.DataFrame]:
    """Chunked, sequential yfinance download with retries.

    One threaded call for ~515 names failed 145 of them on 2026-11-20
    ("possibly delisted" = Yahoo throttling) and hit yfinance's sqlite tz
    cache ("unable to open database file"). Rows that do come back from a
    throttled bulk call cannot be trusted either, so: small chunks, no
    threads, a private cache dir, retry what is missing, and report it.
    """
    import tempfile
    import time

    import yfinance as yf

    yf.set_tz_cache_location(tempfile.mkdtemp(prefix="yf-cache-"))
    want = [*syms, "SPY"]
    out: dict[str, pd.DataFrame] = {}
    for attempt in range(3):
        missing = [s for s in want if s not in out]
        if not missing:
            break
        for i in range(0, len(missing), 40):
            chunk = missing[i : i + 40]
            try:
                raw = yf.download(
                    " ".join(chunk),
                    period=period,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
            except Exception as e:
                print(f"chunk failed ({e}); retrying later")
                time.sleep(5)
                continue
            for s in chunk:
                try:
                    d = raw[s] if len(chunk) > 1 else raw
                    d = d.dropna(how="all").rename(columns=str.lower)
                    d = d.rename(columns={"adj close": "adj_close"})
                    if len(d) > 260 and d["close"].iloc[-1] > 0:
                        out[s] = d[["open", "high", "low", "close", "volume", "adj_close"]]
                except Exception:
                    continue
            time.sleep(1.5)
        if attempt < 2 and any(s not in out for s in want):
            time.sleep(10)
    gone = [s for s in want if s not in out]
    print(f"fetched {len(out)}/{len(want)}; still missing {len(gone)}: {gone[:20]}")
    for s in ("SPY", "NVDA", "AAPL", "AMD"):
        if s in out:
            d = out[s]
            print(
                f"  sanity {s}: last {d.index[-1].date()} close {d['close'].iloc[-1]:.2f}"
                f" 52w-high {d['close'].iloc[-252:].max():.2f}"
            )
    if "SPY" not in out:
        raise SystemExit("SPY missing — cannot judge the regime; rerun later")
    return out


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = data["SPY"]["adj_close"]
    spy_r5 = spy / spy.shift(5) - 1
    spy_r10 = spy / spy.shift(10) - 1
    spy_stress = (spy < spy.rolling(50).mean()) & (spy_r10 < STRESS_10D)
    spy_fwd = spy.shift(-HOLD) / spy.shift(-1) - 1  # approx: next close -> +5
    rows = []
    for sym, d in data.items():
        if sym == "SPY" or len(d) < 260:
            continue
        adj = d["adj_close"]
        f = d["adj_close"] / d["close"]
        o, h, lo = d["open"] * f, d["high"] * f, d["low"] * f
        x = pd.DataFrame(index=d.index)
        x["sym"] = sym
        x["rsi2"] = rsi(adj, 2)
        x["r5"] = adj / adj.shift(5) - 1
        x["r5_rel"] = x["r5"] - spy_r5.reindex(d.index)
        x["ibs"] = ((adj - lo) / (h - lo).replace(0, np.nan)).clip(0, 1)
        x["above200"] = adj > adj.rolling(200).mean()
        x["dd52"] = adj / adj.rolling(252).max() - 1
        x["z20"] = (adj - adj.rolling(20).mean()) / adj.rolling(20).std()
        x["volx"] = d["volume"] / d["volume"].rolling(20).mean()
        x["stress"] = spy_stress.reindex(d.index).fillna(False)
        x["fwd"] = adj.shift(-HOLD) / o.shift(-1) - 1  # buy next open, sell close t+5
        x["fwd_spy"] = spy_fwd.reindex(d.index)
        rows.append(x)
    df = pd.concat(rows).reset_index(names="date")
    df["r5_rank"] = df.groupby("date")["r5_rel"].rank(pct=True)
    return df


COMBOS: dict[str, str] = {
    "baseline (any stock, any day)": "rsi2 >= 0",
    "RSI2<5": "rsi2 < 5",
    "RSI2<5 & uptrend(>200d)": "rsi2 < 5 & above200",
    "RSI2<5 & IBS<0.2": "rsi2 < 5 & ibs < 0.2",
    "RSI2<5 & IBS<0.2 & uptrend": "rsi2 < 5 & ibs < 0.2 & above200",
    "RSI2<5 & volume spike>2x": "rsi2 < 5 & volx > 2",
    "worst 5% vs SPY (5d)": "r5_rank <= 0.05",
    "worst 5% vs SPY & uptrend": "r5_rank <= 0.05 & above200",
    "5d drop <= -10%": "r5 <= -0.10",
    "5d drop <= -10% & uptrend": "r5 <= -0.10 & above200",
    "z20<-2.5 & uptrend": "z20 < -2.5 & above200",
    "washed out (-30% off high) & RSI2<5": "dd52 < -0.30 & rsi2 < 5",
    "falling knife: RSI2<5 & downtrend": "rsi2 < 5 & ~above200",
    "capitulation: -10% 5d & vol>2x & uptrend": "r5 <= -0.10 & volx > 2 & above200",
}


def clustered(sub: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    sub = sub.assign(
        up=sub["fwd"] > 0, beat=sub["fwd"] > sub["fwd_spy"], xs=sub["fwd"] - sub["fwd_spy"]
    )
    by = sub.groupby("date")[["up", "beat", "fwd", "xs"]].mean()
    n = len(by)
    if n < 15:
        return {"dates": n}
    boot = by["up"].values[rng.integers(0, n, (400, n))].mean(axis=1)
    return {
        "signals": len(sub),
        "dates": n,
        "P(up)": by["up"].mean(),
        "P_lo": np.quantile(boot, 0.05),
        "P_hi": np.quantile(boot, 0.95),
        "P(beat SPY)": by["beat"].mean(),
        "avg 5d %": by["fwd"].mean() * 100,
        "avg vs SPY %": by["xs"].mean() * 100,
    }


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = df.dropna(subset=["fwd", "fwd_spy", "rsi2", "dd52"])
    yr = pd.to_datetime(df["date"]).dt.year
    out = []
    for name, q in COMBOS.items():
        m = df.query(q)
        for regime, mm in (("all", m), ("stressed", m[m["stress"]])):
            r = clustered(mm, rng)
            if "P(up)" not in r:
                continue
            early = clustered(mm[yr.loc[mm.index] <= 2022], rng).get("P(up)", np.nan)
            late = clustered(mm[yr.loc[mm.index] >= 2023], rng).get("P(up)", np.nan)
            out.append(
                {"combo": name, "regime": regime, **r, "P(up) 18-22": early, "P(up) 23+": late}
            )
    return pd.DataFrame(out)


def rank_now(df: pd.DataFrame, table: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    last = df["date"].max()
    now = df[df["date"] == last].copy()
    stressed = bool(now["stress"].iloc[0])
    regime = "stressed" if stressed else "all"
    t = table[(table["regime"] == regime) & (~table["combo"].str.startswith("baseline"))]
    hits = []
    for _, row in t.iterrows():
        for sym in now.query(COMBOS[row["combo"]])["sym"]:
            hits.append(
                {
                    "sym": sym,
                    "combo": row["combo"],
                    "P(up)": row["P(up)"],
                    "P(beat SPY)": row["P(beat SPY)"],
                    "avg 5d %": row["avg 5d %"],
                    "P(up) 23+": row["P(up) 23+"],
                }
            )
    if not hits:
        return pd.DataFrame()
    h = pd.DataFrame(hits)
    best = h.sort_values("P(up)", ascending=False).groupby("sym").head(1)
    best["n_combos"] = best["sym"].map(h.groupby("sym").size())
    best = best.merge(now[["sym", "r5", "rsi2", "ibs", "dd52", "above200"]], on="sym")
    print(f"\nAs of {pd.Timestamp(last).date()} — market regime: {regime}")
    return best.sort_values(["P(up)", "n_combos"], ascending=False).head(top)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download fresh 10y data (yfinance)")
    ap.add_argument("--out", default=str(ROOT / "output/bounce_screen"))
    a = ap.parse_args()
    syms = universe()
    data = load_fetch(syms) if a.fetch else load_cache(syms)
    print(f"loaded {len(data)} series")
    df = features(data)
    table = backtest(df)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(table.round(3).to_string(index=False))
    picks = rank_now(df, table)
    print(picks.round(3).to_string(index=False))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "combos.csv", index=False)
    picks.to_csv(out / "picks_now.csv", index=False)
    print(f"\nwrote {out}/combos.csv and picks_now.csv")


if __name__ == "__main__":
    main()
