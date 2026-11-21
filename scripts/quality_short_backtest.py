"""Long quality / short junk-and-expensive: does it make money, point-in-time?

Yan (2026-11-20): put ~25% of the tradable book into a tactical sleeve that
buys high-quality names and shorts bad, overvalued ones. Before any agent
gets a short button, this has to earn it on information that was actually
public on each rebalance date.

Point-in-time discipline
------------------------
Fundamentals come from SEC EDGAR XBRL ``companyfacts``. Every fact carries
the date it was FILED, and a number only enters the model after it was
filed (+2 days, for the 10-Q to actually be read). No yfinance ``info``
(that is today's snapshot — pure look-ahead). TTM flows are built the
standard way: last 10-K + (current YTD - prior-year YTD).

Two stages
----------
  # 1. On the Mac (needs internet; SEC requires a contact in the UA):
  export SEC_USER_AGENT="Your Name your@email"
  uv run python scripts/quality_short_backtest.py fetch
  # 2. Anywhere with the repo:
  uv run python scripts/quality_short_backtest.py run

Known biases, stated up front
-----------------------------
* Universe = TODAY's S&P 500 + Nasdaq-100. Junk that went bankrupt or was
  dropped is missing — exactly what a short book would have profited from.
  So the SHORT side is biased AGAINST itself and the LONG side is flattered.
* Prices: data/parquet cache (2018 → May 2026), so ~2019-2026 is testable:
  COVID crash, 2021 junk rally, 2022 bear, 2023-25 AI melt-up.
* Costs: 10 bp per side traded + short borrow (default 2%/yr, general
  collateral is ~0.3% but junk is often hard-to-borrow). No short rebate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EDGAR_DIR = ROOT / "data/edgar"
FILING_LAG_DAYS = 2

# concept -> candidate us-gaap tags, first one found wins per filing
FLOWS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
STOCKS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}
SHARES = ["EntityCommonStockSharesOutstanding"]  # dei, cover page, instant


# ----------------------------------------------------------------- fetch
def universe() -> list[str]:
    u = yaml.safe_load((ROOT / "config/universes.generated.yaml").read_text())["universes"]
    return sorted(set(u["sp500"]["symbols"]) | set(u["nasdaq100"]["symbols"]))


def fetch() -> None:
    import httpx

    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise SystemExit('set SEC_USER_AGENT="Your Name you@email" (SEC requires a contact)')
    h = {"User-Agent": ua, "Accept-Encoding": "gzip"}
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=h, timeout=30) as c:
        tick = c.get("https://www.sec.gov/files/company_tickers.json").json()
        cik = {v["ticker"].upper(): int(v["cik_str"]) for v in tick.values()}
        todo = universe()
        for i, sym in enumerate(todo):
            out = EDGAR_DIR / f"{sym}.json"
            if out.exists():
                continue
            k = cik.get(sym) or cik.get(sym.replace(".", "-")) or cik.get(sym.replace("-", "."))
            if not k:
                print(f"  no CIK for {sym}")
                continue
            r = c.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{k:010d}.json")
            if r.status_code != 200:
                print(f"  {sym}: HTTP {r.status_code}")
                time.sleep(1)
                continue
            facts = r.json().get("facts", {})
            keep: dict[str, dict] = {}
            want = {t for tags in (*FLOWS.values(), *STOCKS.values()) for t in tags}
            for t in want:
                if t in facts.get("us-gaap", {}):
                    keep[t] = facts["us-gaap"][t]["units"]
            for t in SHARES:
                if t in facts.get("dei", {}):
                    keep[t] = facts["dei"][t]["units"]
            out.write_text(json.dumps(keep))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
            time.sleep(0.12)  # SEC asks for <= 10 req/s
    print(f"done -> {EDGAR_DIR}")


# ----------------------------------------------------------------- PIT build
def _rows(units: dict, unit: str) -> pd.DataFrame:
    rs = units.get(unit) or []
    df = pd.DataFrame(rs)
    if df.empty:
        return df
    df = df[df["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])].copy()
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"]) + pd.Timedelta(days=FILING_LAG_DAYS)
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"])
        df["days"] = (df["end"] - df["start"]).dt.days
    return df


def _pick(raw: dict, tags: list[str], unit: str = "USD") -> pd.DataFrame:
    """Concatenate candidate tags; earlier tags win on the same period."""
    parts = []
    for prio, t in enumerate(tags):
        if t in raw:
            d = _rows(raw[t], unit)
            if not d.empty:
                parts.append(d.assign(prio=prio))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _records(df: pd.DataFrame) -> list[tuple]:
    """(filed, start, end, days, val, prio) — one row per period, as FIRST
    reported under the highest-priority tag. As-first-reported is the
    conservative point-in-time choice: a restatement cannot leak backwards."""
    if df.empty:
        return []
    if "days" not in df:
        df = df.assign(start=df["end"], days=0)
    best: dict[tuple, tuple] = {}
    cols = ["filed", "start", "end", "days", "val", "prio"]
    for f, st, en, d, v, pr in df[cols].itertuples(index=False):
        k = (st, en)
        cur = best.get(k)
        if cur is None or (pr, f) < (cur[5], cur[0]):
            best[k] = (f, st, en, d, v, pr)
    return sorted(best.values(), key=lambda r: r[0])


def ttm_series(df: pd.DataFrame) -> pd.Series:
    """Event series: at each filing date, the TTM value knowable then."""
    if df.empty or "days" not in df:
        return pd.Series(dtype=float)
    recs = _records(df)
    known: list[tuple] = []
    out = {}
    i = 0
    for fdate in sorted({r[0] for r in recs}):
        while i < len(recs) and recs[i][0] <= fdate:
            known.append(recs[i])
            i += 1
        ann = [r for r in known if 350 <= r[3] <= 380]
        if not ann:
            continue
        a = max(ann, key=lambda r: r[2])
        val = a[4]
        ytd = [r for r in known if 80 <= r[3] <= 290 and r[2] > a[2]]
        if ytd:
            cur = max(ytd, key=lambda r: r[2])
            prev = [
                r for r in known if abs(r[3] - cur[3]) <= 15 and 350 <= (cur[2] - r[2]).days <= 380
            ]
            if prev:
                val = a[4] + cur[4] - max(prev, key=lambda r: r[0])[4]
        out[fdate] = val
    return pd.Series(out, dtype=float)


def stock_series(df: pd.DataFrame) -> pd.Series:
    """Event series of the latest balance-sheet value known at each filing."""
    if df.empty:
        return pd.Series(dtype=float)
    out = {}
    latest = None
    for f, _st, en, _d, v, _p in _records(df):
        if latest is None or en >= latest[0]:
            latest = (en, v)
        out[f] = latest[1]
    return pd.Series(out, dtype=float)


def build_fundamentals(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for p in sorted(EDGAR_DIR.glob("*.json")):
        sym = p.stem
        raw = json.loads(p.read_text())
        ser = {}
        for name, tags in FLOWS.items():
            ser[name] = ttm_series(_pick(raw, tags))
        for name, tags in STOCKS.items():
            ser[name] = stock_series(_pick(raw, tags))
        sh = _pick(raw, SHARES, "shares")
        ser["shares"] = stock_series(sh) if not sh.empty else pd.Series(dtype=float)
        frame = pd.DataFrame(index=dates)
        for name, s in ser.items():
            if s.empty:
                frame[name] = np.nan
                continue
            s = s.sort_index()
            s = s[~s.index.duplicated(keep="last")]
            frame[name] = s.reindex(s.index.union(dates)).ffill().reindex(dates)
        # stale guard: a value older than 15 months is not "current"
        frame["sym"] = sym
        rows.append(frame.reset_index(names="date"))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------- factors
def zs(s: pd.Series) -> pd.Series:
    s = s.clip(s.quantile(0.02), s.quantile(0.98))
    return (s - s.mean()) / s.std()


def _long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    """Wide (date x sym) -> long [date, sym, name]."""
    w = wide.rename_axis(index="date", columns="sym")
    return w.stack().rename(name).reset_index()


def load_prices() -> pd.DataFrame:
    closes = {}
    for sym in [*universe(), "SPY"]:
        for ac in ("equity", "etf"):
            p = ROOT / f"data/parquet/{ac}/{sym}/1d.parquet"
            if p.exists():
                closes[sym] = pd.read_parquet(p)["adj_close"]
                break
    px = pd.DataFrame(closes)
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def score(
    f: pd.DataFrame, px_m: pd.DataFrame, raw_m: pd.DataFrame, exfin: bool = False
) -> pd.DataFrame:
    """Cross-sectional quality (Q) and value (V) composites per month."""
    f = f.copy()
    price = _long(raw_m, "price")
    f = f.merge(price, on=["date", "sym"], how="inner")
    gp = f["gross_profit"].fillna(f["revenue"] - f["cogs"])
    f["mcap"] = f["price"] * f["shares"]
    a = f["assets"].where(f["assets"] > 0)
    f["gpa"] = gp / a  # Novy-Marx gross profitability
    f["roa"] = f["net_income"] / a
    f["accruals"] = (f["net_income"] - f["cfo"]) / a  # low = good
    f["lev"] = f["liabilities"] / a  # low = good
    f["ey"] = f["net_income"] / f["mcap"]
    f["fcfy"] = (f["cfo"] - f["capex"].fillna(0)) / f["mcap"]
    f["sy"] = f["revenue"] / f["mcap"]
    f["bp"] = f["equity"] / f["mcap"]
    mom = _long(px_m.shift(1) / px_m.shift(7) - 1, "mom6")
    f = f.merge(mom, on=["date", "sym"], how="left")
    f = f[f["mcap"] > 0]
    if exfin:
        # Banks/insurers have no gross profit and 90% "leverage" by design;
        # left in, they ARE the junk decile. Drop them and the leverage term.
        f = f[f["gpa"].notna()]

    def per_date(g: pd.DataFrame) -> pd.DataFrame:
        qs = [zs(g["gpa"]), zs(g["roa"]), -zs(g["accruals"])]
        if not exfin:
            qs.append(-zs(g["lev"]))
        q = pd.concat(qs, axis=1)
        v = pd.concat([zs(g["ey"]), zs(g["fcfy"]), zs(g["sy"]), zs(g["bp"])], axis=1)
        g = g.assign(Q=q.mean(axis=1, skipna=True), V=v.mean(axis=1, skipna=True))
        g.loc[q.notna().sum(axis=1) < 2, "Q"] = np.nan
        g.loc[v.notna().sum(axis=1) < 2, "V"] = np.nan
        return g

    return pd.concat([per_date(g) for _, g in f.groupby("date")], ignore_index=True)


# ----------------------------------------------------------------- backtest
def run(borrow: float, cost_bp: float, frac: float, exfin: bool = False) -> None:
    px = load_prices()
    month_ends = px.resample("ME").last().index
    dates = pd.DatetimeIndex([px.index[px.index <= d][-1] for d in month_ends])
    raw_m = px.loc[dates]
    raw_m.index = dates
    cache = ROOT / "output/quality_short/fundamentals_pit.parquet"
    if cache.exists():
        fund = pd.read_parquet(cache)
    else:
        fund = build_fundamentals(dates)
        cache.parent.mkdir(parents=True, exist_ok=True)
        fund.to_parquet(cache)
    sc = score(fund, raw_m, raw_m, exfin)
    fwd = _long(raw_m.shift(-1) / raw_m - 1, "fwd")
    sc = sc.merge(fwd, on=["date", "sym"], how="left")
    spy_fwd = (raw_m["SPY"].shift(-1) / raw_m["SPY"] - 1).rename("spy")
    print(f"coverage: median {sc.groupby('date')['Q'].count().median():.0f} names/month with Q")

    def book(g: pd.DataFrame, strat: str) -> tuple[set, set]:
        g = g.dropna(subset=["Q", "fwd"])
        n = max(int(len(g) * frac), 5)
        qv = g["Q"] + g["V"].fillna(0)
        if strat == "QMJ: long top-Q, short bottom-Q":
            return set(g.nlargest(n, "Q")["sym"]), set(g.nsmallest(n, "Q")["sym"])
        if strat == "long top-Q, short junk+expensive":
            return set(g.nlargest(n, "Q")["sym"]), set(g.loc[qv.nsmallest(n).index, "sym"])
        if strat == "long top-Q, short junk+expensive+falling":
            bad = g.assign(qv=qv)[g["mom6"] < 0]
            return set(g.nlargest(n, "Q")["sym"]), set(bad.nsmallest(n, "qv")["sym"])
        if strat == "long Q+V, short Q+V":
            return set(g.loc[qv.nlargest(n).index, "sym"]), set(g.loc[qv.nsmallest(n).index, "sym"])
        raise KeyError(strat)

    strats = [
        "QMJ: long top-Q, short bottom-Q",
        "long top-Q, short junk+expensive",
        "long top-Q, short junk+expensive+falling",
        "long Q+V, short Q+V",
    ]
    rows, curves = [], {}
    for strat in strats:
        prev_long: set = set()
        prev_short: set = set()
        recs = []
        for d, g in sc.groupby("date"):
            if pd.isna(spy_fwd.get(d)):
                continue
            longs, shorts = book(g, strat)
            if len(longs) < 5 or len(shorts) < 5:
                continue
            r = g.set_index("sym")["fwd"]
            rl, rs = r.reindex(list(longs)).mean(), r.reindex(list(shorts)).mean()
            turn = (
                len(longs ^ prev_long) / max(len(longs), 1)
                + len(shorts ^ prev_short) / max(len(shorts), 1)
            ) / 2
            cost = turn * cost_bp / 1e4 * 2 + borrow / 12
            recs.append(
                {
                    "date": d,
                    "long": rl,
                    "short": rs,
                    "ls": rl - rs - cost,
                    "long_xs": rl - spy_fwd[d],
                    "short_xs": spy_fwd[d] - rs,
                    "spy": spy_fwd[d],
                    "turn": turn,
                }
            )
            prev_long, prev_short = longs, shorts
        m = pd.DataFrame(recs).set_index("date")
        curves[strat] = (1 + m["ls"]).cumprod()
        yrs = len(m) / 12
        ann = (1 + m["ls"]).prod() ** (1 / yrs) - 1
        vol = m["ls"].std() * np.sqrt(12)
        dd = (curves[strat] / curves[strat].cummax() - 1).min()
        by_year = m["ls"].groupby(m.index.year).apply(lambda x: (1 + x).prod() - 1)
        rows.append(
            {
                "strategy": strat,
                "months": len(m),
                "ann_ret": ann,
                "ann_vol": vol,
                "sharpe": ann / vol if vol else np.nan,
                "max_dd": dd,
                "hit": (m["ls"] > 0).mean(),
                "long_leg_vs_SPY/yr": m["long_xs"].mean() * 12,
                "short_leg_vs_SPY/yr": m["short_xs"].mean() * 12,
                "corr_to_SPY": m["ls"].corr(m["spy"]),
                "turnover/mo": m["turn"].mean(),
                **{str(y): v for y, v in by_year.items()},
            }
        )
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))
    out = ROOT / "output/quality_short"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv")
    print(f"wrote {out}/summary.csv, curves.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run"])
    ap.add_argument("--borrow", type=float, default=0.02, help="annual short borrow cost")
    ap.add_argument("--cost-bp", type=float, default=10.0, help="bp per side traded")
    ap.add_argument("--frac", type=float, default=0.10, help="fraction per leg (0.10 = decile)")
    ap.add_argument("--exfin", action="store_true", help="drop financials + leverage term")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch()
    else:
        run(a.borrow, a.cost_bp, a.frac, a.exfin)


if __name__ == "__main__":
    main()
