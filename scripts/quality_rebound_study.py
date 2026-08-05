"""Does quality mega-cap bleeding mean-revert? — an honest first pass.

Yan's hypothesis (2026-08-05): "big, very high quality stocks — NVDA,
MSFT, GOOG, AMZN — tend to always rebound after weeks or months of
bleeding, and the sweet spot is found via a mix of technical chart, how
long it has been down, and whether it is oversold."

The claim has three separable parts and this script tests them apart,
because conflating them is how a truism gets mistaken for an edge:

1. Do these names rebound after drawdowns? (Almost certainly yes — they
   rebound after everything, because they went up over the sample. This
   number alone proves nothing.)
2. Do they rebound MORE than they usually do? That is the only question
   that matters, so every conditional return is reported against the
   SAME name's unconditional base rate over the SAME horizon.
3. Do they rebound more than SPY? A strategy that beats a stock's own
   base rate but loses to the index is not worth the single-name risk.

Deliberately NOT a Strategy subclass yet. The first question is whether
there is anything here at all; wiring a signal into the live path before
answering that is how the desk ends up trading a hunch with a backtest
stapled to it.

Survivorship bias is the elephant: this universe is picked BECAUSE these
names won. Read every number below as an upper bound.

    uv run python scripts/quality_rebound_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading.runtime.portfolio_stats import _read_close

# Mega-cap quality, as Yan described it. Survivorship-biased on purpose —
# it is his stated universe, and the point is to test HIS claim.
QUALITY = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "AVGO",
    "TSLA",
    "LLY",
    "JPM",
    "V",
    "MA",
    "UNH",
    "COST",
    "WMT",
    "HD",
    "PG",
    "JNJ",
    "XOM",
    "ORCL",
    "NFLX",
    "AMD",
    "CRM",
    "ADBE",
    "QCOM",
    "TXN",
]
HORIZONS = (21, 63, 126)  # ~1m, ~3m, ~6m


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def features(close: pd.Series) -> pd.DataFrame:
    """Drawdown, oversold, and how long the bleeding has run."""
    peak = close.rolling(252, min_periods=60).max()
    f = pd.DataFrame(index=close.index)
    f["dd"] = close / peak - 1.0
    f["rsi"] = rsi(close)
    # Days since the trailing peak — Yan's "how long has it been down".
    f["days_since_high"] = close.groupby((close >= peak).cumsum()).cumcount()  # type: ignore[arg-type]
    # "A level where it bounced once already": price within 3% of the
    # lowest low of the last 6 months, i.e. retesting a floor that held.
    floor = close.rolling(126, min_periods=60).min()
    f["at_prior_floor"] = close / floor - 1.0 <= 0.03
    return f


def forward(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1.0


def main() -> None:
    data_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data/parquet"
    )
    spy = _read_close(data_dir, "SPY")
    if spy is None:
        print("no SPY in the cache — backfill first")
        return

    rows: list[dict[str, object]] = []
    for sym in QUALITY:
        close = _read_close(data_dir, sym)
        if close is None or len(close) < 400:
            continue
        close = close.sort_index()
        f = features(close)
        bench = spy.reindex(close.index).ffill()
        for h in HORIZONS:
            fwd = forward(close, h)
            fwd_spy = forward(bench, h)
            excess = fwd - fwd_spy
            base = fwd.dropna()
            # The three conditions, separately and together.
            conds = {
                "drawdown >15%": f["dd"] <= -0.15,
                "oversold (RSI<35)": f["rsi"] < 35,
                "down >60 sessions": f["days_since_high"] >= 60,
                "at a prior floor": f["at_prior_floor"],
                "ALL FOUR": (
                    (f["dd"] <= -0.15)
                    & (f["rsi"] < 35)
                    & (f["days_since_high"] >= 60)
                    & f["at_prior_floor"]
                ),
            }
            for name, mask in conds.items():
                sel = fwd[mask.fillna(False)].dropna()
                if len(sel) < 20:  # too few observations to mean anything
                    continue
                rows.append(
                    {
                        "symbol": sym,
                        "horizon": h,
                        "condition": name,
                        "n": len(sel),
                        "mean": sel.mean(),
                        "base_mean": base.mean(),
                        "lift": sel.mean() - base.mean(),
                        "hit": (sel > 0).mean(),
                        "base_hit": (base > 0).mean(),
                        "excess_vs_spy": excess[mask.fillna(False)].dropna().mean(),
                    }
                )

    if not rows:
        print("no usable history — backfill the quality names first")
        return

    df = pd.DataFrame(rows)
    print(f"\n{len(df.symbol.unique())} names · {df.n.sum():,} conditional observations")
    print("\n'lift' = conditional mean MINUS the same name's unconditional mean.")
    print(
        "A positive raw return with ~zero lift means the name drifts up, not that the signal works.\n"
    )

    agg = (
        df.groupby(["condition", "horizon"])
        .agg(
            names=("symbol", "nunique"),
            obs=("n", "sum"),
            mean=("mean", "mean"),
            base=("base_mean", "mean"),
            lift=("lift", "mean"),
            hit=("hit", "mean"),
            base_hit=("base_hit", "mean"),
            vs_spy=("excess_vs_spy", "mean"),
        )
        .reset_index()
    )
    for col in ("mean", "base", "lift", "vs_spy"):
        agg[col] = (agg[col] * 100).round(2)
    for col in ("hit", "base_hit"):
        agg[col] = (agg[col] * 100).round(1)
    print(agg.to_string(index=False))

    # The share of names where the signal helps at all — an average lift
    # carried by two names is not a rule, it is those two names.
    print("\nBreadth — % of names with positive lift (a mean can hide a lottery):")
    breadth = (
        df.assign(win=df.lift > 0)
        .groupby(["condition", "horizon"])["win"]
        .mean()
        .mul(100)
        .round(0)
        .reset_index()
    )
    print(breadth.to_string(index=False))
    print("\nCaveat: this universe was chosen because it won. Treat as an upper bound.")
    by_year(data_dir, spy)


def by_year(data_dir: Path, spy: pd.Series) -> None:
    """The only table that decides anything.

    A pooled average over eight years hides WHEN the edge existed. These
    signals fire in market-wide selloffs, so the pooled number is a
    weighted average of a handful of regimes — and if the good ones are
    all in the past, the aggregate is a backward-looking artifact rather
    than something to trade next month.
    """
    rows = []
    for sym in QUALITY:
        close = _read_close(data_dir, sym)
        if close is None or len(close) < 400:
            continue
        close = close.sort_index()
        f = features(close)
        mask = (
            (f["dd"] <= -0.15)
            & (f["rsi"] < 35)
            & (f["days_since_high"] >= 60)
            & f["at_prior_floor"]
        ).fillna(False)
        fwd = forward(close, 63)
        bench = forward(spy.reindex(close.index).ffill(), 63)
        for t in close.index[mask]:
            if pd.notna(fwd.get(t)) and pd.notna(bench.get(t)):
                rows.append({"t": t, "excess": fwd[t] - bench[t]})
    if not rows:
        return
    h = pd.DataFrame(rows)
    g = h.groupby(h["t"].dt.year).agg(
        obs=("excess", "size"),
        mean_excess_pct=("excess", lambda s: round(s.mean() * 100, 1)),
        pct_positive=("excess", lambda s: round((s > 0).mean() * 100)),
    )
    print("\nALL FOUR, 63d excess vs SPY, BY YEAR — does the edge still exist?\n")
    print(g.to_string())


if __name__ == "__main__":
    main()
