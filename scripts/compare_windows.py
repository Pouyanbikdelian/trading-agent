r"""Compare top_k_momentum across two date windows vs SPY and QQQ.

Two windows the user asked about:
  A: 2015-01-01 -> 2020-01-01  (pre-COVID 5y)
  B: 2020-01-01 -> today       (COVID crash + 2022 bear + AI bull)

Why this script exists separately from scripts/benchmark_2020.py
-----------------------------------------------------------------
We want a single side-by-side report that:
  * Pulls fresh yfinance data for ANY missing history (cache only has 2018+)
  * Runs the production strategy as configured in .env.example
  * Reports against both SPY (S&P 500 proxy) and QQQ (NDX-100 proxy)
  * Surfaces flaws — e.g., does the model crater in early-2020 even with
    the regime/derisk overlay enabled? Where does the in-sample edge come
    from?

The strategy uses NO overlays here — we want to see the raw signal first.
Once we trust the raw, we can layer regime_derisk and look at the
incremental improvement.

Run:
    uv run python scripts/compare_windows.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading.backtest import CostModel, compute_metrics, run_vectorized
from trading.core.universes import load_universe
from trading.selection import regime_derisk
from trading.strategies import TopKMomentum, TopKMomentumParams

# ---- Configuration -------------------------------------------------------

WINDOWS: list[tuple[str, datetime, datetime]] = [
    (
        "2015-2020 (pre-COVID)",
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    ),
    (
        "2020-now (COVID + bear + AI bull)",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 13, tzinfo=timezone.utc),
    ),
]

# Winning config (see docs/winning_config.md)
PARAMS = TopKMomentumParams(
    k=15,
    lookback=126,
    skip=21,
    rebalance=63,
    abs_momentum_threshold=0.0,
    target_gross=1.0,
    max_per_position=0.20,
    vol_lookback=60,
    min_decorrelated=0,
)
COSTS = CostModel(commission_bps=3.0, slippage_bps=7.0)
INITIAL = 100_000.0

UNIVERSE_NAME = "sp500"
BENCHMARKS = ("SPY", "QQQ")


# ---- Data plumbing -------------------------------------------------------


def _yf_download(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Single batched yfinance call. Faster than the cache adapter and
    bypasses the project's Parquet layer — fine for a one-off analysis."""
    import yfinance as yf

    print(f"  fetching {len(symbols)} symbols from yfinance...")
    raw = yf.download(
        " ".join(symbols),
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    # yfinance returns a multi-level frame; pull Close per symbol.
    out: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                s = raw[sym]["Close"].dropna()
            except KeyError:
                continue
            if len(s) > 0:
                out[sym] = s
    else:
        # single-symbol path
        s = raw["Close"].dropna()
        if len(s) > 0:
            out[symbols[0]] = s
    df = pd.DataFrame(out)
    # Make tz-aware UTC for downstream alignment.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _bench_metrics(prices: pd.Series, costs: CostModel) -> dict[str, float]:
    """Buy-and-hold metrics for a single benchmark series."""
    p = prices.dropna()
    ret = p.pct_change().fillna(0.0)
    eq = INITIAL * (1.0 + ret).cumprod()
    n = len(p)
    years = n / 252.0
    total = float(eq.iloc[-1] / INITIAL - 1.0)
    cagr = float((eq.iloc[-1] / INITIAL) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(ret.std(ddof=1) * np.sqrt(252))
    sharpe = float((ret.mean() * 252) / vol) if vol > 0 else 0.0
    rolling_max = eq.cummax()
    drawdown = float((eq / rolling_max - 1.0).min())
    return {
        "total_return": total,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "final_equity": float(eq.iloc[-1]),
    }


# ---- Run one window ------------------------------------------------------


def _run_window(label: str, start: datetime, end: datetime) -> None:
    print(f"\n{'=' * 72}\n  Window: {label}")
    print(f"  {start.date()}  ->  {end.date()}")
    print("=" * 72)

    universe_syms = [i.symbol for i in load_universe(UNIVERSE_NAME)]
    all_syms = universe_syms + list(BENCHMARKS)

    df = _yf_download(all_syms, start, end)
    df = df.dropna(axis=1, thresh=int(0.95 * len(df)))  # need ≥95% coverage
    df = df.ffill().dropna(how="any")
    print(f"  usable symbols after coverage filter: {df.shape[1]} / {len(all_syms)}")
    print(f"  rows: {len(df)}")

    # ---- benchmarks
    print("\n  Benchmarks (buy & hold, same window, same costs basis):")
    for b in BENCHMARKS:
        if b not in df.columns:
            print(f"    {b}: not enough data")
            continue
        m = _bench_metrics(df[b], COSTS)
        print(
            f"    {b:<4}  CAGR {m['cagr']:>7.2%}  Sharpe {m['sharpe']:>5.2f}  "
            f"MaxDD {m['max_drawdown']:>7.2%}  ${m['final_equity']:>11,.0f}"
        )

    # ---- strategy
    strat_universe_cols = [c for c in df.columns if c not in BENCHMARKS]
    strat_prices = df[strat_universe_cols]
    strat = TopKMomentum(params=PARAMS)
    weights = strat.generate(strat_prices)

    # Run two variants: raw signal vs same signal with regime_derisk overlay.
    # The overlay needs SPY as the benchmark, which we already have in df.
    variants: list[tuple[str, pd.DataFrame]] = [
        ("raw (no overlay)        ", weights),
    ]
    if "SPY" in df.columns:
        # Pass an explicit SPY series so the overlay doesn't need SPY
        # in the strategy's price frame.
        weights_dr = regime_derisk(
            weights,
            strat_prices,
            benchmark="SPY",
            benchmark_prices=df["SPY"],
            trend_window=200,
            fast_window=50,
            confirm_days=5,
            derisk_scale=0.30,
            deep_derisk_scale=0.10,
        )
        variants.append(("+ regime_derisk overlay ", weights_dr))

    strategy_final: dict[str, float] = {}
    for label_v, w in variants:
        result = run_vectorized(strat_prices, w, costs=COSTS)
        metrics = compute_metrics(result, periods_per_year=252)
        eq = INITIAL * (1.0 + result.returns.fillna(0.0)).cumprod()
        final = float(eq.iloc[-1])
        strategy_final[label_v] = final
        tag = label_v.strip()
        print(f"\n  Strategy: top_k_momentum, {tag}:")
        print(
            f"    CAGR {metrics['cagr']:>7.2%}  Sharpe {metrics['sharpe']:>5.2f}  "
            f"MaxDD {metrics['max_drawdown']:>7.2%}  ${final:>11,.0f}"
        )
        print(
            f"    annual vol {metrics['ann_vol']:.2%}  "
            f"sortino {metrics['sortino']:.2f}  "
            f"calmar {metrics['calmar']:.2f}  "
            f"trades {int(metrics['n_trades'])}"
        )

    # ---- ratios vs benchmarks
    for b in BENCHMARKS:
        if b not in df.columns:
            continue
        b_final = INITIAL * float(df[b].iloc[-1] / df[b].iloc[0])
        for label_v, final in strategy_final.items():
            print(f"  {label_v.strip():<25} vs {b}:  {final / b_final:.2f}x the dollars")


def main() -> None:
    for label, start, end in WINDOWS:
        _run_window(label, start, end)
    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
