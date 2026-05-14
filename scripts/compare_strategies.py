"""Compare strategies + combiners on a universe.

Runs every single strategy standalone, then every combiner over the same
strategy set, and prints a sorted leaderboard. Optionally tracks the
DSR-weighted combiner's allocations over time so you can see how dynamic
the strategy mix is.

Usage::

    uv run python scripts/compare_strategies.py <universe> [--from DATE]

Currently runs on:
    donchian, ema_cross, xsec_momentum, rsi2, risk_parity
  Combiners:
    equal_weight, inverse_vol, min_variance, dsr_weighted
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from trading.backtest import (
    CostModel,
    compute_metrics,
    run_vectorized,
)
from trading.core.config import settings
from trading.core.types import Instrument
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.selection.combine import (
    dsr_weighted,
    equal_weight,
    inverse_vol,
    min_variance,
)
from trading.strategies import get_strategy


# Build strategy params dynamically: weight_per_asset is sized to 1/N so
# fully-long state lands at 100% gross, matching what the risk manager
# would scale to in the live runner.
def make_strategy_params(n_symbols: int) -> dict[str, dict]:
    per_asset = 1.0 / max(n_symbols, 1)
    return {
        "donchian":      {"weight_per_asset": per_asset},
        "ema_cross":     {"weight_per_asset": per_asset},
        "xsec_momentum": {"long_only": True, "top_frac": 0.2, "bottom_frac": 0.2},
        "rsi2":          {"weight_per_asset": per_asset},
        "risk_parity":   {"target_gross": 1.0},
    }


def load_prices(
    universe: str,
    start: datetime,
    end: datetime,
    *,
    min_history_frac: float = 0.95,
) -> pd.DataFrame:
    """Load prices for a universe, dropping symbols with insufficient history.

    yfinance recent-IPO names (ARM, GFS, etc.) only have ~1 year of data;
    inner-joining on dates with them would collapse the window. We instead
    drop symbols whose available history is < ``min_history_frac`` of the
    requested range, then inner-join on the rest.
    """
    cache = ParquetCache(settings.data_dir)
    series: dict[str, pd.Series] = {}
    for ins in load_universe(universe):
        df = cache.read(ins, "1D")
        if df.empty or "adj_close" not in df.columns:
            continue
        s = df["adj_close"].dropna()
        s = s[(s.index >= start) & (s.index <= end)]
        if not s.empty:
            series[ins.symbol] = s
    if not series:
        return pd.DataFrame()

    # Build a date index from the symbol with the longest history; require
    # each symbol to have >= min_history_frac of those dates.
    max_dates = max(len(s) for s in series.values())
    threshold = int(max_dates * min_history_frac)
    kept = {sym: s for sym, s in series.items() if len(s) >= threshold}
    dropped = set(series) - set(kept)
    if dropped:
        print(f"  dropped {len(dropped)} symbols with < {min_history_frac:.0%} history: "
              f"{sorted(dropped)[:6]}{'...' if len(dropped) > 6 else ''}")
    prices = pd.DataFrame(kept).sort_index().dropna(how="any")
    return prices


def backtest_one(prices: pd.DataFrame, weights: pd.DataFrame, costs: CostModel) -> dict:
    result = run_vectorized(prices, weights, costs=costs)
    m = compute_metrics(result, periods_per_year=252)
    return m


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("universe", help="Universe name from config/universes.yaml")
    p.add_argument(
        "--from", dest="start", default="2018-01-01",
        help="Start date (default: 2018-01-01)",
    )
    p.add_argument(
        "--to", dest="end", default=datetime.now(tz=timezone.utc).date().isoformat(),
        help="End date (default: today UTC)",
    )
    args = p.parse_args(argv)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print(f"loading {args.universe} prices...")
    prices = load_prices(args.universe, start, end)
    if prices.empty:
        print(f"no cached prices for {args.universe}", file=sys.stderr)
        return 1
    print(f"  {prices.shape[1]} symbols, {len(prices)} bars "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")
    costs = CostModel(commission_bps=1.0, slippage_bps=2.0)

    strategy_params = make_strategy_params(prices.shape[1])

    # --- 1. Run each strategy standalone ---------------------------------
    weights_by_strategy: dict[str, pd.DataFrame] = {}
    returns_by_strategy: dict[str, pd.Series] = {}
    rows: list[dict] = []

    print("\n=== single strategies ===")
    for name, params in strategy_params.items():
        try:
            cls = get_strategy(name)
            strat = cls(**params)
            w = strat.generate(prices)
            m = backtest_one(prices, w, costs)
            weights_by_strategy[name] = w
            # Use cost-free returns for the combiner so the combiner doesn't
            # double-count transaction costs.
            r = run_vectorized(prices, w, costs=CostModel(commission_bps=0, slippage_bps=0))
            returns_by_strategy[name] = r.returns
            rows.append({"name": name, **m})
            print(f"  {name:18s} CAGR={m['cagr']:>7.2%}  Sharpe={m['sharpe']:>5.2f}  "
                  f"MaxDD={m['max_drawdown']:>7.2%}  Trades={int(m['n_trades']):>5d}")
        except Exception as e:
            print(f"  {name:18s} FAILED: {e!r}")

    # --- 2. Run each combiner over the same strategy set ------------------
    # Pick the three strategies with the highest standalone Sharpe — combining
    # losers tends to dilute the winners.
    if len(rows) >= 3:
        top_three = sorted(rows, key=lambda r: r["sharpe"], reverse=True)[:3]
        chosen = [r["name"] for r in top_three]
    else:
        chosen = list(weights_by_strategy)

    sub_w = {n: weights_by_strategy[n] for n in chosen}
    sub_r = {n: returns_by_strategy[n] for n in chosen}

    print(f"\n=== combiners over {chosen} ===")
    combiner_results: list[dict] = []
    for combiner_name in ["equal_weight", "inverse_vol", "min_variance", "dsr_weighted"]:
        try:
            if combiner_name == "equal_weight":
                w = equal_weight(sub_w)
            elif combiner_name == "inverse_vol":
                w = inverse_vol(sub_w, sub_r, lookback=60)
            elif combiner_name == "min_variance":
                w = min_variance(sub_w, sub_r, lookback=60)
            elif combiner_name == "dsr_weighted":
                w = dsr_weighted(sub_w, sub_r, periods_per_year=252)
            m = backtest_one(prices, w, costs)
            combiner_results.append({"name": f"combiner:{combiner_name}", **m})
            print(f"  {combiner_name:18s} CAGR={m['cagr']:>7.2%}  Sharpe={m['sharpe']:>5.2f}  "
                  f"MaxDD={m['max_drawdown']:>7.2%}  Trades={int(m['n_trades']):>5d}")
        except Exception as e:
            print(f"  {combiner_name:18s} FAILED: {e!r}")

    # --- 3. Final leaderboard -------------------------------------------
    all_rows = rows + combiner_results
    leaderboard = sorted(all_rows, key=lambda r: r["sharpe"], reverse=True)
    print("\n=== leaderboard by Sharpe ===")
    print(f"  {'#':>2} {'name':22s} {'CAGR':>8} {'Sharpe':>7} {'Sortino':>7} {'MaxDD':>8} {'Calmar':>7}")
    for i, r in enumerate(leaderboard, 1):
        print(f"  {i:>2} {r['name']:22s} {r['cagr']:>8.2%} {r['sharpe']:>7.2f} "
              f"{r['sortino']:>7.2f} {r['max_drawdown']:>8.2%} {r['calmar']:>7.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
