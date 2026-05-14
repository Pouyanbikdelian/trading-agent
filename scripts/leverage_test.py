"""Vol-targeted combiner test: can we beat QQQ on CAGR at QQQ's risk level?

The honest hypothesis: our smart combiner has a higher Sharpe than the
index (~1.07 vs 0.91) but lower vol (because it sometimes holds cash).
If we vol-target the combiner to the index's realized vol, in theory we
should capture (combiner_Sharpe / index_Sharpe) × index_return on the
same risk budget.

Test:
1. Build the best combiner from the tuning runs.
2. Apply trading.selection.overlay.vol_target with target=0.24 (QQQ's vol).
3. Compare full curve to QQQ and SPY buy-and-hold.

Run via: uv run python scripts/leverage_test.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

from trading.backtest import (
    CostModel,
    compute_metrics,
    run_vectorized,
)
from trading.core.config import settings
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.selection.combine import inverse_vol
from trading.selection.overlay import vol_target
from trading.strategies import get_strategy


START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 13, tzinfo=timezone.utc)


def load_prices(universe: str, *, min_frac: float = 0.95) -> pd.DataFrame:
    cache = ParquetCache(settings.data_dir)
    series: dict[str, pd.Series] = {}
    for ins in load_universe(universe):
        df = cache.read(ins, "1D")
        if df.empty or "adj_close" not in df.columns:
            continue
        s = df["adj_close"].dropna()
        s = s[(s.index >= START) & (s.index <= END)]
        if not s.empty:
            series[ins.symbol] = s
    if not series:
        return pd.DataFrame()
    max_dates = max(len(s) for s in series.values())
    threshold = int(max_dates * min_frac)
    kept = {sym: s for sym, s in series.items() if len(s) >= threshold}
    return pd.DataFrame(kept).sort_index().dropna(how="any")


def benchmark(symbol: str, prices: pd.DataFrame) -> dict:
    cache = ParquetCache(settings.data_dir)
    from trading.core.types import AssetClass, Instrument

    ins = Instrument(symbol=symbol, asset_class=AssetClass.ETF)
    df = cache.read(ins, "1D")
    s = df["adj_close"].loc[prices.index[0] : prices.index[-1]]
    eq = s / s.iloc[0]
    daily = eq.pct_change().dropna()
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / n_years) - 1)
    vol = float(daily.std() * (252**0.5))
    sharpe = float(daily.mean() / daily.std() * (252**0.5))
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "name": symbol,
        "total": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
    }


def main() -> int:
    prices = load_prices("nasdaq100")
    if prices.empty:
        print("no nasdaq100 prices cached", file=sys.stderr)
        return 1
    print(f"NDX universe: {prices.shape[1]} symbols, {len(prices)} bars")
    n = prices.shape[1]
    costs = CostModel(commission_bps=1.0, slippage_bps=2.0)

    # --- 1. Build the three tuned strategies + their weights -------------
    strategies = {
        "donchian_tuned": get_strategy("donchian")(lookback=20, weight_per_asset=1.0 / n),
        "risk_parity_tuned": get_strategy("risk_parity")(vol_lookback=40, rebalance=21),
        "xsec_tuned": get_strategy("xsec_momentum")(
            lookback=126, skip=5, top_frac=0.2, long_only=True
        ),
    }

    weights_by = {name: s.generate(prices) for name, s in strategies.items()}
    no_cost = CostModel(commission_bps=0, slippage_bps=0)
    returns_by = {
        name: run_vectorized(prices, w, costs=no_cost).returns for name, w in weights_by.items()
    }

    # --- 2. Inverse-vol combine -----------------------------------------
    combined = inverse_vol(weights_by, returns_by, lookback=60)

    # --- 3. Three test configurations -----------------------------------
    print(f"\nBenchmarks 2018-2026:")
    qqq = benchmark("QQQ", prices)
    spy = benchmark("SPY", prices)
    rows = [spy, qqq]

    print(f"\n=== Vol-targeting sweep ===")
    # No vol target (baseline)
    base = run_vectorized(prices, combined, costs=costs)
    m = compute_metrics(base, periods_per_year=252)
    rows.append({"name": "combiner_no_overlay", **m})

    # Various target vol levels
    for target_vol in [0.10, 0.15, 0.20, 0.24, 0.30]:
        scaled = vol_target(
            combined,
            prices,
            target_vol=target_vol,
            lookback=60,
            periods_per_year=252,
            max_leverage=4.0,
        )
        result = run_vectorized(prices, scaled, costs=costs)
        m = compute_metrics(result, periods_per_year=252)
        rows.append({"name": f"combiner@vol={target_vol:.0%}", **m})

    # Print sorted by CAGR
    print(f"\n  {'name':30s} {'CAGR':>8} {'vol':>7} {'Sharpe':>7} {'MaxDD':>9}")
    leaderboard = sorted(rows, key=lambda r: r["cagr"], reverse=True)
    for r in leaderboard:
        vol_v = r.get("ann_vol", 0)
        print(
            f"  {r['name']:30s} {r['cagr']:>8.2%} {vol_v:>7.2%} {r['sharpe']:>7.3f} {r['max_drawdown']:>9.2%}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
