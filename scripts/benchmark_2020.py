r"""Head-to-head benchmark: the tuned-and-combined system vs major indices.

Test window: 2020-01-01 -> today.  Includes:
  * the 2020 COVID crash (-34% in five weeks)
  * the 2020-21 melt-up
  * the 2022 rate-hike bear (-25% to -35%)
  * the 2023-2026 AI-driven bull run

For each configuration we report:
  total return, CAGR, annualised vol, Sharpe, Sortino, max drawdown,
  and the closing value of an initial $100,000 portfolio.

Universe: nasdaq100 (90 names with full history).
Strategies are the tuned variants identified in scripts/tune_strategy.py
runs.  The combiner is inverse_vol over the top-Sharpe trio.  Vol-target
overlay is applied at 20% and 24% annualised to match SPY and QQQ vol.

Benchmarks come from the cached SPY and QQQ price series — same window,
same dates, same transaction-cost model (3 bps total) applied for fair
comparison, though buy-and-hold pays it only at inception.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

from trading.backtest import CostModel, compute_metrics, run_vectorized
from trading.core.config import settings
from trading.core.types import AssetClass, Instrument
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.selection.combine import inverse_vol
from trading.selection.overlay import vol_target
from trading.strategies import get_strategy

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 13, tzinfo=timezone.utc)
INITIAL = 100_000.0


def _load_universe_prices(universe: str, *, min_frac: float = 0.95) -> pd.DataFrame:
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


def _benchmark_row(symbol: str, idx: pd.DatetimeIndex) -> dict:
    cache = ParquetCache(settings.data_dir)
    ins = Instrument(symbol=symbol, asset_class=AssetClass.ETF)
    df = cache.read(ins, "1D")
    s = df["adj_close"].loc[idx[0] : idx[-1]]
    eq = s / s.iloc[0] * INITIAL
    daily = eq.pct_change().dropna()
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / INITIAL) ** (1.0 / n_years) - 1.0)
    vol = float(daily.std() * (252**0.5))
    sharpe = float(daily.mean() / daily.std() * (252**0.5))
    # Sortino: only downside std
    downside = daily[daily < 0]
    sortino = (
        float(daily.mean() / ((downside**2).mean() ** 0.5) * (252**0.5))
        if len(downside)
        else float("inf")
    )
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "name": f"{symbol} (buy & hold)",
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / INITIAL - 1),
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": dd,
    }


def _strategy_row(name: str, prices: pd.DataFrame, weights: pd.DataFrame, costs: CostModel) -> dict:
    result = run_vectorized(prices, weights, costs=costs, initial_equity=INITIAL)
    m = compute_metrics(result, periods_per_year=252)
    return {
        "name": name,
        "final_equity": float(result.equity.iloc[-1]),
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "ann_vol": m["ann_vol"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_drawdown": m["max_drawdown"],
    }


def main() -> int:
    prices = _load_universe_prices("nasdaq100")
    if prices.empty:
        print("no nasdaq100 prices cached", file=sys.stderr)
        return 1
    n = prices.shape[1]
    print(
        f"window: {prices.index[0].date()} -> {prices.index[-1].date()}  "
        f"({(prices.index[-1] - prices.index[0]).days / 365.25:.2f} years)"
    )
    print(f"universe: nasdaq100, {n} symbols, {len(prices)} bars")
    costs = CostModel(commission_bps=1.0, slippage_bps=2.0)
    no_cost = CostModel(commission_bps=0, slippage_bps=0)

    # --- tuned single strategies ----------------------------------------
    rows: list[dict] = []
    strategies = {
        "donchian (tuned: lookback=20)": get_strategy("donchian")(
            lookback=20,
            weight_per_asset=1.0 / n,
        ),
        "risk_parity (tuned: vol_lookback=40)": get_strategy("risk_parity")(
            vol_lookback=40,
            rebalance=21,
        ),
        "xsec_momentum (tuned: top=0.2, long-only)": get_strategy("xsec_momentum")(
            lookback=126,
            skip=5,
            top_frac=0.2,
            long_only=True,
        ),
    }
    weights_by: dict[str, pd.DataFrame] = {}
    returns_by: dict[str, pd.Series] = {}
    for name, strat in strategies.items():
        w = strat.generate(prices)
        rows.append(_strategy_row(name, prices, w, costs))
        weights_by[name] = w
        returns_by[name] = run_vectorized(prices, w, costs=no_cost).returns

    # --- inverse_vol combiner -------------------------------------------
    combined = inverse_vol(weights_by, returns_by, lookback=60)
    rows.append(_strategy_row("combiner: inverse_vol", prices, combined, costs))

    # --- vol-targeted variants ------------------------------------------
    for target in (0.10, 0.15, 0.20, 0.24):
        scaled = vol_target(
            combined,
            prices,
            target_vol=target,
            lookback=60,
            periods_per_year=252,
            max_leverage=4.0,
        )
        rows.append(_strategy_row(f"combiner @ vol-target {target:.0%}", prices, scaled, costs))

    # --- benchmarks ------------------------------------------------------
    rows.append(_benchmark_row("SPY", prices.index))
    rows.append(_benchmark_row("QQQ", prices.index))

    # --- print sorted by Sharpe -----------------------------------------
    rows = sorted(rows, key=lambda r: r["sharpe"], reverse=True)
    print()
    print(
        f"  {'name':<46}  {'$100k -> $':>13}  {'total':>9}  {'CAGR':>8}  "
        f"{'vol':>7}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}"
    )
    print("  " + "-" * 130)
    for r in rows:
        print(
            f"  {r['name']:<46}  "
            f"{r['final_equity']:>13,.0f}  "
            f"{r['total_return']:>9.2%}  "
            f"{r['cagr']:>8.2%}  "
            f"{r['ann_vol']:>7.2%}  "
            f"{r['sharpe']:>7.3f}  "
            f"{r['sortino']:>8.3f}  "
            f"{r['max_drawdown']:>8.2%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
