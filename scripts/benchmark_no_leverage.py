r"""No-leverage benchmark 2018-2026 with visualisations.

Constraint: gross exposure is hard-capped at 1.0 (the risk manager's
default :code:`MAX_GROSS_EXPOSURE`).  No borrowing, no margin, no
vol-target leverage above the natural portfolio volatility.

What the no-leverage cap rules out
----------------------------------
The vol-targeted variants we saw earlier (at 20-30% annualised vol)
required ~2-3x leverage on the underlying combiner.  Without leverage,
the *highest* portfolio volatility we can deliver is the volatility of
the most-aggressive un-levered strategy — risk_parity sized to
target_gross=1.0, which runs ~20% vol naturally.

So the comparison set becomes:

* tuned single strategies, each at 100% gross max
* the inverse-vol combiner over the trio (also at 100% gross max)
* vol-target overlay with max_leverage=1.0 (downsizing only — same
  behaviour as a circuit breaker; no upsizing)
* SPY + QQQ buy-and-hold as references

Outputs
-------
Writes four PNGs into ``visualize/``:

* ``equity_curves.png``    — $100k -> $end for every config + indices
* ``drawdowns.png``        — drawdown curves over time
* ``risk_return.png``      — scatter of CAGR vs annualised vol
* ``rolling_sharpe.png``   — 252-day rolling Sharpe per config

Run::

    uv run python scripts/benchmark_no_leverage.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trading.backtest import CostModel, compute_metrics, run_vectorized
from trading.core.config import settings
from trading.core.types import AssetClass, Instrument
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.selection.combine import inverse_vol
from trading.selection.overlay import vol_target
from trading.strategies import get_strategy


START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 13, tzinfo=timezone.utc)
INITIAL = 100_000.0
OUT_DIR = Path(__file__).resolve().parents[1] / "visualize"


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


def _benchmark_equity(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    cache = ParquetCache(settings.data_dir)
    ins = Instrument(symbol=symbol, asset_class=AssetClass.ETF)
    df = cache.read(ins, "1D")
    s = df["adj_close"].loc[idx[0] : idx[-1]]
    return (s / s.iloc[0] * INITIAL).rename(symbol)


def _metrics_from_equity(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0)
    vol = float(daily.std() * (252**0.5))
    sharpe = float(daily.mean() / daily.std() * (252**0.5)) if daily.std() > 0 else 0.0
    downside = daily[daily < 0]
    sortino = (
        float(daily.mean() / ((downside**2).mean() ** 0.5) * (252**0.5))
        if len(downside)
        else float("inf")
    )
    dd_series = equity / equity.cummax() - 1
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd_series.min()),
        "drawdown_series": dd_series,
        "rolling_sharpe": (daily.rolling(252).mean() / daily.rolling(252).std()) * (252**0.5),
    }


def _build_configs(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return a {name: weights_df} dict for every strategy + combiner we
    want to show.  All sized to keep gross exposure <= 1.0."""
    n = prices.shape[1]
    strategies = {
        "donchian (l=20)": get_strategy("donchian")(lookback=20, weight_per_asset=1.0 / n),
        "risk_parity": get_strategy("risk_parity")(vol_lookback=40, rebalance=21),
        "xsec_momentum": get_strategy("xsec_momentum")(
            lookback=126,
            skip=5,
            top_frac=0.2,
            long_only=True,
        ),
    }
    weights_by = {name: s.generate(prices) for name, s in strategies.items()}

    # Inverse-vol combiner.
    no_cost = CostModel(commission_bps=0, slippage_bps=0)
    returns_by = {
        name: run_vectorized(prices, w, costs=no_cost).returns for name, w in weights_by.items()
    }
    combined = inverse_vol(weights_by, returns_by, lookback=60)

    # Add the combiner and a "circuit-breaker" variant: vol_target with
    # max_leverage=1.0 only downsizes; it cannot upsize past 100% gross.
    configs = dict(weights_by)
    configs["combiner (inv_vol)"] = combined
    configs["combiner + vol-cap@15%"] = vol_target(
        combined,
        prices,
        target_vol=0.15,
        lookback=60,
        periods_per_year=252,
        max_leverage=1.0,
    )
    return configs


def main() -> int:
    prices = _load_universe_prices("nasdaq100")
    if prices.empty:
        print("no nasdaq100 prices cached", file=sys.stderr)
        return 1
    print(
        f"window: {prices.index[0].date()} -> {prices.index[-1].date()}  "
        f"({(prices.index[-1] - prices.index[0]).days / 365.25:.2f} years)"
    )
    print(f"universe: nasdaq100, {prices.shape[1]} symbols, {len(prices)} bars")

    costs = CostModel(commission_bps=1.0, slippage_bps=2.0)
    configs = _build_configs(prices)

    # --- backtest each config + benchmarks ------------------------------
    equity_curves: dict[str, pd.Series] = {}
    metrics: dict[str, dict] = {}
    for name, w in configs.items():
        result = run_vectorized(prices, w, costs=costs, initial_equity=INITIAL)
        equity_curves[name] = result.equity
        metrics[name] = _metrics_from_equity(result.equity)

    for sym in ("SPY", "QQQ"):
        eq = _benchmark_equity(sym, prices.index)
        equity_curves[f"{sym} (buy & hold)"] = eq
        metrics[f"{sym} (buy & hold)"] = _metrics_from_equity(eq)

    # --- print sorted table ---------------------------------------------
    print()
    print(
        f"  {'name':<32}  {'$100k -> $':>13}  {'total':>9}  {'CAGR':>8}  "
        f"{'vol':>7}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}"
    )
    print("  " + "-" * 116)
    for name in sorted(metrics, key=lambda n: -metrics[n]["sharpe"]):
        m = metrics[name]
        print(
            f"  {name:<32}  "
            f"{m['final_equity']:>13,.0f}  "
            f"{m['total_return']:>9.2%}  "
            f"{m['cagr']:>8.2%}  "
            f"{m['ann_vol']:>7.2%}  "
            f"{m['sharpe']:>7.3f}  "
            f"{m['sortino']:>8.3f}  "
            f"{m['max_drawdown']:>8.2%}"
        )

    # --- generate visualisations ----------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_equity_curves(equity_curves, OUT_DIR / "equity_curves.png")
    _plot_drawdowns(metrics, OUT_DIR / "drawdowns.png")
    _plot_risk_return(metrics, OUT_DIR / "risk_return.png")
    _plot_rolling_sharpe(metrics, OUT_DIR / "rolling_sharpe.png")
    print(f"\nwrote 4 plots into {OUT_DIR}")
    return 0


# --------------------------------------------------------------- plotting ----


def _plot_equity_curves(equity_curves: dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    # Plot indices in a distinctive style.
    for name, eq in equity_curves.items():
        is_benchmark = "buy & hold" in name
        ax.plot(
            eq.index,
            eq.values,
            label=name,
            linewidth=2.0 if is_benchmark else 1.2,
            linestyle="--" if is_benchmark else "-",
            alpha=0.95 if is_benchmark else 0.85,
        )
    ax.set_yscale("log")
    ax.set_title(
        f"Portfolio equity, $100,000 -> today  ({equity_curves[next(iter(equity_curves))].index[0].date()} -> {equity_curves[next(iter(equity_curves))].index[-1].date()})"
    )
    ax.set_xlabel("date")
    ax.set_ylabel("portfolio value ($, log scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_drawdowns(metrics: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, m in metrics.items():
        is_benchmark = "buy & hold" in name
        ax.plot(
            m["drawdown_series"].index,
            m["drawdown_series"].values * 100,
            label=name,
            linewidth=2.0 if is_benchmark else 1.0,
            linestyle="--" if is_benchmark else "-",
            alpha=0.9 if is_benchmark else 0.8,
        )
    ax.set_title("Drawdowns from running peak (%)")
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_risk_return(metrics: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, m in metrics.items():
        is_benchmark = "buy & hold" in name
        ax.scatter(
            m["ann_vol"] * 100,
            m["cagr"] * 100,
            s=180 if is_benchmark else 110,
            marker="s" if is_benchmark else "o",
            edgecolors="black",
            linewidths=1.2,
            alpha=0.85,
        )
        ax.annotate(
            name,
            (m["ann_vol"] * 100, m["cagr"] * 100),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Risk-return frontier (annualised)")
    ax.set_xlabel("annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_rolling_sharpe(metrics: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, m in metrics.items():
        is_benchmark = "buy & hold" in name
        ax.plot(
            m["rolling_sharpe"].index,
            m["rolling_sharpe"].values,
            label=name,
            linewidth=2.0 if is_benchmark else 1.0,
            linestyle="--" if is_benchmark else "-",
            alpha=0.9 if is_benchmark else 0.8,
        )
    ax.set_title("Rolling 252-day Sharpe ratio")
    ax.set_xlabel("date")
    ax.set_ylabel("Sharpe (annualised)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
