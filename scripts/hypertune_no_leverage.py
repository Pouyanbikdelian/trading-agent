r"""No-leverage hypertune — beat the indices on CAGR without borrowing.

Diagnosis from the previous benchmark
-------------------------------------
risk_parity alone already matches QQQ on CAGR (19.76% vs 19.94%) at
lower vol; what was costing us was the inverse-vol combiner, which down-
weights risk_parity precisely because it has the highest realised vol of
the three strategies — even though that vol was being well-compensated.

This script tries the configurations that are theoretically capable of
closing or beating the index gap at no extra leverage:

1. ``risk_parity`` on a momentum-filtered top-K universe (concentration).
2. ``sharpe_weighted`` combiner (reward strategies that actually deliver).
3. ``equal_weight`` combiner (no anti-correlation between weight and
   realised return).
4. Single-strategy "concentrate-then-buy-and-hold" plays: top-K by
   trailing return, sized to 100% gross via inverse-vol.

Universe pre-filter: top-K names by trailing 252-day return at each
month-end. Re-filters monthly so we don't ride a single bull cohort.

Outputs
-------
* Sorted leaderboard in stdout.
* ``visualize/hypertune_equity.png`` and ``visualize/hypertune_drawdown.png``
* ``visualize/hypertune_risk_return.png`` scatter.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading.backtest import CostModel, run_vectorized
from trading.core.config import settings
from trading.core.types import AssetClass, Instrument
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.selection.combine import (
    equal_weight,
    inverse_vol,
    sharpe_weighted,
)
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


def _momentum_topk_mask(
    prices: pd.DataFrame, *, k: int, lookback: int = 252, rebalance: int = 21
) -> pd.DataFrame:
    r"""Boolean mask: True where instrument is in the top-K by trailing return
    at the most recent rebalance bar. Rebalanced every ``rebalance`` bars."""
    ret = prices.pct_change(lookback)
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    rebal_bars = np.arange(len(prices)) % rebalance == 0
    # First rebalance once we have `lookback` bars of history.
    rebal_bars[:lookback] = False
    rank_state = pd.Series(False, index=prices.columns)
    for i in range(len(prices.index)):
        if rebal_bars[i]:
            r = ret.iloc[i]
            if r.notna().any():
                top = r.nlargest(k).index
                rank_state = pd.Series(False, index=prices.columns)
                rank_state.loc[top] = True
        mask.iloc[i] = rank_state
    return mask


def _apply_universe_mask(
    weights: pd.DataFrame, mask: pd.DataFrame, *, max_per_position: float = 0.20
) -> pd.DataFrame:
    r"""Zero out weights for names not in the top-K mask, then redistribute
    the freed capital across the surviving names so the row's gross
    exposure matches what it was before masking. Individual positions are
    capped at ``max_per_position`` (default 20%) so concentration doesn't
    blow a single name into the portfolio.

    This is *not* leverage: we're redistributing the same 100% of capital
    onto fewer names, not borrowing to buy more.
    """
    masked = weights.where(mask, 0.0)
    pre = weights.abs().sum(axis=1)
    post = masked.abs().sum(axis=1)
    scale = (pre / post).replace([np.inf, -np.inf, np.nan], 0.0)
    redistributed = masked.mul(scale, axis=0)
    # Per-position cap: clip absolute size, preserve sign. After clipping
    # the row's gross can drop below the target — that's fine, we'd rather
    # carry cash than over-concentrate.
    sign = np.sign(redistributed.values)
    capped = sign * np.minimum(np.abs(redistributed.values), max_per_position)
    return pd.DataFrame(capped, index=redistributed.index, columns=redistributed.columns)


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
    }


def _build_configs(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = prices.shape[1]

    rp = get_strategy("risk_parity")(vol_lookback=40, rebalance=21)
    rp_w = rp.generate(prices)

    don = get_strategy("donchian")(lookback=20, weight_per_asset=1.0 / n)
    don_w = don.generate(prices)

    xs = get_strategy("xsec_momentum")(
        lookback=126,
        skip=5,
        top_frac=0.2,
        long_only=True,
    )
    xs_w = xs.generate(prices)

    no_cost = CostModel(commission_bps=0, slippage_bps=0)
    rp_r = run_vectorized(prices, rp_w, costs=no_cost).returns
    don_r = run_vectorized(prices, don_w, costs=no_cost).returns
    xs_r = run_vectorized(prices, xs_w, costs=no_cost).returns

    weights_by = {"donchian": don_w, "risk_parity": rp_w, "xsec_momentum": xs_w}
    returns_by = {"donchian": don_r, "risk_parity": rp_r, "xsec_momentum": xs_r}

    # Concentration: top-K momentum filters
    mask_30 = _momentum_topk_mask(prices, k=30)
    mask_20 = _momentum_topk_mask(prices, k=20)
    mask_10 = _momentum_topk_mask(prices, k=10)

    configs: dict[str, pd.DataFrame] = {
        # --- baselines ---
        "risk_parity (full universe)": rp_w,
        "donchian (l=20, full)": don_w,
        "combiner inv_vol (current)": inverse_vol(weights_by, returns_by, lookback=60),
        # --- new combiners ---
        "combiner equal_weight": equal_weight(weights_by),
        "combiner sharpe_weighted": sharpe_weighted(weights_by, returns_by, lookback=60),
        # --- concentration plays ---
        "risk_parity top-30 momentum": _apply_universe_mask(rp_w, mask_30),
        "risk_parity top-20 momentum": _apply_universe_mask(rp_w, mask_20),
        "risk_parity top-10 momentum": _apply_universe_mask(rp_w, mask_10),
        # --- combiner over concentrated universe ---
        "combiner sharpe_w + top-20 momo": _apply_universe_mask(
            sharpe_weighted(weights_by, returns_by, lookback=60),
            mask_20,
        ),
    }
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
    print(f"universe: nasdaq100, {prices.shape[1]} symbols, {len(prices)} bars\n")

    costs = CostModel(commission_bps=1.0, slippage_bps=2.0)
    configs = _build_configs(prices)

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

    # ----- leaderboard sorted by CAGR (the metric the user cares about) -----
    print(
        f"  {'name':<38}  {'$100k -> $':>13}  {'total':>9}  {'CAGR':>8}  "
        f"{'vol':>7}  {'Sharpe':>7}  {'MaxDD':>8}"
    )
    print("  " + "-" * 110)
    for name in sorted(metrics, key=lambda n: -metrics[n]["cagr"]):
        m = metrics[name]
        marker = "  *" if "buy & hold" in name else "   "
        print(
            f"{marker}{name:<38}  "
            f"{m['final_equity']:>13,.0f}  "
            f"{m['total_return']:>9.2%}  "
            f"{m['cagr']:>8.2%}  "
            f"{m['ann_vol']:>7.2%}  "
            f"{m['sharpe']:>7.3f}  "
            f"{m['max_drawdown']:>8.2%}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_equity(equity_curves, OUT_DIR / "hypertune_equity.png")
    _plot_drawdown(metrics, OUT_DIR / "hypertune_drawdown.png")
    _plot_risk_return(metrics, OUT_DIR / "hypertune_risk_return.png")
    print(f"\nplots written to {OUT_DIR}")
    return 0


# --------------------------------------------------------------- plotting ----


def _plot_equity(curves: dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, eq in curves.items():
        is_bench = "buy & hold" in name
        ax.plot(
            eq.index,
            eq.values,
            label=name,
            linewidth=2.2 if is_bench else 1.1,
            linestyle="--" if is_bench else "-",
            alpha=0.95 if is_bench else 0.85,
        )
    ax.set_yscale("log")
    ax.set_title("Hypertuned configurations — $100,000 equity (no leverage)")
    ax.set_xlabel("date")
    ax.set_ylabel("portfolio value ($, log)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_drawdown(metrics: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, m in metrics.items():
        is_bench = "buy & hold" in name
        ax.plot(
            m["drawdown_series"].index,
            m["drawdown_series"].values * 100,
            label=name,
            linewidth=2.2 if is_bench else 1.0,
            linestyle="--" if is_bench else "-",
            alpha=0.9,
        )
    ax.set_title("Drawdowns from running peak (%)")
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_risk_return(metrics: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for name, m in metrics.items():
        is_bench = "buy & hold" in name
        ax.scatter(
            m["ann_vol"] * 100,
            m["cagr"] * 100,
            s=220 if is_bench else 120,
            marker="s" if is_bench else "o",
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
    ax.set_title("Risk-return frontier — hypertuned configurations")
    ax.set_xlabel("annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
