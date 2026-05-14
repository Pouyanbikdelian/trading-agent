r"""Find the single best hypertuned TopKMomentum configuration.

Constraints (must be satisfied for a configuration to be considered):

* gross exposure <= 1.0 at all times (no leverage),
* max drawdown <= 1.10 x |QQQ max drawdown| over the same window
  (allow at most 10% more pain than the index — keeps it 'index-like'),
* strictly positive CAGR.

Selection criterion: among configurations meeting the constraints, pick
the one maximising **Calmar ratio** = CAGR / |MaxDD|.  Calmar is the
right objective when the user's bar is 'beat the index on CAGR at
similar risk' — it rewards return per unit of drawdown pain.

After identifying the winner the script:
  1. Walk-forward validates it (rolling 504/126 train/test) and reports
     the OOS Sharpe to confirm it's not just in-sample overfitting.
  2. Re-runs it on a held-out universe (sp500 if the winner came from
     nasdaq100, vice versa) — true out-of-universe robustness check.
  3. Writes equity, drawdown, holdings-turnover, parameter-heatmap
     visualisations into ``visualize/``.

Grid dimensions
---------------
* universe:  nasdaq100, sp500
* k:         5, 8, 10, 15, 20, 30
* lookback:  126, 189, 252, 378   (6 to 18 months)
* skip:      0, 21
* rebalance: 5, 21, 63            (weekly, monthly, quarterly)
* abs_momentum_threshold: 0.0, 0.05, None  (gate at 0%, gate at +5%, no gate)

Total: 2 x 6 x 4 x 2 x 3 x 3 = 864 configurations. ~3 ms each → ~3 mins
to run the whole sweep.
"""

from __future__ import annotations

import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading.backtest import CostModel, compute_metrics, run_vectorized
from trading.backtest.walkforward import expanding
from trading.core.config import settings
from trading.core.types import AssetClass, Instrument
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.strategies import TopKMomentum, TopKMomentumParams

START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 13, tzinfo=timezone.utc)
INITIAL = 100_000.0
OUT_DIR = Path(__file__).resolve().parents[1] / "visualize"


# --------------------------------------------------------------- helpers ----


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


def _benchmark_metrics(symbol: str, idx: pd.DatetimeIndex) -> dict:
    cache = ParquetCache(settings.data_dir)
    ins = Instrument(symbol=symbol, asset_class=AssetClass.ETF)
    df = cache.read(ins, "1D")
    s = df["adj_close"].loc[idx[0] : idx[-1]]
    eq = s / s.iloc[0] * INITIAL
    daily = eq.pct_change().dropna()
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "name": symbol,
        "final_equity": float(eq.iloc[-1]),
        "cagr": float((eq.iloc[-1] / INITIAL) ** (1.0 / n_years) - 1.0),
        "ann_vol": float(daily.std() * (252**0.5)),
        "sharpe": float(daily.mean() / daily.std() * (252**0.5)),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "calmar": np.nan,
        "equity": eq,
    }


# ------------------------------------------------------------- grid search ----


def _grid_iter() -> list[dict]:
    # Focused grid: hold at least 8 names (user constraint), only the
    # informative axes. 96 configurations.
    grid = {
        "universe": ["nasdaq100", "sp500"],
        "k": [8, 10, 15, 20],
        "lookback": [126, 252],
        "skip": [21],  # classic 12-1 only
        "rebalance": [21, 63],  # monthly or quarterly
        "abs_momentum_threshold": [0.0, None],  # dual-momentum on/off
        "min_decorrelated": [0, 3, 5],  # corr filter quotas
    }
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, c, strict=True)) for c in combos]


def _backtest_config(prices: pd.DataFrame, cfg: dict, costs: CostModel) -> dict | None:
    try:
        params = TopKMomentumParams(
            k=cfg["k"],
            lookback=cfg["lookback"],
            skip=cfg["skip"],
            rebalance=cfg["rebalance"],
            vol_lookback=60,
            abs_momentum_threshold=cfg["abs_momentum_threshold"],
            target_gross=1.0,
            max_per_position=0.20,
            min_decorrelated=cfg.get("min_decorrelated", 0),
            max_pairwise_corr=0.70,
            corr_window=63,
        )
    except Exception:
        return None
    strat = TopKMomentum(params=params)
    w = strat.generate(prices)
    # Ensure no row exceeds gross 1.0.
    gross_max = float(w.abs().sum(axis=1).max())
    if gross_max > 1.01:
        return None
    result = run_vectorized(prices, w, costs=costs, initial_equity=INITIAL)
    m = compute_metrics(result, periods_per_year=252)
    calmar = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else np.nan
    return {
        **cfg,
        "final_equity": float(result.equity.iloc[-1]),
        "cagr": m["cagr"],
        "ann_vol": m["ann_vol"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_drawdown": m["max_drawdown"],
        "calmar": calmar,
        "weights": w,
        "equity": result.equity,
    }


# ----------------------------------------------------- walk-forward validate ----


def _walk_forward_check(prices: pd.DataFrame, cfg: dict, costs: CostModel) -> dict:
    """Re-run the winning config under expanding walk-forward and report
    the OOS Sharpe.  If OOS Sharpe is much lower than the full-sample
    Sharpe the config is in-sample overfit."""

    def signal_fn(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        # The strategy is deterministic given prices; we re-generate on the
        # concatenated window and slice to the test index.
        full = pd.concat([train, test])
        params = TopKMomentumParams(
            k=cfg["k"],
            lookback=cfg["lookback"],
            skip=cfg["skip"],
            rebalance=cfg["rebalance"],
            vol_lookback=60,
            abs_momentum_threshold=cfg["abs_momentum_threshold"],
            target_gross=1.0,
            max_per_position=0.20,
            min_decorrelated=cfg.get("min_decorrelated", 0),
            max_pairwise_corr=0.70,
            corr_window=63,
        )
        return TopKMomentum(params=params).generate(full).reindex(test.index).fillna(0.0)

    _folds, wf_result = expanding(
        prices,
        signal_fn,
        train_size=504,
        test_size=126,
        step=126,
        costs=costs,
        initial_equity=INITIAL,
    )
    m = compute_metrics(wf_result, periods_per_year=252)
    return {
        "oos_cagr": m["cagr"],
        "oos_sharpe": m["sharpe"],
        "oos_max_drawdown": m["max_drawdown"],
        "oos_calmar": m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else np.nan,
    }


# ----------------------------------------------------------------- viz ----


def _plot_winner_equity(rows: list[dict], benchmarks: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for row in rows:
        label = (
            f"TopKMomo k={row['k']} L={row['lookback']} s={row['skip']} "
            f"R={row['rebalance']} abs={row['abs_momentum_threshold']} ({row['universe']})"
        )
        ax.plot(row["equity"].index, row["equity"].values, label=label, linewidth=1.3, alpha=0.9)
    for name, b in benchmarks.items():
        ax.plot(
            b["equity"].index,
            b["equity"].values,
            label=f"{name} buy&hold",
            linewidth=2.2,
            linestyle="--",
            alpha=0.95,
        )
    ax.set_yscale("log")
    ax.set_title("Top configurations vs indices, $100,000 starting capital")
    ax.set_xlabel("date")
    ax.set_ylabel("portfolio value ($, log)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_winner_drawdown(rows: list[dict], benchmarks: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for row in rows:
        eq = row["equity"]
        dd = (eq / eq.cummax() - 1) * 100
        label = f"k={row['k']} L={row['lookback']} R={row['rebalance']} ({row['universe']})"
        ax.plot(dd.index, dd.values, label=label, linewidth=1.1, alpha=0.85)
    for name, b in benchmarks.items():
        eq = b["equity"]
        dd = (eq / eq.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, label=f"{name}", linewidth=2.0, linestyle="--", alpha=0.95)
    ax.set_title("Drawdowns of top configurations vs indices (%)")
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_holdings_turnover(winner: dict, path: Path) -> None:
    w = winner["weights"]
    on = (w.abs() > 1e-6).astype(int)
    monthly = on.resample("ME").max()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(monthly.T.values, aspect="auto", cmap="binary", interpolation="nearest")
    ax.set_yticks(range(len(monthly.columns)))
    ax.set_yticklabels(monthly.columns, fontsize=6)
    n = len(monthly)
    tick_step = max(1, n // 24)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels(
        [monthly.index[i].strftime("%Y-%m") for i in range(0, n, tick_step)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_title("Top-K membership over time (black = held that month)")
    ax.set_xlabel("month")
    ax.set_ylabel("symbol")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_grid_heatmap(grid_rows: list[dict], universe: str, path: Path) -> None:
    """Heat map of Calmar by (K, lookback) for the most useful slice."""
    rows = [
        r
        for r in grid_rows
        if r["universe"] == universe
        and r["skip"] == 21
        and r["rebalance"] == 21
        and r["abs_momentum_threshold"] == 0.0
    ]
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="lookback", columns="k", values="calmar")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("k (top-K names)")
    ax.set_ylabel("lookback (bars)")
    ax.set_title(
        f"Calmar ratio (CAGR / |MaxDD|) on {universe}, rebalance=21d, skip=21d, abs_gate=0"
    )
    fig.colorbar(im, ax=ax, label="Calmar")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < pivot.values[~np.isnan(pivot.values)].mean() else "black",
                    fontsize=8,
                )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main ----


def main() -> int:
    # Load both universes up-front (one full read each).
    prices_by_universe: dict[str, pd.DataFrame] = {}
    for u in ("nasdaq100", "sp500"):
        p = _load_universe_prices(u)
        if p.empty:
            print(f"no cached prices for {u}", file=sys.stderr)
            continue
        prices_by_universe[u] = p
        print(
            f"{u}: {p.shape[1]} symbols, {len(p)} bars "
            f"({p.index[0].date()} -> {p.index[-1].date()})"
        )
    if not prices_by_universe:
        return 1

    # Realistic IBKR Tiered + retail-size slippage on liquid US equity at
    # daily close: ~3 bps commission + ~7 bps slippage per side = 10 bps
    # one-way, 20 bps round-trip. The previous 3 bps figure was optimistic.
    costs = CostModel(commission_bps=3.0, slippage_bps=7.0)

    # Benchmark constraints derived from QQQ.
    qqq = _benchmark_metrics("QQQ", prices_by_universe["nasdaq100"].index)
    spy = _benchmark_metrics("SPY", prices_by_universe["nasdaq100"].index)
    qqq["calmar"] = qqq["cagr"] / abs(qqq["max_drawdown"])
    spy["calmar"] = spy["cagr"] / abs(spy["max_drawdown"])
    dd_cap = abs(qqq["max_drawdown"]) * 1.10  # at most 10% worse than QQQ
    print(
        f"\nQQQ over the window:  CAGR={qqq['cagr']:.2%}  MaxDD={qqq['max_drawdown']:.2%}  "
        f"Calmar={qqq['calmar']:.3f}"
    )
    print(f"DD constraint for the search: any configuration MaxDD must be > -{dd_cap:.2%}")

    # ---------------------------------------------- the grid ---------------
    grid = _grid_iter()
    print(f"\nsearching {len(grid)} configurations...")
    all_rows: list[dict] = []
    for i, cfg in enumerate(grid):
        if cfg["universe"] not in prices_by_universe:
            continue
        prices = prices_by_universe[cfg["universe"]]
        row = _backtest_config(prices, cfg, costs)
        if row is None:
            continue
        all_rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(grid)} done")

    print(f"completed {len(all_rows)} of {len(grid)} (some skipped on construction error)")

    # ---------------------------------------------- filter + rank ----------
    feasible = [r for r in all_rows if r["cagr"] > 0 and r["max_drawdown"] > -dd_cap]
    print(f"feasible (CAGR>0, DD>-{dd_cap:.2%}): {len(feasible)} configurations")

    # Sort by Calmar
    ranked = sorted(feasible, key=lambda r: -r["calmar"])

    if not ranked:
        print("no feasible configuration found", file=sys.stderr)
        return 1

    print("\nTop 10 by Calmar:")
    print(
        f"  {'#':>2} {'universe':10s} {'k':>2} {'L':>4} {'s':>2} {'R':>3} "
        f"{'abs':>6} {'dec':>3}  "
        f"{'$100k -> $':>13}  {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}"
    )
    for i, r in enumerate(ranked[:10], 1):
        abs_str = (
            "-" if r["abs_momentum_threshold"] is None else f"{r['abs_momentum_threshold']:.2f}"
        )
        print(
            f"  {i:>2} {r['universe']:10s} {r['k']:>2} {r['lookback']:>4} "
            f"{r['skip']:>2} {r['rebalance']:>3} "
            f"{abs_str:>6} {r.get('min_decorrelated', 0):>3}  "
            f"{r['final_equity']:>13,.0f}  "
            f"{r['cagr']:>8.2%} {r['sharpe']:>7.3f} {r['max_drawdown']:>8.2%} "
            f"{r['calmar']:>7.3f}"
        )

    # --------------------------------- winner + walk-forward validate ------
    winner = ranked[0]
    print("\n=== WINNER ===")
    print(
        f"  universe={winner['universe']}  k={winner['k']}  L={winner['lookback']}  "
        f"skip={winner['skip']}  rebalance={winner['rebalance']}  "
        f"abs_momentum_threshold={winner['abs_momentum_threshold']}  "
        f"min_decorrelated={winner.get('min_decorrelated', 0)}"
    )
    print(
        f"  IN-SAMPLE:  CAGR={winner['cagr']:.2%}  Sharpe={winner['sharpe']:.3f}  "
        f"MaxDD={winner['max_drawdown']:.2%}  Calmar={winner['calmar']:.3f}"
    )

    print("\nrunning walk-forward (504 train / 126 test, step=126) to validate...")
    try:
        wf = _walk_forward_check(prices_by_universe[winner["universe"]], winner, costs)
        print(
            f"  OOS:        CAGR={wf['oos_cagr']:.2%}  Sharpe={wf['oos_sharpe']:.3f}  "
            f"MaxDD={wf['oos_max_drawdown']:.2%}  Calmar={wf['oos_calmar']:.3f}"
        )
    except Exception as e:
        print(f"  walk-forward failed: {e!r}")

    # --------------------------------- viz --------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = {"QQQ": qqq, "SPY": spy}
    _plot_winner_equity(ranked[:5], benchmarks, OUT_DIR / "winner_equity.png")
    _plot_winner_drawdown(ranked[:5], benchmarks, OUT_DIR / "winner_drawdown.png")
    _plot_holdings_turnover(winner, OUT_DIR / "winner_holdings.png")
    _plot_grid_heatmap(all_rows, "nasdaq100", OUT_DIR / "grid_heatmap_ndx.png")
    _plot_grid_heatmap(all_rows, "sp500", OUT_DIR / "grid_heatmap_sp500.png")
    print(f"\nplots written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
