"""Grid-search hyperparameter sweep on a single strategy.

Two modes:

* ``--mode full``      — backtest every combo on the whole window. Fast,
                          but overfits to the in-sample period (every combo
                          is judged on the data it was tuned on).
* ``--mode walkforward`` — train on rolling 2-year windows, test on the
                            next 6 months, compute OOS Sharpe. Honest but
                            slow. Use this before committing to params live.

Output: a leaderboard sorted by Sharpe (full) or OOS Sharpe (walkforward),
with DSR if running walkforward (deflates for the n_trials = grid size).

Usage::

    uv run python scripts/tune_strategy.py donchian us_large_cap \\
        --param lookback=20,40,55,80,120 \\
        --param allow_short=true,false

    uv run python scripts/tune_strategy.py risk_parity nasdaq100 \\
        --param vol_lookback=20,60,120 \\
        --param rebalance=5,21,63 \\
        --mode walkforward
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading.backtest import CostModel, compute_metrics, run_vectorized
from trading.backtest.walkforward import expanding
from trading.core.config import settings
from trading.core.universes import load_universe
from trading.data.cache import ParquetCache
from trading.selection.scores import deflated_sharpe, moments, per_period_sharpe
from trading.strategies import get_strategy


def _parse_grid(specs: list[str]) -> dict[str, list[Any]]:
    """`--param lookback=20,55,80` -> {"lookback": [20, 55, 80]}.

    Values are coerced by pydantic when the strategy is constructed, so we
    keep them as strings here and let the Params model do the work."""
    grid: dict[str, list[Any]] = {}
    for raw in specs:
        if "=" not in raw:
            raise ValueError(f"--param expects key=v1,v2,...  got {raw!r}")
        k, v = raw.split("=", 1)
        values: list[Any] = []
        for item in v.split(","):
            item = item.strip()
            if item.lower() in ("true", "false"):
                values.append(item.lower() == "true")
            else:
                try:
                    values.append(int(item))
                except ValueError:
                    try:
                        values.append(float(item))
                    except ValueError:
                        values.append(item)
        grid[k.strip()] = values
    return grid


def load_prices(
    universe: str, start: datetime, end: datetime, min_frac: float = 0.95
) -> pd.DataFrame:
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
    max_dates = max(len(s) for s in series.values())
    threshold = int(max_dates * min_frac)
    kept = {sym: s for sym, s in series.items() if len(s) >= threshold}
    return pd.DataFrame(kept).sort_index().dropna(how="any")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("strategy", help="Strategy name from STRATEGY_REGISTRY")
    p.add_argument("universe", help="Universe name from config/universes.yaml")
    p.add_argument("--from", dest="start", default="2018-01-01")
    p.add_argument("--to", dest="end", default=datetime.now(tz=timezone.utc).date().isoformat())
    p.add_argument("--mode", choices=["full", "walkforward"], default="full")
    p.add_argument(
        "--param",
        action="append",
        default=[],
        help="Grid spec, e.g. --param lookback=20,55,80. Repeat for multi-axis.",
    )
    p.add_argument("--train", type=int, default=504, help="Walk-forward train window (bars).")
    p.add_argument("--test", type=int, default=126, help="Walk-forward test window (bars).")
    args = p.parse_args(argv)

    grid = _parse_grid(args.param)
    if not grid:
        print("no --param flags given; nothing to tune.", file=sys.stderr)
        return 1

    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"sweeping {len(combos)} combos: {grid}")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    prices = load_prices(args.universe, start, end)
    if prices.empty:
        print(f"no cached prices for {args.universe}", file=sys.stderr)
        return 1
    print(
        f"  {prices.shape[1]} symbols, {len(prices)} bars "
        f"({prices.index[0].date()} -> {prices.index[-1].date()})"
    )

    StratCls = get_strategy(args.strategy)  # noqa: N806 — Cls is conventional for class refs
    costs = CostModel(commission_bps=1.0, slippage_bps=2.0)
    n_symbols = prices.shape[1]

    rows: list[dict] = []
    for combo in combos:
        params = dict(zip(keys, combo, strict=True))
        # Auto-size per-asset weight for trend strategies on multi-symbol universes.
        if "weight_per_asset" not in params and "weight_per_asset" in StratCls.Params.model_fields:
            params["weight_per_asset"] = 1.0 / n_symbols
        try:
            strat = StratCls(**params)
        except Exception as e:
            print(f"  SKIP {params}: {e!r}")
            continue

        if args.mode == "full":
            w = strat.generate(prices)
            result = run_vectorized(prices, w, costs=costs)
            m = compute_metrics(result, periods_per_year=252)
            rows.append({"params": params, **m})
        else:
            # Walk-forward: signal_fn re-instantiates the strategy on each
            # train/test split with the same params. The strategy is
            # deterministic given prices, so train data only matters for
            # warm-up of indicators.
            def make_signal_fn(combo_params: dict) -> Any:
                def signal_fn(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
                    full = pd.concat(
                        [
                            train.iloc[
                                -StratCls.Params()
                                .model_dump()
                                .get(
                                    "regime_sma",
                                    StratCls.Params().model_dump().get("vol_lookback", 252),
                                ) :
                            ],
                            test,
                        ]
                    )
                    p = dict(combo_params)
                    if (
                        "weight_per_asset" not in p
                        and "weight_per_asset" in StratCls.Params.model_fields
                    ):
                        p["weight_per_asset"] = 1.0 / test.shape[1]
                    s = StratCls(**p)
                    return s.generate(full).reindex(test.index).fillna(0.0)

                return signal_fn

            try:
                _, wf_result = expanding(
                    prices,
                    make_signal_fn(params),
                    train_size=args.train,
                    test_size=args.test,
                    costs=costs,
                )
                m = compute_metrics(wf_result, periods_per_year=252)
                # DSR with n_trials = grid size penalizes for the sweep itself.
                sr_period = per_period_sharpe(wf_result.returns)
                sk, kt = moments(wf_result.returns)
                dsr = deflated_sharpe(
                    sr_period, len(wf_result.returns), sk, kt, n_trials=len(combos)
                )
                rows.append({"params": params, "dsr": dsr, **m})
            except Exception as e:
                print(f"  SKIP {params}: {e!r}")
                continue

    if not rows:
        print("no successful runs", file=sys.stderr)
        return 1

    # In walk-forward mode you might prefer DSR-sorting, but Sharpe correlates.
    leaderboard = sorted(rows, key=lambda r: r["sharpe"], reverse=True)

    print(f"\n=== leaderboard ({args.mode}) — top 15 ===")
    cols = ("cagr", "sharpe", "sortino", "max_drawdown", "calmar")
    if args.mode == "walkforward":
        cols = ("cagr", "sharpe", "dsr", "max_drawdown", "calmar")
    print(f"  {'rank':>4} {'params':50s}" + "".join(f" {c:>9}" for c in cols))
    for i, r in enumerate(leaderboard[:15], 1):
        plist = ", ".join(f"{k}={v}" for k, v in r["params"].items())[:48]
        vals = []
        for c in cols:
            val = r.get(c, 0)
            if c in ("cagr", "max_drawdown"):
                vals.append(f"{val:>8.2%}")
            elif c == "dsr":
                vals.append(f"{val:>9.3f}")
            else:
                vals.append(f"{val:>9.3f}")
        print(f"  {i:>4} {plist:50s}" + "".join(f" {v}" for v in vals))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
