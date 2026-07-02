"""Guard-layer parameter study: sweep trail params, cycle cadence, TP, and
trailing-stop *formulas* over the live top_k_momentum config.

Three stop modes:
  live     — distance fixed at entry (what runs in production today)
  dynamic  — distance re-measured daily from current 14d vol; the stop
             LEVEL only ever rises, so vol compression tightens the trail
             and vol expansion never loosens an already-won level
  ratchet  — entry-measured distance that shrinks as the position's gain
             grows: dist_t = dist_entry * max(floor, 1 - tighten * gain);
             breathes early, locks in late

Tune on 2019-01..2022-12, validate on 2023-01..2026-05 — parameters are
picked on the tune window only; the validate column is the honest number.

Usage: uv run python scripts/guard_sweep.py [stage_a|stage_b|adaptive]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from trading.backtest import metrics as bt_metrics  # noqa: E402
from trading.strategies.top_k_momentum import TopKMomentum  # noqa: E402

TUNE = ("2019-01", "2022-12")
VALID = ("2023-01", "2026-05")
COST_RATE = 3.0 / 1e4  # 1bp commission + 2bp slippage, both sides
_VOL_LB = 14


def load_prices() -> pd.DataFrame:
    series = {}
    for p in sorted(Path("data/parquet/equity").glob("*/1d.parquet")):
        try:
            df = pd.read_parquet(p)
            s = df["adj_close"].dropna()
        except Exception:
            continue
        if (
            len(s)
            and str(s.index[0])[:10] <= "2018-07-01"
            and str(s.index[-1])[:10] >= "2026-05-01"
        ):
            series[p.parent.name] = s
    px = pd.DataFrame(series).sort_index().loc["2018-01-01":]
    return px.dropna(axis=1, thresh=int(len(px) * 0.98)).ffill().dropna(how="any")


def simulate(
    px: pd.DataFrame,
    tgt: pd.DataFrame,
    *,
    mode: str = "live",
    atr_mult: float = 3.0,
    lo: float = 8.0,
    hi: float = 20.0,
    tp_pct: float | None = None,
    cycle: int = 5,
    ratchet_floor: float = 0.4,
    ratchet_tighten: float = 1.2,
) -> pd.Series:
    """Daily returns of the guarded book. Generalization of
    trading.backtest.guards_overlay with selectable stop formulas."""
    ret = px.pct_change().fillna(0.0).to_numpy()
    vol = (px.pct_change().abs().rolling(_VOL_LB).mean() * 100.0).to_numpy()
    p_np, t_np = px.to_numpy(), tgt.to_numpy()
    n_t, n_n = p_np.shape

    w = np.zeros(n_n)
    entry = np.full(n_n, np.nan)
    hwm = np.full(n_n, np.nan)
    dist0 = np.full(n_n, np.nan)  # entry-time distance
    stop_lvl = np.full(n_n, np.nan)  # monotone trail level
    daily = np.zeros(n_t)

    def clamp(v: float) -> float:
        return float(np.clip(v, lo, hi))

    def arm(j: int, t: int) -> None:
        entry[j] = hwm[j] = p_np[t, j]
        v = vol[t, j]
        dist0[j] = lo if not np.isfinite(v) else clamp(atr_mult * v)
        stop_lvl[j] = p_np[t, j] * (1.0 - dist0[j] / 100.0)

    for j in np.flatnonzero(t_np[0] > 0):
        w[j] = t_np[0, j]
        arm(j, 0)
    daily[0] = -float(np.abs(w).sum()) * COST_RATE

    for t in range(1, n_t):
        gross = float(w @ ret[t])
        turnover = 0.0
        for j in np.flatnonzero(w > 0):
            p = p_np[t, j]
            hwm[j] = max(hwm[j], p)
            if mode == "live":
                dist = dist0[j]
            elif mode == "dynamic":
                v = vol[t, j]
                dist = dist0[j] if not np.isfinite(v) else clamp(atr_mult * v)
            else:  # ratchet
                gain = max(0.0, hwm[j] / entry[j] - 1.0)
                dist = dist0[j] * max(ratchet_floor, 1.0 - ratchet_tighten * gain)
            stop_lvl[j] = max(stop_lvl[j], hwm[j] * (1.0 - dist / 100.0))
            hit_tp = tp_pct is not None and p >= entry[j] * (1.0 + tp_pct / 100.0)
            if p <= stop_lvl[j] or hit_tp:
                turnover += w[j]
                w[j] = 0.0
                entry[j] = hwm[j] = dist0[j] = stop_lvl[j] = np.nan
        if t % cycle == 0:
            desired = t_np[t]
            for j in range(n_n):
                if desired[j] != w[j]:
                    if desired[j] > 0 and w[j] == 0.0:
                        arm(j, t)
                    elif desired[j] == 0.0:
                        entry[j] = hwm[j] = dist0[j] = stop_lvl[j] = np.nan
                    turnover += abs(desired[j] - w[j])
                    w[j] = desired[j]
        daily[t] = gross - turnover * COST_RATE

    return pd.Series(daily, index=px.index)


def window_stats(rets: pd.Series, a: str, b: str) -> tuple[float, float, float]:
    r = rets.loc[a:b]
    eq = (1.0 + r).cumprod()
    return (
        100 * bt_metrics.cagr(eq, 252),
        bt_metrics.sharpe(r, 252),
        100 * bt_metrics.max_drawdown(eq),
    )


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage_a"
    px = load_prices()
    tgt = (
        TopKMomentum(TopKMomentum.Params(k=8, lookback=126, skip=21, rebalance=63))
        .generate(px)
        .reindex(px.index)
        .fillna(0.0)
    )
    print(f"{px.shape[1]} symbols, {px.shape[0]} bars — stage {stage}", flush=True)

    if stage == "stage_a":
        grid = [
            dict(atr_mult=m, lo=lo, hi=hi, cycle=c, tp_pct=tp)
            for m in (2.0, 2.5, 3.0, 4.0)
            for lo, hi in ((6.0, 15.0), (8.0, 20.0), (10.0, 25.0))
            for c in (3, 5, 10)
            for tp in (None,)
        ]
    elif stage == "stage_b":
        # finalists from stage A are passed inline below after inspection
        grid = [eval(x) for x in sys.argv[2:]]  # operator tool; input is ours
    else:  # adaptive
        grid = [
            dict(mode=mode, atr_mult=m, lo=lo, hi=hi, cycle=5, **extra)
            for mode in ("dynamic", "ratchet")
            for m in (2.5, 3.0, 4.0)
            for lo, hi in ((6.0, 15.0), (8.0, 20.0), (10.0, 25.0))
            for extra in (
                [{}]
                if mode == "dynamic"
                else [
                    {"ratchet_floor": 0.4, "ratchet_tighten": 1.2},
                    {"ratchet_floor": 0.3, "ratchet_tighten": 2.0},
                ]
            )
        ]

    rows = []
    for g in grid:
        rets = simulate(px, tgt, **g)
        tc, ts, tm = window_stats(rets, *TUNE)
        vc, vs, vm = window_stats(rets, *VALID)
        rows.append(
            {
                **{k: (v if v is not None else "-") for k, v in g.items()},
                "tune_cagr": round(tc, 1),
                "tune_sharpe": round(ts, 2),
                "tune_mdd": round(tm, 1),
                "val_cagr": round(vc, 1),
                "val_sharpe": round(vs, 2),
                "val_mdd": round(vm, 1),
            }
        )
        print(".", end="", flush=True)
    df = pd.DataFrame(rows).sort_values("tune_sharpe", ascending=False)
    out = Path(f"/tmp/guard_sweep_{stage}.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(f"\nsaved {out}\ntop 12 by TUNE sharpe (validate columns are the honest ones):")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
