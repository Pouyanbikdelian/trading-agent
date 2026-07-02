"""Guard-aware backtest: the live trailing-stop/TP layer over strategy weights.

The vectorized engine answers "what does the *signal* earn"; this module
answers "what does the *system* earn" by replaying the runtime guard
semantics from ``trading.runtime.guards`` path-dependently:

* stop distance is fixed at entry: ``clamp(atr_mult × mean(|daily move|,
  14d) × 100, trail_min, trail_max)`` percent — same formula as
  ``guards._stop_distance_pct``;
* each position trails its high-water mark; close ≤ HWM × (1 − dist)
  exits the whole position at that day's close;
* optional static take-profit at ``tp_pct`` above entry;
* the weekly cycle (every ``cycle_every`` bars ≈ the Friday cron)
  re-applies the strategy's current target row — which re-buys a
  guard-exited name with a fresh HWM and freshly measured stop distance,
  exactly like the live runner does.

Known simplifications, disclosed rather than hidden: guards fire on
daily closes (live checks intraday), fills are at close with the same
cost model as the engine, and holds/committee/risk-manager interventions
are not replayed. Positions are assumed long-only (the live guard layer
only handles qty > 0 books).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from trading.backtest.costs import CostModel

_VOL_LOOKBACK = 14  # sessions; mirrors guards._vol_pct


class GuardedResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    equity: pd.Series
    returns: pd.Series
    stop_exits: int
    tp_exits: int
    reentries: int


def run_with_guards(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    costs: CostModel | None = None,
    atr_mult: float = 3.0,
    trail_min_pct: float = 8.0,
    trail_max_pct: float = 20.0,
    tp_pct: float | None = None,
    cycle_every: int = 5,
    ratchet_floor: float | None = None,
    ratchet_tighten: float | None = None,
    initial_equity: float = 1.0,
) -> GuardedResult:
    """Replay strategy targets through the guard layer, day by day.

    ``weights`` is the engine-style target matrix (rebalance rows carried
    forward). Targets are only *applied* on cycle bars — off-cycle, the
    book drifts with whatever the guards left standing.
    """
    if costs is None:
        costs = CostModel()
    common = prices.columns.intersection(weights.columns)
    px = prices[common].astype(float)
    tgt = weights[common].reindex(px.index).fillna(0.0).astype(float)
    if px.isna().any().any():
        raise ValueError("prices contain NaN; forward-fill before simulating")

    ret = px.pct_change().fillna(0.0).to_numpy()
    # Entry-day stop distance source: trailing mean |move| in percent.
    vol_pct = (px.pct_change().abs().rolling(_VOL_LOOKBACK).mean() * 100.0).to_numpy()
    px_np = px.to_numpy()
    tgt_np = tgt.to_numpy()
    n_t, n_n = px_np.shape

    cost_rate = (costs.commission_bps + costs.slippage_bps) / 1e4
    # Same activation rule as guards.check_guards: both knobs, sensible.
    ratchet_on = (
        ratchet_floor is not None
        and ratchet_tighten is not None
        and 0.0 < ratchet_floor < 1.0
        and ratchet_tighten > 0.0
    )
    w = np.zeros(n_n)  # current weights (post-guard)
    entry = np.full(n_n, np.nan)  # entry price per held name
    hwm = np.full(n_n, np.nan)
    stop_pct = np.full(n_n, np.nan)
    stop_lvl = np.full(n_n, np.nan)  # monotone published stop level

    daily = np.zeros(n_t)
    stop_exits = tp_exits = reentries = 0

    def _arm(j: int, t: int) -> None:
        """Fresh guard state for a (re-)entered position — guards.py drops
        state when a position disappears, so re-entries start clean."""
        entry[j] = px_np[t, j]
        hwm[j] = px_np[t, j]
        v = vol_pct[t, j]
        stop_pct[j] = (
            trail_min_pct
            if not np.isfinite(v)
            else float(np.clip(atr_mult * v, trail_min_pct, trail_max_pct))
        )
        stop_lvl[j] = px_np[t, j] * (1.0 - stop_pct[j] / 100.0)

    # Day 0: the runner starts with the strategy's targets applied.
    for j in np.flatnonzero(tgt_np[0] > 0):
        w[j] = tgt_np[0, j]
        _arm(j, 0)
        reentries += 1
    daily[0] = -float(np.abs(w).sum()) * cost_rate  # entry costs

    for t in range(1, n_t):
        # 1. Mark: portfolio return from yesterday's book.
        gross = float(w @ ret[t])
        turnover = 0.0

        # 2. Guard pass on today's close.
        held = np.flatnonzero(w > 0)
        for j in held:
            p = px_np[t, j]
            hwm[j] = max(hwm[j], p)
            eff = stop_pct[j]
            if ratchet_on:
                gain = max(0.0, hwm[j] / entry[j] - 1.0)
                eff *= max(float(ratchet_floor or 0.0), 1.0 - float(ratchet_tighten or 0.0) * gain)
            stop_lvl[j] = max(stop_lvl[j], hwm[j] * (1.0 - eff / 100.0))
            hit_stop = p <= stop_lvl[j]
            hit_tp = tp_pct is not None and p >= entry[j] * (1.0 + tp_pct / 100.0)
            if hit_stop or hit_tp:
                turnover += w[j]
                w[j] = 0.0
                entry[j] = hwm[j] = stop_pct[j] = stop_lvl[j] = np.nan
                if hit_stop:
                    stop_exits += 1
                else:
                    tp_exits += 1

        # 3. Cycle bar: re-apply the strategy's current targets (the weekly
        #    cron). This is where guard-exited names come back if the
        #    strategy still wants them.
        if t % cycle_every == 0:
            desired = tgt_np[t]
            for j in range(n_n):
                if desired[j] != w[j]:
                    if desired[j] > 0 and w[j] == 0.0:
                        reentries += 1
                        _arm(j, t)
                    elif desired[j] == 0.0:
                        entry[j] = hwm[j] = stop_pct[j] = stop_lvl[j] = np.nan
                    turnover += abs(desired[j] - w[j])
                    w[j] = desired[j]

        daily[t] = gross - turnover * cost_rate

    returns = pd.Series(daily, index=px.index)
    equity = initial_equity * (1.0 + returns).cumprod()
    return GuardedResult(
        equity=equity,
        returns=returns,
        stop_exits=stop_exits,
        tp_exits=tp_exits,
        reentries=reentries,
    )
