r"""Style-rotation advisor — "which style has been paying lately?"

Ranks every registered strategy on *trailing* performance (default 3, 6
and 9 month windows) using the local Parquet price cache, and pushes a
Telegram proposal when the leader differs from the strategy currently
deployed. Strictly advisory: like the SPY/VIX and HMM advisors, this
module NEVER touches mode.json, the runner config, or any order path.
The operator decides; the deployment changes only via .env/compose.

Why trailing-window ranking and not full walk-forward: the walk-forward
harness (selection/) answers "which strategy is *robustly* good"; this
module answers the much narrower operational question "which style is
*currently* being paid" — momentum chasing at the meta level, kept
honest by showing all three windows side by side plus a deflated view
(annualized Sharpe), so a one-month fluke doesn't read like an edge.

Debounce: we persist the last proposed leader to ``state/style_advisor.json``
and only re-alert when the leader CHANGES (or on first run). A weekly
cron with a stable leader stays silent.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.core.config import settings
from trading.core.logging import logger

STATE_FILENAME = "style_advisor.json"

#: Strategies that need cross-sectional breadth or pair structure can be
#: evaluated too — anything raising inside generate() is skipped with a
#: log line rather than failing the poll.
DEFAULT_WINDOWS_MONTHS = (3, 6, 9)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _ann_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 20 or float(r.std()) == 0.0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def rank_styles(
    prices: pd.DataFrame,
    *,
    strategy_names: list[str] | None = None,
    windows_months: tuple[int, ...] = DEFAULT_WINDOWS_MONTHS,
    warmup_bars: int = 280,
) -> pd.DataFrame:
    """Backtest each registered strategy over the trailing windows.

    ``prices``: daily close frame (symbols as columns) covering at least
    ``max(windows) + warmup`` history. Strategies generate weights over
    the FULL frame (so indicators are warm), then performance is sliced
    to each trailing window — no lookahead is introduced because every
    strategy's ``generate`` is already causal (enforced by their tests).

    Returns a DataFrame indexed by strategy name with one column per
    window: annualized Sharpe over that trailing slice, plus a
    ``total_return_{m}m`` column for reading convenience.
    """
    from trading.backtest.engine import run_vectorized
    from trading.strategies import base as strat_base

    names = strategy_names or sorted(strat_base.STRATEGY_REGISTRY)
    rows: dict[str, dict[str, float]] = {}
    for name in names:
        cls = strat_base.STRATEGY_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            weights = cls().generate(prices)
            result = run_vectorized(prices, weights)
        except Exception as e:
            logger.bind(component="style_advisor").info(f"skipping {name}: {type(e).__name__}: {e}")
            continue
        row: dict[str, float] = {}
        for m in windows_months:
            bars = int(m * 21)
            sliced = result.returns.iloc[-bars:]
            row[f"sharpe_{m}m"] = _ann_sharpe(sliced)
            row[f"ret_{m}m"] = float((1.0 + sliced).prod() - 1.0)
        rows[name] = row
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index")
    # Blended score: equal-weight mean of the per-window Sharpes. Using
    # Sharpe (not raw return) keeps a low-vol style competitive and
    # makes windows comparable.
    sharpe_cols = [c for c in df.columns if c.startswith("sharpe_")]
    df["score"] = df[sharpe_cols].mean(axis=1)
    return df.sort_values("score", ascending=False)


def _format_proposal(
    table: pd.DataFrame, current: str | None, windows_months: tuple[int, ...]
) -> str:
    leader = str(table.index[0])
    lines = [
        "🔁 *Style rotation check* — trailing "
        + "/".join(f"{m}m" for m in windows_months)
        + " (annualized Sharpe)",
        "```",
        f"{'strategy':<18}" + "".join(f"{f'{m}m':>7}" for m in windows_months) + f"{'score':>8}",
    ]
    for name, row in table.head(6).iterrows():
        mark = "→" if name == leader else " "
        cells = "".join(f"{row[f'sharpe_{m}m']:>7.2f}" for m in windows_months)
        lines.append(f"{mark}{name:<17}{cells}{row['score']:>8.2f}")
    lines.append("```")
    if current and leader != current:
        lines.append(
            f"Deployed: `{current}` — trailing leader: `{leader}`.\n"
            f"_Advisory only. To switch: set `STRATEGY={leader}` in .env and "
            "`docker compose up -d trader`. Consider the walk-forward OOS "
            "ranking (`trading select ...`) before acting on a hot hand._"
        )
    elif current:
        lines.append(f"_Deployed `{current}` is still the trailing leader. No action._")
    return "\n".join(lines)


async def poll_and_alert(
    *,
    prices: pd.DataFrame,
    current_strategy: str | None = None,
    state_path: Path | None = None,
    windows_months: tuple[int, ...] = DEFAULT_WINDOWS_MONTHS,
) -> dict[str, Any]:
    """One poll: rank styles, alert if the leader changed since last poll
    (or if the leader differs from the deployed strategy on first run)."""
    state_path = state_path or (settings.state_dir / STATE_FILENAME)
    table = rank_styles(prices, windows_months=windows_months)
    if table.empty:
        logger.bind(component="style_advisor").info("no strategies rankable; skipping")
        return {"ranked": 0, "alert_sent": False}

    leader = str(table.index[0])
    prior = _read_state(state_path)
    prior_leader = prior.get("leader")

    should_alert = leader != prior_leader or (
        prior_leader is None and current_strategy and leader != current_strategy
    )
    sent = False
    if should_alert:
        sent = await _send_telegram(_format_proposal(table, current_strategy, windows_months))

    _write_state(
        state_path,
        {
            "leader": leader,
            "scores": {str(k): float(v) for k, v in table["score"].items()},
            "last_polled_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return {"ranked": len(table), "leader": leader, "alert_sent": bool(sent)}


async def _send_telegram(text: str) -> bool:
    try:
        from trading.bot.notifier import send_message
    except Exception:
        logger.warning("style_advisor: cannot import telegram notifier; alert dropped")
        return False
    return await send_message(text)
