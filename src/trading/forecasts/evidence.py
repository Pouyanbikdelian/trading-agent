"""The forecasts as PM evidence. Off unless FORECAST_PM_EVIDENCE is set.

Operator's decision, 2026-11-11: let the forecasts inform what the PM
proposes now, without waiting for them to earn a track record, and judge
the result on the approval card. Recorded here because it was a choice and
not an oversight: at the time the mechanical model had tested as a coin
flip over eight years and the LLM forecasters had never been graded.

What that buys and what it does not:

* The PM sees every forecast next to the forecasters' live track record, so
  it is told in the same breath how much each number has earned.
* A forecast may break a tie or prompt a closer look. It may not open a
  position by itself and it never sets a size — sizing by an unproven
  number is the rule that was removed from the charter the same day.
* Once the forecasts steer picks, "did the PM add value" and "did the
  forecasts add value" stop being separable from live results alone. The
  shadow ledger keeps grading every forecast regardless, which is why this
  can be switched on without losing the measurement.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from trading.forecasts import store

#: A forecast older than this many calendar days is history, not evidence.
MAX_AGE_DAYS = 5

HOW_TO_READ = (
    "Probability that the stock's return beats SPY over the next 5 and 21 sessions, as of "
    "the stated close. 'baseline' is a mechanical model on price features; 'llm_numbers' "
    "and 'llm_news' are a language model given the same numbers, without and with "
    "headlines. These are UNPROVEN: read track_record before using any of them. 0.50 "
    "means no view; most values sit within a few points of it, and a difference of two "
    "or three points between stocks is noise."
)


def enabled() -> bool:
    try:
        from trading.core.config import settings

        return bool(settings.forecast_pm_evidence)
    except Exception:
        return False


def _label(source: str) -> str:
    if source == store.BASELINE:
        return "baseline"
    return "llm_" + source.rsplit(":", 1)[-1]


def _track_record(state_dir: Path) -> list[dict[str, Any]]:
    summary = store.read_status(state_dir).get("summary") or {}
    paired = {(p["source"], p["horizon"]): p for p in summary.get("paired") or []}
    out = []
    for row in summary.get("sources") or []:
        pair = paired.get((row["source"], row["horizon"]), {})
        out.append(
            {
                "forecaster": _label(str(row["source"])),
                "horizon_sessions": row["horizon"],
                "graded": row.get("graded", 0),
                "skill_vs_coin_flip": row.get("skill_vs_coin_flip"),
                "skill_vs_baseline": pair.get("skill_vs_baseline"),
                "verdict": pair.get("verdict") or row.get("verdict") or "no outcomes yet",
            }
        )
    return out


def pm_evidence(
    state_dir: Path | str, symbols: set[str], *, now: datetime
) -> dict[str, Any] | None:
    """Latest forecasts for ``symbols``, or None when off, absent or stale."""
    state_dir = Path(state_dir)
    if not enabled() or not symbols or not store.db_path(state_dir).exists():
        return None
    wanted = sorted({s.upper() for s in symbols})
    with closing(sqlite3.connect(store.db_path(state_dir), timeout=30)) as conn:
        latest = conn.execute(
            "SELECT MAX(asof) FROM forecasts WHERE source = ?", (store.BASELINE,)
        ).fetchone()[0]
        if not latest:
            return None
        age = (now.date() - datetime.fromisoformat(latest).date()).days
        if age < 0 or age > MAX_AGE_DAYS:
            return None
        marks = ",".join("?" * len(wanted))
        rows = conn.execute(
            f"SELECT symbol, source, horizon, prob FROM forecasts "
            f"WHERE asof = ? AND symbol IN ({marks})",
            (latest, *wanted),
        ).fetchall()
    forecasts: dict[str, dict[str, float]] = {}
    for symbol, source, horizon, prob in rows:
        forecasts.setdefault(symbol, {})[f"{_label(source)}_p{int(horizon)}"] = round(
            float(prob), 3
        )
    if not forecasts:
        return None
    return {
        "as_of_close": latest,
        "how_to_read": HOW_TO_READ,
        "track_record": _track_record(state_dir),
        "forecasts": forecasts,
    }
