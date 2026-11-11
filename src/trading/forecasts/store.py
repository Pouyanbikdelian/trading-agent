"""Forecast ledger: what was predicted, by whom, and what then happened."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.core.clock import UtcClock

DB_NAME = "forecasts.db"
BASELINE = "baseline"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    asof        TEXT NOT NULL,      -- session whose close the forecast used
    symbol      TEXT NOT NULL,
    horizon     INTEGER NOT NULL,   -- sessions ahead
    source      TEXT NOT NULL,      -- 'baseline' or 'llm:<model>'
    prob        REAL NOT NULL,      -- P(return beats the benchmark)
    created_at  TEXT NOT NULL,
    note        TEXT,
    outcome     INTEGER,            -- 1 beat, 0 did not; NULL until graded
    excess      REAL,
    graded_at   TEXT,
    PRIMARY KEY (asof, symbol, horizon, source)
);
CREATE INDEX IF NOT EXISTS forecasts_open ON forecasts (outcome, asof);
CREATE TABLE IF NOT EXISTS llm_batches (
    batch_id     TEXT PRIMARY KEY,
    asof         TEXT NOT NULL,
    model        TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    requests     TEXT NOT NULL,     -- JSON: custom_id -> {lens, symbols}
    state        TEXT NOT NULL,     -- pending | collected | failed
    closed_at    TEXT,
    detail       TEXT
);
"""


def db_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / DB_NAME


def _connect(state_dir: Path | str) -> sqlite3.Connection:
    path = db_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save(state_dir: Path | str, rows: list[dict[str, Any]], *, now: datetime | None = None) -> int:
    """Insert forecasts. A forecast, once written, is never changed.

    ``INSERT OR IGNORE`` on the natural key: re-running the job for the same
    session cannot replace a prediction with a better-informed one.
    """
    stamp = (now or UtcClock().now()).isoformat()
    payload = []
    for row in rows:
        prob = float(row["prob"])
        if not np.isfinite(prob) or not 0.0 < prob < 1.0:
            continue
        payload.append(
            (
                str(row["asof"]),
                str(row["symbol"]).upper(),
                int(row["horizon"]),
                str(row["source"]),
                prob,
                stamp,
                str(row.get("note") or "")[:300] or None,
            )
        )
    with closing(_connect(state_dir)) as conn, conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO forecasts "
            "(asof, symbol, horizon, source, prob, created_at, note) VALUES (?,?,?,?,?,?,?)",
            payload,
        )
        return conn.total_changes - before


def has_forecasts(state_dir: Path | str, asof: str, source: str) -> bool:
    with closing(_connect(state_dir)) as conn:
        row = conn.execute(
            "SELECT 1 FROM forecasts WHERE asof = ? AND source = ? LIMIT 1", (asof, source)
        ).fetchone()
    return row is not None


def grade_due(
    state_dir: Path | str,
    close: pd.DataFrame,
    bench: pd.Series,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Settle every forecast whose horizon has fully passed.

    The outcome bar must be dated before today (UTC): the daily cache can
    hold today's unfinished candle, and grading against it would score a
    forecast on a price that is not yet the close.
    """
    now = now or UtcClock().now()
    today = pd.Timestamp(now).tz_convert("UTC").normalize()
    position = {d: i for i, d in enumerate(close.index)}
    graded = waiting = unpriced = 0
    with closing(_connect(state_dir)) as conn, conn:
        open_rows = conn.execute(
            "SELECT asof, symbol, horizon, source FROM forecasts WHERE outcome IS NULL"
        ).fetchall()
        for row in open_rows:
            start = position.get(pd.Timestamp(row["asof"], tz="UTC"))
            if start is None:
                unpriced += 1
                continue
            end = start + int(row["horizon"])
            if end >= len(close.index) or close.index[end] >= today:
                waiting += 1
                continue
            if row["symbol"] not in close.columns:
                unpriced += 1
                continue
            p0, p1 = close[row["symbol"]].iloc[start], close[row["symbol"]].iloc[end]
            b0, b1 = bench.iloc[start], bench.iloc[end]
            if not all(np.isfinite(v) and v > 0 for v in (p0, p1, b0, b1)):
                unpriced += 1
                continue
            excess = float(p1 / p0 - b1 / b0)
            conn.execute(
                "UPDATE forecasts SET outcome = ?, excess = ?, graded_at = ? "
                "WHERE asof = ? AND symbol = ? AND horizon = ? AND source = ? "
                "AND outcome IS NULL",
                (
                    int(excess > 0),
                    excess,
                    now.isoformat(),
                    row["asof"],
                    row["symbol"],
                    row["horizon"],
                    row["source"],
                ),
            )
            graded += 1
    return {"graded": graded, "waiting": waiting, "unpriced": unpriced}


def _verdict(periods: int, t: float | None, better: str, worse: str) -> str:
    if periods < 8 or t is None:
        return f"too early: {periods} independent period(s), need at least 8"
    if t >= 2.0:
        return better
    if t <= -2.0:
        return worse
    return f"no detectable difference yet ({periods} independent periods)"


def _t(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype="float64")
    if len(arr) < 8 or arr.std(ddof=1) == 0:
        return None
    return float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))


def summary(state_dir: Path | str) -> dict[str, Any]:
    """Skill per forecaster, and each LLM against the baseline on the SAME
    forecasts — same stock, same day, same horizon.

    Sample size is counted in independent periods (one day per ``horizon``
    sessions), not forecasts: fifty stocks on one day mostly rise and fall
    together, and tomorrow's 21-session window is today's shifted by one.
    """
    if not db_path(state_dir).exists():
        return {"sources": [], "paired": [], "open": 0, "total": 0}
    with closing(_connect(state_dir)) as conn:
        frame = pd.read_sql_query("SELECT * FROM forecasts", conn)
    if frame.empty:
        return {"sources": [], "paired": [], "open": 0, "total": 0}
    done = frame[frame["outcome"].notna()].copy()
    done["sq"] = (done["prob"] - done["outcome"]) ** 2
    sources: list[dict[str, Any]] = []
    for (source, horizon), all_rows in frame.groupby(["source", "horizon"]):
        rows = done[(done["source"] == source) & (done["horizon"] == horizon)]
        entry: dict[str, Any] = {
            "source": source,
            "horizon": int(str(horizon)),
            "forecasts": len(all_rows),
            "graded": len(rows),
            "first": str(all_rows["asof"].min()),
            "last": str(all_rows["asof"].max()),
        }
        if len(rows):
            daily = rows.groupby("asof")["sq"].mean().sort_index()
            independent = (0.25 - daily.iloc[:: int(str(horizon))]).tolist()
            entry.update(
                {
                    "hit_rate": round(
                        float(((rows["prob"] > 0.5) == (rows["outcome"] == 1)).mean()), 4
                    ),
                    "base_rate": round(float(rows["outcome"].mean()), 4),
                    "mean_prob": round(float(rows["prob"].mean()), 4),
                    "brier": round(float(rows["sq"].mean()), 5),
                    "skill_vs_coin_flip": round(1 - float(rows["sq"].mean()) / 0.25, 5),
                    "independent_periods": len(independent),
                    "verdict": _verdict(
                        len(independent),
                        _t(independent),
                        "better than a coin flip",
                        "worse than a coin flip",
                    ),
                }
            )
        sources.append(entry)
    paired: list[dict[str, Any]] = []
    key = ["asof", "symbol", "horizon"]
    base = done[done["source"] == BASELINE][[*key, "sq"]].rename(columns={"sq": "sq_base"})
    for source in sorted(set(done["source"]) - {BASELINE}):
        both = done[done["source"] == source][[*key, "sq"]].merge(base, on=key)
        for horizon, rows in both.groupby("horizon"):
            daily = (rows["sq_base"] - rows["sq"]).groupby(rows["asof"]).mean().sort_index()
            independent = daily.iloc[:: int(str(horizon))].tolist()
            paired.append(
                {
                    "source": source,
                    "horizon": int(str(horizon)),
                    "common_forecasts": len(rows),
                    "brier": round(float(rows["sq"].mean()), 5),
                    "brier_baseline": round(float(rows["sq_base"].mean()), 5),
                    "skill_vs_baseline": round(
                        1 - float(rows["sq"].mean()) / float(rows["sq_base"].mean()), 5
                    ),
                    "independent_periods": len(independent),
                    "verdict": _verdict(
                        len(independent),
                        _t(independent),
                        "beats the free baseline",
                        "worse than the free baseline",
                    ),
                }
            )
    return {
        "sources": sources,
        "paired": paired,
        "open": int(frame["outcome"].isna().sum()),
        "total": len(frame),
    }


def add_batch(
    state_dir: Path | str,
    batch_id: str,
    *,
    asof: str,
    model: str,
    requests: dict[str, Any],
    now: datetime | None = None,
) -> None:
    with closing(_connect(state_dir)) as conn, conn:
        conn.execute(
            "INSERT OR IGNORE INTO llm_batches "
            "(batch_id, asof, model, submitted_at, requests, state) VALUES (?,?,?,?,?,'pending')",
            (batch_id, asof, model, (now or UtcClock().now()).isoformat(), json.dumps(requests)),
        )


def batches(
    state_dir: Path | str, *, state: str | None = None, asof: str | None = None
) -> list[dict[str, Any]]:
    if not db_path(state_dir).exists():
        return []
    query = "SELECT * FROM llm_batches WHERE 1=1"
    args: list[str] = []
    if state is not None:
        query, args = query + " AND state = ?", [*args, state]
    if asof is not None:
        query, args = query + " AND asof = ?", [*args, asof]
    with closing(_connect(state_dir)) as conn:
        rows = conn.execute(query + " ORDER BY submitted_at", args).fetchall()
    return [{**dict(r), "requests": json.loads(r["requests"])} for r in rows]


def close_batch(
    state_dir: Path | str, batch_id: str, state: str, detail: str, *, now: datetime | None = None
) -> None:
    with closing(_connect(state_dir)) as conn, conn:
        conn.execute(
            "UPDATE llm_batches SET state = ?, detail = ?, closed_at = ? "
            "WHERE batch_id = ? AND state = 'pending'",
            (state, detail[:300], (now or UtcClock().now()).isoformat(), batch_id),
        )


def status_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / "forecast_shadow_status.json"


def write_status(state_dir: Path | str, status: dict[str, Any]) -> None:
    path = status_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, default=str, indent=1))
    tmp.replace(path)


def read_status(state_dir: Path | str) -> dict[str, Any]:
    try:
        loaded = json.loads(status_path(state_dir).read_text())
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}
