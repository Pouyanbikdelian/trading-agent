"""SQLite persistence for runner state.

What we persist
---------------
* ``account_snapshots`` — one row per cycle, capturing cash + equity + the
  full position map. Lets us reconstruct an equity curve without scraping
  the broker.
* ``cycles`` — one row per cycle attempt, with the outcome (ok / halted /
  error / no_orders), counts of orders submitted and fills received, the
  serialized risk decisions, and any exception text. This is what the
  operator skims when something looked off yesterday.

The order/fill store from Phase 6 stays separate — it's tied to broker
state, while this one is tied to runner state. Different lifecycle, different
file.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Position,
)

if TYPE_CHECKING:  # pragma: no cover
    from trading.runner.cycle import CycleReport


_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    cash        REAL NOT NULL,
    equity      REAL NOT NULL,
    positions_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON account_snapshots(ts);

CREATE TABLE IF NOT EXISTS cycles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  REAL NOT NULL,
    status              TEXT NOT NULL,
    orders_submitted    INTEGER NOT NULL,
    fills_received      INTEGER NOT NULL,
    decisions_json      TEXT NOT NULL,
    error               TEXT,
    duration_ms         REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cycles_ts ON cycles(ts);
"""


def _positions_to_json(positions: dict[str, Position]) -> str:
    """Serialize position map to JSON. We round-trip the full pydantic model
    so deserialization is symmetric — pydantic.model_dump_json on every value."""
    return json.dumps({k: json.loads(v.model_dump_json()) for k, v in positions.items()})


def _positions_from_json(s: str) -> dict[str, Position]:
    raw = json.loads(s)
    out: dict[str, Position] = {}
    for k, payload in raw.items():
        ins_payload = payload["instrument"]
        ins_payload["asset_class"] = AssetClass(ins_payload["asset_class"])
        payload["instrument"] = Instrument(**ins_payload)
        out[k] = Position(**payload)
    return out


class RunnerStore:
    """SQLite persistence for runner state. Idempotent migrations on open."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------- snapshots

    def save_snapshot(self, snap: AccountSnapshot) -> None:
        if snap.ts.tzinfo is None:
            raise ValueError("AccountSnapshot.ts must be timezone-aware")
        self.conn.execute(
            "INSERT INTO account_snapshots (ts, cash, equity, positions_json) VALUES (?, ?, ?, ?)",
            (snap.ts.timestamp(), snap.cash, snap.equity, _positions_to_json(snap.positions)),
        )

    def latest_snapshot(self) -> AccountSnapshot | None:
        row = self.conn.execute(
            "SELECT * FROM account_snapshots ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return AccountSnapshot(
            ts=datetime.fromtimestamp(row["ts"], tz=timezone.utc),
            cash=row["cash"],
            equity=row["equity"],
            positions=_positions_from_json(row["positions_json"]),
        )

    def equity_curve(self) -> list[tuple[datetime, float]]:
        rows = self.conn.execute(
            "SELECT ts, equity FROM account_snapshots ORDER BY ts ASC"
        ).fetchall()
        return [(datetime.fromtimestamp(r["ts"], tz=timezone.utc), r["equity"]) for r in rows]

    # ----------------------------------------------------------- cycles

    def save_cycle(self, report: "CycleReport") -> None:
        self.conn.execute(
            """
            INSERT INTO cycles (ts, status, orders_submitted, fills_received,
                                decisions_json, error, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.ts.timestamp(),
                report.status,
                report.orders_submitted,
                report.fills_received,
                json.dumps([d.model_dump() for d in report.decisions]),
                report.error,
                report.duration_ms,
            ),
        )

    def recent_cycles(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT * FROM cycles ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {
                "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc),
                "status": r["status"],
                "orders_submitted": r["orders_submitted"],
                "fills_received": r["fills_received"],
                "error": r["error"],
                "duration_ms": r["duration_ms"],
            }
            for r in rows
        ]
