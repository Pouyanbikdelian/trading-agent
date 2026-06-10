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
            # check_same_thread=False is REQUIRED in production: the
            # runner opens the conn from one thread (often the startup
            # reconciliation worker), then the cycle saves snapshots
            # from a DIFFERENT worker thread (asyncio.to_thread). Without
            # this flag SQLite raises ProgrammingError mid-cycle. Safe
            # because we serialise writes through the runner — no real
            # concurrent access — and WAL handles concurrent reads.
            self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if self.path != ":memory:":
                # Audit fix #14: verify WAL actually took effect. WAL can
                # silently fall back (e.g. on a network filesystem) and we'd
                # lose crash-safety guarantees. Log loudly if so.
                actual = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if actual and actual[0].lower() != "wal":
                    from trading.core.logging import logger as _logger

                    _logger.bind(component="runner_store").warning(
                        f"could not enable WAL on {self.path} "
                        f"(journal_mode={actual[0]}); writes are less crash-safe"
                    )
                # synchronous=NORMAL pairs well with WAL: safe durability
                # at a small fraction of FULL's fsync cost.
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate(self._conn)
        return self._conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Idempotent column additions for databases created by older
        versions. ALTER TABLE ADD COLUMN is cheap in SQLite and a no-op
        guard via pragma keeps re-opens fast."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(account_snapshots)")}
        if "base_currency" not in cols:
            conn.execute(
                "ALTER TABLE account_snapshots ADD COLUMN base_currency TEXT NOT NULL DEFAULT 'USD'"
            )
        if "cash_by_currency_json" not in cols:
            conn.execute(
                "ALTER TABLE account_snapshots ADD COLUMN cash_by_currency_json TEXT NOT NULL DEFAULT '{}'"
            )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------- snapshots

    def save_snapshot(self, snap: AccountSnapshot) -> None:
        if snap.ts.tzinfo is None:
            raise ValueError("AccountSnapshot.ts must be timezone-aware")
        self.conn.execute(
            """
            INSERT INTO account_snapshots
                (ts, cash, equity, positions_json, base_currency, cash_by_currency_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snap.ts.timestamp(),
                snap.cash,
                snap.equity,
                _positions_to_json(snap.positions),
                snap.base_currency,
                json.dumps(snap.cash_by_currency or {}),
            ),
        )

    def latest_snapshot(self) -> AccountSnapshot | None:
        row = self.conn.execute(
            "SELECT * FROM account_snapshots ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        keys = row.keys()
        return AccountSnapshot(
            ts=datetime.fromtimestamp(row["ts"], tz=timezone.utc),
            cash=row["cash"],
            equity=row["equity"],
            positions=_positions_from_json(row["positions_json"]),
            base_currency=(row["base_currency"] if "base_currency" in keys else "USD"),
            cash_by_currency=(
                json.loads(row["cash_by_currency_json"]) if "cash_by_currency_json" in keys else {}
            ),
        )

    def day_equity_bounds(self, since: datetime) -> tuple[float, float] | None:
        """(first, last) equity among snapshots at/after ``since``.

        Powers the daily P&L Telegram summary: ``since`` is the start of
        the trading day (UTC). Returns None when fewer than two
        snapshots exist in the window — a fresh deploy or a dead feed
        should yield silence, not a fabricated 0.0% day.
        """
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")
        ts0 = since.timestamp()
        first = self.conn.execute(
            "SELECT equity FROM account_snapshots WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
            (ts0,),
        ).fetchone()
        last = self.conn.execute(
            "SELECT equity FROM account_snapshots WHERE ts >= ? ORDER BY ts DESC LIMIT 1",
            (ts0,),
        ).fetchone()
        if first is None or last is None:
            return None
        n = self.conn.execute(
            "SELECT COUNT(*) AS c FROM account_snapshots WHERE ts >= ?", (ts0,)
        ).fetchone()["c"]
        if n < 2:
            return None
        return float(first["equity"]), float(last["equity"])

    def equity_curve(self) -> list[tuple[datetime, float]]:
        rows = self.conn.execute(
            "SELECT ts, equity FROM account_snapshots ORDER BY ts ASC"
        ).fetchall()
        return [(datetime.fromtimestamp(r["ts"], tz=timezone.utc), r["equity"]) for r in rows]

    # ----------------------------------------------------------- cycles

    def save_cycle(self, report: CycleReport) -> None:
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
