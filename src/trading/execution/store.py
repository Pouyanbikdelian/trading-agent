"""SQLite-backed persistence for orders and fills.

The store is the system's source of truth across restarts: the live runner
crashes, the operator restarts it, and reconciliation reads what *we* believe
the state was so we can diff it against what the broker reports. Without
this, a restart would mean either (a) re-submitting the same orders or
(b) forgetting fills we already reacted to.

Schema choices
--------------
* One file per environment (``research.db``, ``paper.db``, ``live.db``).
  Mixing them in one DB is asking for accidents.
* ``client_order_id`` is the PK on ``orders`` — that's the idempotency
  guarantee the Broker Protocol relies on.
* ``ts`` columns store seconds since epoch (REAL) — sqlite has no native
  tz-aware datetime, and storing ISO strings makes range queries slow.
* Schema migrations are CREATE TABLE IF NOT EXISTS — append-only. Breaking
  changes get a new column with a default value, never a destructive ALTER.

Why stdlib ``sqlite3`` and not SQLAlchemy: the schema has 3 tables, no
relationships beyond a single foreign key, and we don't need an ORM to
type-cast our pydantic models — it's faster code review without it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from trading.core.types import (
    AssetClass,
    Fill,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id     TEXT PRIMARY KEY,
    instrument_json     TEXT NOT NULL,
    side                TEXT NOT NULL,
    quantity            REAL NOT NULL,
    order_type          TEXT NOT NULL,
    limit_price         REAL,
    stop_price          REAL,
    tif                 TEXT NOT NULL,
    created_at          REAL NOT NULL,
    status              TEXT NOT NULL,
    broker_order_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL REFERENCES orders(client_order_id),
    ts              REAL NOT NULL,
    quantity        REAL NOT NULL,
    price           REAL NOT NULL,
    commission      REAL NOT NULL DEFAULT 0,
    venue           TEXT
);

CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_ts       ON fills(ts);
"""


def _ts_to_epoch(ts: datetime) -> float:
    if ts.tzinfo is None:
        raise ValueError("datetimes written to the store must be timezone-aware")
    return ts.timestamp()


def _epoch_to_ts(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _instrument_to_json(ins: Instrument) -> str:
    return ins.model_dump_json()


def _instrument_from_json(s: str) -> Instrument:
    raw = json.loads(s)
    raw["asset_class"] = AssetClass(raw["asset_class"])
    return Instrument(**raw)


class OrderStore:
    """SQLite persistence for the order lifecycle.

    Cheap to instantiate — opens (and migrates) the DB lazily so tests can
    use ``:memory:`` paths without setup.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------ wiring

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            # WAL for concurrent reads; only matters for the live runner.
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------ orders

    def save_order(self, order: Order, *, broker_order_id: str | None = None) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO orders (
                client_order_id, instrument_json, side, quantity, order_type,
                limit_price, stop_price, tif, created_at, status, broker_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.client_order_id,
                _instrument_to_json(order.instrument),
                order.side.value,
                order.quantity,
                order.order_type.value,
                order.limit_price,
                order.stop_price,
                order.tif.value,
                _ts_to_epoch(order.created_at),
                OrderStatus.PENDING.value,
                broker_order_id,
            ),
        )

    def update_status(
        self,
        client_order_id: str,
        status: OrderStatus,
        broker_order_id: str | None = None,
    ) -> None:
        if broker_order_id is None:
            self.conn.execute(
                "UPDATE orders SET status = ? WHERE client_order_id = ?",
                (status.value, client_order_id),
            )
        else:
            self.conn.execute(
                "UPDATE orders SET status = ?, broker_order_id = ? WHERE client_order_id = ?",
                (status.value, broker_order_id, client_order_id),
            )

    def load_orders(
        self,
        *,
        status: OrderStatus | None = None,
        since: datetime | None = None,
    ) -> list[tuple[Order, OrderStatus, str | None]]:
        """Return ``[(order, status, broker_order_id), ...]`` matching the filters."""
        q = "SELECT * FROM orders"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(_ts_to_epoch(since))
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at ASC"
        rows = self.conn.execute(q, params).fetchall()
        return [self._row_to_order(r) for r in rows]

    def _row_to_order(self, r: sqlite3.Row) -> tuple[Order, OrderStatus, str | None]:
        order = Order(
            client_order_id=r["client_order_id"],
            instrument=_instrument_from_json(r["instrument_json"]),
            side=Side(r["side"]),
            quantity=r["quantity"],
            order_type=OrderType(r["order_type"]),
            limit_price=r["limit_price"],
            stop_price=r["stop_price"],
            tif=TimeInForce(r["tif"]),
            created_at=_epoch_to_ts(r["created_at"]),
        )
        return order, OrderStatus(r["status"]), r["broker_order_id"]

    # ------------------------------------------------------------ fills

    def save_fill(self, fill: Fill, *, client_order_id: str) -> None:
        self.conn.execute(
            """
            INSERT INTO fills (order_id, ts, quantity, price, commission, venue)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                client_order_id,
                _ts_to_epoch(fill.ts),
                fill.quantity,
                fill.price,
                fill.commission,
                fill.venue,
            ),
        )

    def load_fills(
        self,
        *,
        client_order_id: str | None = None,
        since: datetime | None = None,
    ) -> list[Fill]:
        q = "SELECT * FROM fills"
        clauses: list[str] = []
        params: list[object] = []
        if client_order_id is not None:
            clauses.append("order_id = ?")
            params.append(client_order_id)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(_ts_to_epoch(since))
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY ts ASC"
        rows = self.conn.execute(q, params).fetchall()
        return [
            Fill(
                order_id=r["order_id"],
                ts=_epoch_to_ts(r["ts"]),
                quantity=r["quantity"],
                price=r["price"],
                commission=r["commission"],
                venue=r["venue"],
            )
            for r in rows
        ]
