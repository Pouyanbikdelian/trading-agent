"""Live-tab data layer — hermetic: fixture SQLite files, no network."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading.dashboard.live import (
    attribution_today,
    convert_curve_to_usd,
    daily_curve,
    daily_pnl_bars,
    fills_with_symbols,
    realized_by_symbol,
)

# ------------------------------------------------------------ fixtures

_ORDERS_SCHEMA = """
CREATE TABLE orders (
    client_order_id TEXT PRIMARY KEY, instrument_json TEXT NOT NULL,
    side TEXT NOT NULL, quantity REAL NOT NULL, order_type TEXT NOT NULL,
    limit_price REAL, stop_price REAL, tif TEXT NOT NULL,
    created_at REAL NOT NULL, status TEXT NOT NULL, broker_order_id TEXT);
CREATE TABLE fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL, ts REAL NOT NULL, quantity REAL NOT NULL,
    price REAL NOT NULL, commission REAL NOT NULL DEFAULT 0, venue TEXT);
"""

_SNAPSHOT_SCHEMA = """
CREATE TABLE account_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
    cash REAL NOT NULL, equity REAL NOT NULL, positions_json TEXT NOT NULL);
"""


def _mk_orders_db(path: Path, fills: list[tuple[str, str, float, float, float, float]]) -> None:
    """fills: (symbol, side, ts_epoch, qty, price, commission)."""
    conn = sqlite3.connect(path)
    conn.executescript(_ORDERS_SCHEMA)
    for i, (sym, side, ts, qty, px, fee) in enumerate(fills):
        oid = f"o{i}"
        ins = json.dumps({"symbol": sym, "asset_class": "equity", "currency": "USD"})
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (oid, ins, side, qty, "MARKET", None, None, "DAY", ts, "FILLED", None),
        )
        conn.execute(
            "INSERT INTO fills (order_id, ts, quantity, price, commission) VALUES (?,?,?,?,?)",
            (oid, ts, qty, px, fee),
        )
    conn.commit()
    conn.close()


def _mk_runner_db(path: Path, snaps: list[tuple[float, dict[str, dict]]]) -> None:
    """snaps: (ts_epoch, {symbol: {quantity, avg_price, unrealized_pnl, realized_pnl}})."""
    conn = sqlite3.connect(path)
    conn.executescript(_SNAPSHOT_SCHEMA)
    for ts, positions in snaps:
        pj = json.dumps(
            {
                s: {
                    "instrument": {"symbol": s, "asset_class": "equity", "currency": "USD"},
                    **p,
                }
                for s, p in positions.items()
            }
        )
        conn.execute(
            "INSERT INTO account_snapshots (ts, cash, equity, positions_json) VALUES (?,?,?,?)",
            (ts, 1000.0, 2000.0, pj),
        )
    conn.commit()
    conn.close()


# ------------------------------------------------------------ realized


def test_realized_by_symbol_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "orders.db"
    _mk_orders_db(
        db,
        [
            ("NVDA", "BUY", 1000.0, 10, 100.0, 1.0),
            ("NVDA", "BUY", 2000.0, 10, 120.0, 1.0),  # avg -> 110
            ("NVDA", "SELL", 3000.0, 15, 130.0, 1.5),  # realized 15*(130-110)=300
            ("AAPL", "BUY", 4000.0, 5, 200.0, 0.5),  # still open, no realized
        ],
    )
    fills = fills_with_symbols(db)
    assert [f["symbol"] for f in fills] == ["NVDA", "NVDA", "NVDA", "AAPL"]
    r = realized_by_symbol(fills)
    assert r["NVDA"]["realized"] == 300.0
    assert r["NVDA"]["fees"] == 3.5
    assert r["AAPL"]["realized"] == 0.0


def test_realized_missing_db_is_empty(tmp_path: Path) -> None:
    assert fills_with_symbols(tmp_path / "nope.db") == []


# ------------------------------------------------------------ fx / curves


def test_convert_curve_carries_rate_forward_and_drops_unknown_head() -> None:
    fx = {"2026-01-02": 0.80, "2026-01-05": 0.90}
    pts = [
        {"t": "2026-01-01", "v": 800.0},  # before first rate: dropped, not guessed
        {"t": "2026-01-02", "v": 800.0},  # /0.80 = 1000
        {"t": "2026-01-03", "v": 800.0},  # weekend: carry 0.80
        {"t": "2026-01-05", "v": 900.0},  # /0.90 = 1000
    ]
    out = convert_curve_to_usd(pts, fx)
    assert [p["v"] for p in out] == [1000.0, 1000.0, 1000.0]


def test_convert_without_fx_returns_input() -> None:
    pts = [{"t": "2026-01-01", "v": 5.0}]
    assert convert_curve_to_usd(pts, {}) == pts


def test_daily_curve_excludes_today() -> None:
    now = datetime.now(tz=timezone.utc)
    curve = [
        (now - timedelta(days=2), 100.0),
        (now - timedelta(days=1), 110.0),
        (now, 90.0),  # intraday snapshot — must not appear
    ]
    pts = daily_curve(curve)
    assert [p["v"] for p in pts] == [100.0, 110.0]
    assert all(p["t"] != now.date().isoformat() for p in pts)


def test_daily_pnl_bars_are_diffs() -> None:
    pts = [{"t": "d1", "v": 100.0}, {"t": "d2", "v": 110.0}, {"t": "d3", "v": 104.0}]
    assert daily_pnl_bars(pts) == [{"t": "d2", "v": 10.0}, {"t": "d3", "v": -6.0}]


# ------------------------------------------------------------ attribution


def test_attribution_today_unrealized_delta_and_closed_position(tmp_path: Path) -> None:
    now = datetime.now(tz=timezone.utc)
    yesterday = now - timedelta(days=1)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    runner_db = tmp_path / "runner.db"
    _mk_runner_db(
        runner_db,
        [
            # Yesterday: NVDA +50 unrealized, TSLA +20 unrealized.
            (
                yesterday.timestamp(),
                {
                    "NVDA": {"quantity": 10, "avg_price": 100, "unrealized_pnl": 50.0},
                    "TSLA": {"quantity": 5, "avg_price": 200, "unrealized_pnl": 20.0},
                },
            ),
            # Today: NVDA up to +80; TSLA gone (closed today).
            (
                now.timestamp(),
                {"NVDA": {"quantity": 10, "avg_price": 100, "unrealized_pnl": 80.0}},
            ),
        ],
    )
    orders_db = tmp_path / "orders.db"
    _mk_orders_db(
        orders_db,
        [
            ("TSLA", "BUY", midnight - 90_000, 5, 200.0, 1.0),  # opened before today
            ("TSLA", "SELL", midnight + 3_600, 5, 206.0, 1.0),  # closed today: realized +30
        ],
    )
    rows = attribution_today(runner_db, fills_with_symbols(orders_db))
    by = {r["symbol"]: r for r in rows}
    assert by["NVDA"]["pnl"] == 30.0  # 80 - 50
    # TSLA: realized today +30, minus yesterday's +20 already-counted paper gain.
    assert by["TSLA"]["pnl"] == 10.0
    assert by["TSLA"]["fees"] == 1.0  # only today's commission


def test_attribution_without_today_snapshot_is_empty(tmp_path: Path) -> None:
    runner_db = tmp_path / "runner.db"
    old = datetime.now(tz=timezone.utc) - timedelta(days=3)
    _mk_runner_db(runner_db, [(old.timestamp(), {})])
    assert attribution_today(runner_db, []) == []
