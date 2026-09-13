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
    transaction_cost_summary,
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


# ------------------------------------------------------- transaction costs


def test_legacy_ledger_keeps_missing_commission_currency_explicit(tmp_path: Path) -> None:
    """The pre-currency schema remains readable but cannot be relabelled USD."""
    db = tmp_path / "orders.db"
    _mk_orders_db(db, [("NVDA", "BUY", 1_704_067_200.0, 10, 100.0, 1.25)])

    fills = fills_with_symbols(db)

    assert fills[0]["commission"] == 1.25
    assert fills[0]["commission_currency"] is None


def test_transaction_cost_summary_keeps_currencies_and_legacy_rows_separate() -> None:
    jan_legacy = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
    jan_buy = datetime(2026, 1, 12, tzinfo=timezone.utc).timestamp()
    jan_sell = datetime(2026, 1, 20, tzinfo=timezone.utc).timestamp()
    feb = datetime(2026, 2, 3, tzinfo=timezone.utc).timestamp()
    feb_rebate = datetime(2026, 2, 6, tzinfo=timezone.utc).timestamp()
    summary = transaction_cost_summary(
        [
            {"ts": jan_legacy, "commission": 9.99, "commission_currency": None},
            {"ts": jan_buy, "commission": 1.25, "commission_currency": "USD"},
            {"ts": jan_sell, "commission": 0.75, "commission_currency": "usd"},
            {"ts": feb, "commission": 2.0, "commission_currency": "CHF"},
            {"ts": feb_rebate, "commission": -0.25, "commission_currency": "USD"},
        ]
    )

    assert summary["recorded_from"] == "2026-01-02"
    assert summary["currency_verified_from"] == "2026-01-12"
    assert summary["legacy_execution_count"] == 1
    assert summary["legacy_nonzero_commission_count"] == 1
    assert summary["totals"] == [
        {"currency": "CHF", "amount": 2.0},
        {"currency": "USD", "amount": 1.75},
    ]
    assert summary["months"] == [
        {
            "month": "2026-01",
            "fees": [{"currency": "USD", "amount": 2.0}],
            "cumulative": [{"currency": "USD", "amount": 2.0}],
        },
        {
            "month": "2026-02",
            "fees": [
                {"currency": "CHF", "amount": 2.0},
                {"currency": "USD", "amount": -0.25},
            ],
            "cumulative": [
                {"currency": "CHF", "amount": 2.0},
                {"currency": "USD", "amount": 1.75},
            ],
        },
    ]


def test_transaction_cost_summary_empty_ledger_has_no_coverage() -> None:
    summary = transaction_cost_summary([])

    assert summary["recorded_execution_count"] == 0
    assert summary["recorded_from"] is None
    assert summary["totals"] == []
    assert summary["months"] == []


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


def test_daily_pnl_bars_mask_capital_flows() -> None:
    """A 5x overnight jump is a paper top-up, not a $1.15M trading day."""
    pts = [
        {"t": "d1", "v": 250_000.0},
        {"t": "d2", "v": 1_400_000.0},  # injection
        {"t": "d3", "v": 1_407_000.0},  # real +7k day
    ]
    bars = daily_pnl_bars(pts)
    assert bars[0]["v"] == 0.0 and bars[0]["flow"] == 1_150_000.0
    assert bars[1] == {"t": "d3", "v": 7000.0}


def test_flow_adjusted_return_skips_injections() -> None:
    from trading.dashboard.live import flow_adjusted_return_pct

    pts = [
        {"t": "d1", "v": 250_000.0},
        {"t": "d2", "v": 255_000.0},  # +2%
        {"t": "d3", "v": 1_400_000.0},  # injection — must not count
        {"t": "d4", "v": 1_428_000.0},  # +2%
    ]
    r = flow_adjusted_return_pct(pts)
    assert r is not None and abs(r - 4.04) < 0.01  # 1.02 * 1.02
    assert flow_adjusted_return_pct([{"t": "d1", "v": 1.0}]) is None


def test_flow_adjusted_is_what_the_portfolio_tab_must_use() -> None:
    """Regression for the dashboard reading +979.92%.

    ``drawEq`` took a raw last/first ratio while the Live tab used
    ``flow_adjusted_return_pct`` — the same paper book therefore showed
    +979.92% on one tab and +5.1% on the other, because the curve spans a
    funding event and a deposit is not a return. The JS now mirrors this
    function; this pins the arithmetic it must agree with.
    """
    from trading.dashboard.live import flow_adjusted_return_pct

    pts = [
        {"t": "d1", "v": 123_400.0},
        {"t": "d2", "v": 1_300_000.0},  # the funding event
        {"t": "d3", "v": 1_332_632.0},
    ]
    naive = (pts[-1]["v"] / pts[0]["v"] - 1) * 100
    honest = flow_adjusted_return_pct(pts)
    assert naive > 900  # what the bug reported
    assert honest is not None and honest < 5  # what actually happened


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
