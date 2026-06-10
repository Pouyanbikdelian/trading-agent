"""RunnerStore tests — snapshots, equity curve, cycle history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Position,
    RiskDecision,
)
from trading.runner import CycleReport, RunnerStore


@pytest.fixture
def store() -> RunnerStore:
    return RunnerStore(":memory:")


@pytest.fixture
def aapl() -> Instrument:
    return Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)


@pytest.fixture
def snapshot(aapl: Instrument) -> AccountSnapshot:
    return AccountSnapshot(
        ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
        cash=98_500.0,
        equity=100_000.0,
        positions={
            "equity:AAPL": Position(
                instrument=aapl,
                quantity=10.0,
                avg_price=150.0,
                realized_pnl=5.0,
                unrealized_pnl=0.0,
            ),
        },
    )


def test_save_and_read_back_latest(store: RunnerStore, snapshot: AccountSnapshot) -> None:
    store.save_snapshot(snapshot)
    out = store.latest_snapshot()
    assert out is not None
    assert out.cash == pytest.approx(98_500.0)
    assert out.equity == pytest.approx(100_000.0)
    pos = out.positions["equity:AAPL"]
    assert pos.quantity == 10.0
    assert pos.avg_price == 150.0
    assert pos.instrument.symbol == "AAPL"


def test_latest_when_empty(store: RunnerStore) -> None:
    assert store.latest_snapshot() is None


def test_latest_returns_most_recent(store: RunnerStore, snapshot: AccountSnapshot) -> None:
    later = snapshot.model_copy(
        update={
            "ts": snapshot.ts + timedelta(days=1),
            "equity": 102_000.0,
        }
    )
    store.save_snapshot(snapshot)
    store.save_snapshot(later)
    out = store.latest_snapshot()
    assert out is not None
    assert out.equity == 102_000.0


def test_equity_curve_ordered_ascending(store: RunnerStore, snapshot: AccountSnapshot) -> None:
    store.save_snapshot(
        snapshot.model_copy(
            update={
                "ts": snapshot.ts + timedelta(days=2),
                "equity": 101_000.0,
            }
        )
    )
    store.save_snapshot(snapshot)
    curve = store.equity_curve()
    assert [e for _, e in curve] == [100_000.0, 101_000.0]


def test_save_cycle_round_trips(store: RunnerStore) -> None:
    report = CycleReport(
        ts=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status="ok",
        orders_submitted=3,
        fills_received=3,
        decisions=[
            RiskDecision(action="scale", reason="per-position cap", scale_factor=0.5),
            RiskDecision(action="allow", reason="generated 3 orders"),
        ],
        duration_ms=42.0,
    )
    store.save_cycle(report)
    rows = store.recent_cycles()
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["orders_submitted"] == 3
    assert rows[0]["duration_ms"] == 42.0


def test_recent_cycles_limit_and_order(store: RunnerStore) -> None:
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        store.save_cycle(
            CycleReport(
                ts=base_ts + timedelta(hours=i),
                status="ok",
                orders_submitted=i,
                fills_received=0,
                decisions=[],
            )
        )
    out = store.recent_cycles(limit=3)
    assert [r["orders_submitted"] for r in out] == [4, 3, 2]  # DESC by ts


def test_save_rejects_naive_datetime(store: RunnerStore, aapl: Instrument) -> None:
    naive_snap = AccountSnapshot.model_construct(
        ts=datetime(2024, 1, 1),  # bypass pydantic's validator
        cash=0.0,
        equity=0.0,
        positions={},
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        store.save_snapshot(naive_snap)


def test_snapshot_round_trips_currency_fields(store: RunnerStore) -> None:
    """Regression: June 2026 — save_snapshot dropped base_currency and
    cash_by_currency, so /balances could never show the per-currency
    split and the operator was blind to the CHF/USD margin standoff."""
    snap = AccountSnapshot(
        ts=datetime(2026, 6, 9, 22, 0, tzinfo=timezone.utc),
        cash=1_023_000.0,
        equity=1_024_000.0,
        base_currency="CHF",
        cash_by_currency={"CHF": 815_000.0, "USD": 208_000.0},
    )
    store.save_snapshot(snap)
    out = store.latest_snapshot()
    assert out is not None
    assert out.base_currency == "CHF"
    assert out.cash_by_currency == {
        "CHF": pytest.approx(815_000.0),
        "USD": pytest.approx(208_000.0),
    }


def test_migration_tolerates_pre_currency_rows(tmp_path) -> None:
    """Old databases (rows written before the currency columns existed)
    must read back with defaults instead of raising."""
    import sqlite3

    db = tmp_path / "runner.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE account_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            cash REAL NOT NULL,
            equity REAL NOT NULL,
            positions_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO account_snapshots (ts, cash, equity, positions_json) VALUES (?, ?, ?, ?)",
        (datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp(), 1.0, 2.0, "{}"),
    )
    conn.commit()
    conn.close()

    store = RunnerStore(db)
    out = store.latest_snapshot()
    assert out is not None
    assert out.base_currency == "USD"  # column default
    assert out.cash_by_currency == {}


def test_day_equity_bounds(store: RunnerStore) -> None:
    base = datetime(2026, 6, 10, tzinfo=timezone.utc)
    for hour, eq in [(1, 100_000.0), (12, 101_500.0), (20, 99_700.0)]:
        store.save_snapshot(AccountSnapshot(ts=base + timedelta(hours=hour), cash=eq, equity=eq))
    bounds = store.day_equity_bounds(base)
    assert bounds == (pytest.approx(100_000.0), pytest.approx(99_700.0))
    # Window with <2 snapshots -> None (silence beats a fabricated 0.0%).
    assert store.day_equity_bounds(base + timedelta(hours=19)) is None
    assert store.day_equity_bounds(base + timedelta(days=2)) is None
