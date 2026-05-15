r"""Tests for the daily-report assembly + Markdown renderer + fallback summary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Fill,
    Instrument,
    Order,
    Position,
    Side,
)
from trading.execution.store import OrderStore
from trading.reporting import (
    DailyReport,
    Headline,
    gather_daily_report,
    render_markdown,
    summarise,
)
from trading.reporting.executive_summary import _fallback_summary
from trading.runner.state import RunnerStore


def _prime_state(tmp_path: Path) -> tuple[Path, Path]:
    runner_db = tmp_path / "runner.db"
    orders_db = tmp_path / "orders.db"

    rs = RunnerStore(runner_db)
    ins = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(10):
        rs.save_snapshot(
            AccountSnapshot(
                ts=base.replace(day=10 + i),
                cash=20_000.0,
                equity=100_000.0 + 1_000 * i,  # +1% per day
                positions={
                    "equity:AAPL": Position(
                        instrument=ins,
                        quantity=10.0,
                        avg_price=200.0,
                        unrealized_pnl=10.0 * i,
                    ),
                },
            )
        )

    os_ = OrderStore(orders_db)
    o = Order(
        client_order_id="cid-1",
        instrument=ins,
        side=Side.BUY,
        quantity=10,
        created_at=base,
    )
    os_.save_order(o)
    os_.save_fill(
        Fill(order_id="cid-1", ts=base, quantity=10, price=200.0, commission=0.20),
        client_order_id="cid-1",
    )
    return runner_db, orders_db


def test_gather_daily_report_pulls_from_persisted_state(tmp_path: Path) -> None:
    runner_db, orders_db = _prime_state(tmp_path)
    halt_path = tmp_path / "halt.json"
    hb_path = tmp_path / "heartbeat.json"
    halt_path.write_text('{"halted": false, "reason": ""}')

    report = gather_daily_report(
        runner_db=runner_db,
        orders_db=orders_db,
        halt_state_path=halt_path,
        heartbeat_path=hb_path,
        fetch_vix=False,  # no network in unit tests
        as_of=datetime(2026, 5, 20, tzinfo=timezone.utc),
    )
    assert report.equity == 109_000.0
    assert "AAPL" in report.positions
    assert report.positions["AAPL"]["quantity"] == 10.0
    assert len(report.recent_cycles) == 0  # we didn't save any cycles
    assert report.halted is False


def test_gather_daily_report_empty_state(tmp_path: Path) -> None:
    runner_db = tmp_path / "runner.db"
    orders_db = tmp_path / "orders.db"
    # Open the DBs to materialise the schema, but don't write anything.
    _ = RunnerStore(runner_db).conn
    _ = OrderStore(orders_db).conn
    report = gather_daily_report(
        runner_db=runner_db,
        orders_db=orders_db,
        halt_state_path=tmp_path / "halt.json",
        heartbeat_path=tmp_path / "heartbeat.json",
        fetch_vix=False,
    )
    assert report.equity == 0.0
    assert report.positions == {}


def test_fallback_summary_includes_pnl() -> None:
    report = DailyReport(
        as_of=datetime(2026, 5, 14, tzinfo=timezone.utc),
        cash=20_000.0,
        equity=100_000.0,
        positions={},
        daily_pnl=500.0,
        daily_pnl_pct=0.005,
        week_pnl_pct=0.012,
        month_pnl_pct=0.034,
        ytd_pnl_pct=0.09,
        equity_curve=None,
        recent_cycles=[],
    )
    out = _fallback_summary(report)
    assert "100,000.00" in out
    assert "+0.50%" in out


def test_summarise_uses_fallback_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    report = DailyReport(
        as_of=datetime(2026, 5, 14, tzinfo=timezone.utc),
        cash=20_000.0,
        equity=100_000.0,
        positions={},
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        week_pnl_pct=0.0,
        month_pnl_pct=0.0,
        ytd_pnl_pct=0.0,
        equity_curve=None,
        recent_cycles=[],
    )
    out = summarise(report)
    assert out  # never empty
    assert "100,000.00" in out


def test_render_markdown_emits_sections(tmp_path: Path) -> None:
    report = DailyReport(
        as_of=datetime(2026, 5, 14, tzinfo=timezone.utc),
        cash=20_000.0,
        equity=100_000.0,
        positions={
            "AAPL": {
                "key": "equity:AAPL",
                "quantity": 10.0,
                "avg_price": 200.0,
                "market_value": 2_000.0,
                "weight": 0.02,
                "realized_pnl": 0.0,
                "unrealized_pnl": 5.0,
            },
        },
        daily_pnl=100.0,
        daily_pnl_pct=0.001,
        week_pnl_pct=0.005,
        month_pnl_pct=0.015,
        ytd_pnl_pct=0.08,
        equity_curve=None,
        recent_cycles=[
            {
                "ts": datetime(2026, 5, 14, 16, tzinfo=timezone.utc),
                "status": "ok",
                "orders_submitted": 3,
                "fills_received": 3,
                "error": None,
                "duration_ms": 5500.0,
            },
        ],
        last_cycle_trades=[],
        news_by_symbol={
            "AAPL": [
                Headline(
                    title="Apple announces something",
                    source="Reuters",
                    url="https://example.com/aapl",
                    published=datetime(2026, 5, 14, 9, tzinfo=timezone.utc),
                    matched_symbols=["AAPL"],
                ),
            ]
        },
        vix_level=18.5,
        vix_regime="mid_vol",
    )
    md = render_markdown(report, executive_summary="The system did fine.")
    assert "# Trading Daily Report" in md
    assert "## Executive summary" in md
    assert "The system did fine." in md
    assert "AAPL" in md
    assert "Apple announces something" in md
    assert "mid_vol" in md
