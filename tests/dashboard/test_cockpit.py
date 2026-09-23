"""Cockpit blocks: each answers an operator question from state, read-only."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trading.core.types import AccountSnapshot, AssetClass, Instrument, Position
from trading.dashboard import cockpit

NOW = datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc)
FX = {"USD": 0.8, "EUR": 0.9}


def _settings(**kw):
    base = dict(
        trading_env="live",
        is_live_armed=lambda: True,
        require_cycle_approval=True,
        strategy_sleeve_pct=0.0,
        agent_pm_sleeve_pct=1.0,
        max_drawdown_pct=0.15,
        max_daily_loss_pct=0.02,
        max_gross_exposure=1.0,
        max_position_pct=0.10,
        pm_pre_cycle_lead_minutes=45,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _pos(sym: str, qty: float, avg: float, mark: float, ccy: str = "USD") -> Position:
    inst = Instrument(symbol=sym, asset_class=AssetClass.EQUITY, currency=ccy)
    return Position(instrument=inst, quantity=qty, avg_price=avg, unrealized_pnl=qty * (mark - avg))


def _snapshot(positions: list[Position], *, cash: float = 50_000.0) -> AccountSnapshot:
    value = sum(
        p.quantity * (p.avg_price + p.unrealized_pnl / p.quantity) * FX[p.instrument.currency]
        for p in positions
    )
    return AccountSnapshot(
        ts=NOW,
        cash=cash,
        equity=cash + value,
        positions={p.instrument.key: p for p in positions},
        base_currency="CHF",
        fx_rates=FX,
    )


# ------------------------------------------------------------------ status


def test_unreadable_halt_file_reads_as_halted(tmp_path: Path) -> None:
    (tmp_path / "halt.json").write_text("{corrupt")
    st = cockpit.status_block(tmp_path, _settings())
    assert st["halted"] is True and "unreadable" in st["halt_reason"]


def test_status_reports_arming_approval_and_sleeves(tmp_path: Path) -> None:
    (tmp_path / "broker_liveness.json").write_text(
        json.dumps({"ready": True, "last_success_at": NOW.isoformat()})
    )
    st = cockpit.status_block(tmp_path, _settings())
    assert (
        st["live_armed"]
        and st["require_approval"]
        and st["sleeves"] == {"strategy": 0.0, "pm": 1.0}
    )
    assert st["broker"]["ready"] is True and st["halted"] is False


# -------------------------------------------------------------------- risk


def _halt(
    tmp_path: Path, snap: AccountSnapshot, pins: set[str], *, identity: str | None = "match"
) -> None:
    from trading.risk.limits import HaltState
    from trading.runner.managed_account import managed_view

    desk = managed_view(snap, pins, fx_rates=FX).account
    state = HaltState(
        equity_high_watermark=float(desk.equity) * 1.05,
        daily_equity_open=float(desk.equity) * 1.01,
        baseline_scope=desk.scope,
        baseline_book_identity=desk.risk_book_identity if identity == "match" else identity,
    )
    (tmp_path / "halt.json").write_text(state.model_dump_json())


def test_risk_is_measured_on_the_desk_book(tmp_path: Path) -> None:
    snap = _snapshot([_pos("NVDA", 10, 100, 120), _pos("SHA", 100, 6, 7, "EUR")])
    (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["SHA"]}))
    _halt(tmp_path, snap, {"SHA"})

    r = cockpit.risk_block(tmp_path, _settings(), snap)

    assert r["same_book"] is True
    assert r["desk_equity"] == pytest.approx(float(snap.equity) - 100 * 7 * 0.9)
    assert r["drawdown_pct"] == pytest.approx(1 / 1.05 - 1)
    assert r["day_pct"] == pytest.approx(1 / 1.01 - 1)
    assert r["gross_exposure"] == pytest.approx(10 * 120 * 0.8 / r["desk_equity"])
    assert r["pinned"] == ["SHA"]


def test_a_baseline_for_another_book_reports_no_returns(tmp_path: Path) -> None:
    """The 2026-09-18 −36% scare: managed equity vs a whole-account peak."""
    snap = _snapshot([_pos("NVDA", 10, 100, 120)])
    (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["NVDA"]}))
    _halt(tmp_path, snap, {"NVDA"}, identity='["managed",["OTHER"],[]]')

    r = cockpit.risk_block(tmp_path, _settings(), snap)

    assert r["same_book"] is False
    assert "drawdown_pct" not in r and "different book" in r["note"]


def test_risk_without_a_snapshot_degrades(tmp_path: Path) -> None:
    r = cockpit.risk_block(tmp_path, _settings(), None)
    assert r["note"] == "no account snapshot yet" and r["limits"]["max_drawdown_pct"] == 0.15


# --------------------------------------------------------------- positions


def test_positions_carry_pin_fx_weight_and_stop(tmp_path: Path) -> None:
    snap = _snapshot([_pos("NVDA", 10, 100, 120), _pos("SHA", 100, 6, 7, "EUR")])
    (tmp_path / "holds.json").write_text(json.dumps({"symbols": ["SHA"]}))
    (tmp_path / "guards.json").write_text(
        json.dumps({"positions": {"NVDA": {"stop_level": 110.0}}})
    )

    rows = {r["symbol"]: r for r in cockpit.positions_block(tmp_path, snap)}

    assert rows["SHA"]["pinned"] and not rows["NVDA"]["pinned"]
    assert rows["SHA"]["value_base"] == pytest.approx(630.0)
    assert rows["NVDA"]["value_base"] == pytest.approx(960.0)
    assert rows["NVDA"]["stop_distance_pct"] == pytest.approx(120 / 110 - 1)
    assert rows["NVDA"]["unrealized_pct"] == pytest.approx(0.2)


# ------------------------------------------------------------ cycles/watch


def test_cycles_and_watch_come_from_the_cycles_table(tmp_path: Path) -> None:
    from trading.core.types import RiskDecision
    from trading.runner.cycle import CycleReport
    from trading.runner.state import RunnerStore

    rs = RunnerStore(tmp_path / "runner.db")
    for i in range(4):
        rs.save_cycle(
            CycleReport(
                ts=NOW - timedelta(days=7 * (i + 1)),
                status="halted_review",
                orders_submitted=0,
                fills_received=0,
                decisions=[RiskDecision(action="halt", reason="x")],
            )
        )
    assert [c["status"] for c in cockpit.cycles_block(rs)] == ["halted_review"] * 4
    w = cockpit.watch_block(tmp_path, rs, cron="0 15 * * FRI", tz="America/New_York", now=NOW)
    assert any(f["key"].startswith("stuck:") for f in w["findings"])


# ------------------------------------------------------------------ pending


def test_pending_shows_approval_staged_command_and_open_orders(tmp_path: Path) -> None:
    from trading.bot import confirmations
    from trading.core.types import Order, OrderType, Side, TimeInForce
    from trading.execution.store import OrderStore

    (tmp_path / "cycle_approval_pending.json").write_text(
        json.dumps(
            {
                "id": "c1",
                "ts": NOW.isoformat(),
                "deploy_pct": 40.0,
                "plan": {
                    "orders": [
                        {
                            "symbol": "NVDA",
                            "side": "BUY",
                            "quantity": 5,
                            "notional_base": 600,
                            "context": "new",
                        }
                    ]
                },
            }
        )
    )
    confirmations.stage(tmp_path, "flatten", {}, "flatten every position")
    store = OrderStore(tmp_path / "orders.db")
    store.save_order(
        Order(
            client_order_id="o1",
            instrument=Instrument(symbol="MU", asset_class=AssetClass.EQUITY),
            side=Side.SELL,
            quantity=3,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            created_at=datetime.now(tz=timezone.utc) - timedelta(days=5),
        )
    )

    p = cockpit.pending_block(tmp_path, store)

    assert p["approval"]["orders"][0]["symbol"] == "NVDA"
    assert p["staged"]["kind"] == "flatten"
    assert p["open_orders"][0]["id"] == "o1" and p["open_orders"][0]["age_days"] >= 5


# ------------------------------------------------------------------- agents


def _memory(tmp_path: Path) -> Path:
    from trading.memory.store import MemoryStore

    MemoryStore(tmp_path / "memory").stats()
    return tmp_path / "memory" / "memory.db"


def test_agent_skill_is_measured_against_a_coin_flip(tmp_path: Path) -> None:
    db = _memory(tmp_path)
    conn = sqlite3.connect(db)
    rows = (
        [("good", 0.8, "hit")] * 8
        + [("good", 0.8, "miss")] * 2
        + [("bad", 0.8, "hit")] * 3
        + [("bad", 0.8, "miss")] * 7
    )
    for i, (agent, conf, outcome) in enumerate(rows):
        brier = (conf - (1.0 if outcome == "hit" else 0.0)) ** 2
        conn.execute(
            "INSERT INTO predictions (id, ts, agent, subject, direction, horizon_days, confidence, statement, due_ts, graded_ts, outcome, brier) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"p{i}",
                NOW.timestamp(),
                agent,
                "SPY",
                "up",
                5,
                conf,
                "x",
                NOW.timestamp(),
                NOW.timestamp(),
                outcome,
                brier,
            ),
        )
    conn.execute(
        "INSERT INTO predictions (id, ts, agent, subject, direction, horizon_days, confidence, statement, due_ts) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "open1",
            NOW.timestamp(),
            "good",
            "SPY",
            "up",
            5,
            0.7,
            "x",
            (NOW + timedelta(days=2)).timestamp(),
        ),
    )
    conn.commit()
    conn.close()

    a = cockpit.agents_block(db, now=NOW)
    by = {x["agent"]: x for x in a["agents"]}

    assert by["good"]["skill"] > 0 > by["bad"]["skill"]
    assert by["bad"]["overconfidence"] == pytest.approx(0.8 - 0.3)
    assert a["agents"][0]["agent"] == "good"  # best first
    assert a["open"] == 1 and a["due_7d"] == 1
    assert a["bins"] and a["bins"][0]["n"] == 20


def test_agents_block_never_creates_a_database(tmp_path: Path) -> None:
    out = cockpit.agents_block(tmp_path / "memory" / "memory.db")
    assert out["agents"] == [] and not (tmp_path / "memory").exists()


def test_lessons_count_measured_outcomes_separately_from_votes(tmp_path: Path) -> None:
    db = _memory(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO lessons (id, created_ts, statement, status, support, contradict) VALUES ('l1', ?, 'gold hedges', 'candidate', 5, 0)",
        (NOW.timestamp(),),
    )
    for i, rel in enumerate(["supports", "supports", "contradicts"]):
        conn.execute(
            "INSERT INTO lesson_evidence VALUES ('l1', ?, ?, ?, 'outcome', '')",
            (f"e{i}", rel, NOW.timestamp()),
        )
    conn.execute(
        "INSERT INTO lesson_evidence VALUES ('l1', 'v1', 'supports', ?, 'review', '')",
        (NOW.timestamp(),),
    )
    conn.commit()
    conn.close()

    lb = cockpit.lessons_block(db)
    l1 = lb["lessons"][0]

    assert (l1["measured_support"], l1["measured_contradict"]) == (2, 1)
    assert l1["votes_support"] == 5 and lb["counts"] == {"candidate": 1}


# ---------------------------------------------------------------------- llm


def test_llm_usage_is_aggregated_over_the_window(tmp_path: Path) -> None:
    p = tmp_path / "llm_usage.jsonl"
    recs = [
        {
            "ts": (NOW - timedelta(days=1)).isoformat(),
            "tier": "frontier",
            "latency_ms": 1000,
            "input_tokens": 10,
            "output_tokens": 5,
        },
        {
            "ts": (NOW - timedelta(days=1)).isoformat(),
            "tier": "standard",
            "latency_ms": 3000,
            "input_tokens": 20,
            "output_tokens": 5,
            "http_status": 529,
        },
        {"ts": (NOW - timedelta(days=30)).isoformat(), "tier": "standard", "input_tokens": 999},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")

    out = cockpit.llm_block(p, now=NOW)

    assert out["totals"]["calls"] == 2 and out["totals"]["errors"] == 1
    assert out["totals"]["input_tokens"] == 30
    assert out["by_tier"] == {"frontier": 1, "standard": 1}


# ------------------------------------------------------------------- market


def test_market_temperature_from_cached_spy_and_market_watch(tmp_path: Path) -> None:
    from trading.data.cache import ParquetCache

    idx = pd.date_range("2025-01-01", periods=300, freq="B", tz="UTC")
    closes = [400 + i for i in range(300)]  # steady uptrend, at its high
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1.0,
            "adj_close": closes,
        },
        index=idx,
    )
    df.index.name = "ts"
    ParquetCache(tmp_path).write(Instrument(symbol="SPY", asset_class=AssetClass.ETF), "1D", df)

    m = cockpit.market_block(
        tmp_path, {"latest": {"pct_above_200": 0.8, "vix": 12.0, "vix3m": 14.5}}
    )

    assert m["drawdown_from_high"] == pytest.approx(0.0)
    assert m["spy_vs_200d"] > 0.1
    assert m["label"] == "Stretched" and m["temperature"] >= 0.6
    assert m["stabilised"] is True


def test_market_without_spy_uses_what_it_has(tmp_path: Path) -> None:
    m = cockpit.market_block(tmp_path, {"latest": {"pct_above_200": 0.3}})
    assert [c["key"] for c in m["components"]] == ["breadth"]
    assert m["label"] == "Discount"


# ------------------------------------------------------------------ summary


def test_build_summary_carries_every_cockpit_block_on_empty_state(tmp_path: Path) -> None:
    from trading.dashboard.app import build_summary

    out = build_summary(tmp_path, tmp_path)

    for key in (
        "status",
        "risk",
        "positions",
        "cycles",
        "watch",
        "schedule",
        "pending",
        "agents",
        "lessons_v2",
        "edge",
        "llm",
        "market",
    ):
        assert key in out, key
    assert out["positions"] == [] and out["cycles"] == []
