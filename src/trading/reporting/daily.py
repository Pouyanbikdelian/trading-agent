r"""Daily report — assemble the structured payload the executive summary
and the Markdown renderer both consume.

Three layers:

* ``gather_daily_report(...)`` reads everything we care about from the
  stores: latest account snapshot, equity history, recent cycle outcomes,
  current positions, top moves over the day. Pure pandas/SQLite; no
  network, no LLM.
* ``trading.reporting.news`` fetches news for the held symbols. Optional
  — if no data source is configured the report runs without it.
* ``trading.reporting.executive_summary`` is the optional Anthropic API
  call. If ``ANTHROPIC_API_KEY`` isn't set, the report renders with a
  deterministic bullet-point summary instead of the LLM-generated prose.

The Markdown render is a separate function so the same data can feed a
Telegram alert, an email body, or a JSON API later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from trading.core.config import settings
from trading.execution.store import OrderStore
from trading.regime.vix import VixRegime, fetch_vix_levels
from trading.runner.heartbeat import read_heartbeat
from trading.runner.state import RunnerStore


@dataclass
class DailyReport:
    """Structured snapshot for a single trading day."""

    as_of: datetime

    # account state
    cash: float
    equity: float
    positions: dict[str, dict]  # symbol -> {quantity, avg_price, market_value, weight}

    # PnL — both daily and trailing
    daily_pnl: float
    daily_pnl_pct: float
    week_pnl_pct: float
    month_pnl_pct: float
    ytd_pnl_pct: float

    # equity history (last 252 trading days)
    equity_curve: pd.Series

    # recent cycle outcomes
    recent_cycles: list[dict]

    # trades from the latest cycle
    last_cycle_trades: list[dict] = field(default_factory=list)

    # market regime
    vix_level: float | None = None
    vix_regime: str | None = None

    # risk state
    halted: bool = False
    halt_reason: str = ""

    # heartbeat status
    heartbeat_age_seconds: float | None = None

    # news (filled in by reporting.news if available)
    news_by_symbol: dict[str, list[dict]] = field(default_factory=dict)

    # executive summary (filled in by reporting.executive_summary if available)
    executive_summary: str | None = None


def gather_daily_report(
    *,
    runner_db: Path | None = None,
    orders_db: Path | None = None,
    halt_state_path: Path | None = None,
    heartbeat_path: Path | None = None,
    as_of: datetime | None = None,
    fetch_vix: bool = True,
) -> DailyReport:
    r"""Read the persisted state and assemble a :class:`DailyReport`.

    All path arguments default to the standard runner locations under
    ``settings.state_dir``. The function performs no writes; it is safe
    to call concurrently with the live runner.
    """
    state_dir = settings.state_dir
    runner_db = runner_db or (state_dir / "runner.db")
    orders_db = orders_db or (state_dir / "orders.db")
    halt_state_path = halt_state_path or (state_dir / "halt.json")
    heartbeat_path = heartbeat_path or (state_dir / "heartbeat.json")
    as_of = as_of or datetime.now(tz=timezone.utc)

    runner_store = RunnerStore(runner_db)
    snap = runner_store.latest_snapshot()
    if snap is None:
        # No snapshots yet — return a stub so the report still renders.
        return DailyReport(
            as_of=as_of,
            cash=0.0,
            equity=0.0,
            positions={},
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            week_pnl_pct=0.0,
            month_pnl_pct=0.0,
            ytd_pnl_pct=0.0,
            equity_curve=pd.Series(dtype=float),
            recent_cycles=[],
        )

    curve = pd.Series(
        {ts: eq for ts, eq in runner_store.equity_curve()},
        dtype=float,
    ).sort_index()
    equity_curve_252 = curve.iloc[-252:]

    # PnL windows
    daily_pnl = float(_pnl_over(curve, days=1))
    daily_pnl_pct = float(_pct_change_over(curve, days=1))
    week_pnl_pct = float(_pct_change_over(curve, days=7))
    month_pnl_pct = float(_pct_change_over(curve, days=30))
    ytd_pnl_pct = float(_ytd_pct(curve, as_of=as_of))

    # Position-level details
    positions_view: dict[str, dict] = {}
    for key, pos in snap.positions.items():
        market_value = pos.quantity * pos.avg_price + pos.unrealized_pnl
        weight = market_value / snap.equity if snap.equity > 0 else 0.0
        positions_view[pos.instrument.symbol] = {
            "key": key,
            "quantity": pos.quantity,
            "avg_price": pos.avg_price,
            "market_value": market_value,
            "weight": weight,
            "realized_pnl": pos.realized_pnl,
            "unrealized_pnl": pos.unrealized_pnl,
        }

    recent_cycles = runner_store.recent_cycles(limit=10)

    # Trades from the latest cycle — orders + fills in the last 24h
    order_store = OrderStore(orders_db)
    since = as_of - timedelta(days=1)
    fills = order_store.load_fills(since=since)
    last_trades: list[dict] = [
        {
            "ts": f.ts,
            "order_id": f.order_id,
            "quantity": f.quantity,
            "price": f.price,
            "commission": f.commission,
            "venue": f.venue,
        }
        for f in fills
    ]

    # Market regime
    vix_level: float | None = None
    vix_regime: str | None = None
    if fetch_vix:
        try:
            vix_series = fetch_vix_levels()
            if not vix_series.empty:
                cls = VixRegime(n_states=3).fit(vix_series)
                labels = cls.predict(vix_series)
                last_id = int(labels.iloc[-1])
                vix_level = float(vix_series.iloc[-1])
                vix_regime = VixRegime.label_for(last_id)
        except Exception:
            # VIX fetch is best-effort; failure shouldn't crash the report
            pass

    # Halt state
    halted = False
    halt_reason = ""
    if halt_state_path.exists():
        try:
            import json

            payload = json.loads(halt_state_path.read_text())
            halted = bool(payload.get("halted", False))
            halt_reason = str(payload.get("reason", ""))
        except Exception:
            pass

    # Heartbeat freshness
    hb = read_heartbeat(heartbeat_path)
    hb_age: float | None = None
    if hb and heartbeat_path.exists():
        hb_age = (
            as_of - datetime.fromtimestamp(heartbeat_path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds()

    return DailyReport(
        as_of=as_of,
        cash=snap.cash,
        equity=snap.equity,
        positions=positions_view,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        week_pnl_pct=week_pnl_pct,
        month_pnl_pct=month_pnl_pct,
        ytd_pnl_pct=ytd_pnl_pct,
        equity_curve=equity_curve_252,
        recent_cycles=recent_cycles,
        last_cycle_trades=last_trades,
        vix_level=vix_level,
        vix_regime=vix_regime,
        halted=halted,
        halt_reason=halt_reason,
        heartbeat_age_seconds=hb_age,
    )


def _pnl_over(curve: pd.Series, *, days: int) -> float:
    """Dollar PnL over the last ``days`` business days from the curve."""
    if len(curve) < days + 1:
        return 0.0
    return float(curve.iloc[-1] - curve.iloc[-days - 1])


def _pct_change_over(curve: pd.Series, *, days: int) -> float:
    if len(curve) < days + 1:
        return 0.0
    return float(curve.iloc[-1] / curve.iloc[-days - 1] - 1.0)


def _ytd_pct(curve: pd.Series, *, as_of: datetime) -> float:
    if curve.empty:
        return 0.0
    year_start = pd.Timestamp(year=as_of.year, month=1, day=1, tz=curve.index.tz)
    in_year = curve[curve.index >= year_start]
    if in_year.empty:
        return 0.0
    return float(in_year.iloc[-1] / in_year.iloc[0] - 1.0)
