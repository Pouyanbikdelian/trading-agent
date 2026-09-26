r"""Cockpit blocks for the dashboard: state that answers "what needs me now?".

Why this module exists (2026-09-23). The first dashboard charted what was
easy to chart — headlines, a normalized price spaghetti, raw memory
counters — and left out what an operator of a live account actually has to
know: is the desk halted and why, how far each kill switch is from firing,
did the last cycles trade or fail, what is waiting for approval, and are the
agents any better than a coin flip. All of that already sat in ``state/``.

Every block is read-only and self-contained: a missing or malformed source
degrades that block to an empty/None shape with a ``note``; it never raises
into ``build_summary``. SQLite sources are opened ``mode=ro`` so a dashboard
refresh cannot create or migrate a database.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

NY = "America/New_York"
COIN_FLIP_BRIER = 0.25


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _ro_connect(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------- status


def status_block(state_dir: Path, settings: Any) -> dict[str, Any]:
    """Who is in control and in what state: env, arming, halt, broker, mode."""
    halt_raw = _read_json(state_dir / "halt.json")
    halt_path_exists = (state_dir / "halt.json").exists()
    if halt_path_exists and not isinstance(halt_raw, dict):
        # The risk manager treats an unreadable halt file as halted; so
        # does the dashboard, rather than painting a green light.
        halted, reason, halted_at = True, "halt.json unreadable — treated as halted", None
    else:
        h = halt_raw or {}
        halted, reason, halted_at = (
            bool(h.get("halted")),
            str(h.get("reason") or ""),
            h.get("halted_at"),
        )
    mode_raw = _read_json(state_dir / "mode.json") or {}
    liveness = _read_json(state_dir / "broker_liveness.json") or {}
    errors = _read_json(state_dir / "consecutive_errors.json") or {}
    armed = False
    try:
        armed = bool(settings.is_live_armed())
    except Exception:
        armed = False
    return {
        "env": str(getattr(settings, "trading_env", "") or ""),
        "live_armed": armed,
        "require_approval": bool(getattr(settings, "require_cycle_approval", False)),
        "sleeves": {
            "strategy": getattr(settings, "strategy_sleeve_pct", None),
            "pm": getattr(settings, "agent_pm_sleeve_pct", None),
        },
        "halted": halted,
        "halt_reason": reason,
        "halted_at": halted_at,
        "mode": str(mode_raw.get("mode") or "neutral"),
        "mode_set_at": mode_raw.get("set_at") or "",
        "mode_reason": mode_raw.get("reason") or "",
        "broker": {
            "ready": liveness.get("ready"),
            "checked_at": liveness.get("checked_at"),
            "last_success_at": liveness.get("last_success_at"),
            "detail": liveness.get("detail"),
        },
        "consecutive_errors": int(errors.get("count") or 0) if isinstance(errors, dict) else 0,
    }


# ----------------------------------------------------------------------- risk


def _position_mark(position: Any) -> float | None:
    try:
        qty = float(position.quantity)
        if qty == 0:
            return None
        mark = float(position.avg_price) + float(position.unrealized_pnl or 0.0) / qty
        return mark if mark > 0 else None
    except Exception:
        return None


def _to_base(amount: float, currency: str, base: str, rates: dict[str, float]) -> float | None:
    ccy, base = (currency or "USD").upper(), (base or "USD").upper()
    if ccy == base:
        return amount
    rate = rates.get(ccy)
    return amount * float(rate) if rate else None


def risk_block(state_dir: Path, settings: Any, snapshot: Any) -> dict[str, Any]:
    """Distance to each kill switch, measured the way the risk manager does.

    Uses the managed (desk) view: pinned holdings are taken out exactly as
    ``runner.managed_account.managed_view`` does, and returns are only
    reported when the stored baseline describes the same book (scope and
    identity) — the check whose absence produced the 2026-09-18 −36% scare.
    """
    from trading.risk.limits import HaltState
    from trading.runner.holds import load_holds
    from trading.runner.managed_account import managed_view

    out: dict[str, Any] = {
        "limits": {
            "max_drawdown_pct": getattr(settings, "max_drawdown_pct", None),
            "max_daily_loss_pct": getattr(settings, "max_daily_loss_pct", None),
            "max_gross_exposure": getattr(settings, "max_gross_exposure", None),
            "max_position_pct": getattr(settings, "max_position_pct", None),
        },
        "note": "",
    }
    raw = _read_json(state_dir / "halt.json")
    state = None
    if isinstance(raw, dict):
        try:
            state = HaltState.model_validate(raw)
        except Exception:
            state = None
    if state is not None:
        out.update(
            {
                "hwm": state.equity_high_watermark or None,
                "daily_open": state.daily_equity_open or None,
                "baseline_scope": state.baseline_scope or "account",
                "baseline_session": (
                    state.daily_baseline_session.isoformat()
                    if state.daily_baseline_session
                    else None
                ),
                "baseline_currency": state.daily_baseline_currency,
            }
        )
    if snapshot is None:
        out["note"] = "no account snapshot yet"
        return out
    base = str(getattr(snapshot, "base_currency", "") or "USD").upper()
    rates = dict(getattr(snapshot, "fx_rates", None) or {})
    out["account_equity"] = float(snapshot.equity)
    out["currency"] = base
    out["snapshot_at"] = snapshot.ts.isoformat()
    try:
        view = managed_view(snapshot, load_holds(state_dir, strict=True), fx_rates=rates)
        desk = view.account
    except Exception as e:
        out["note"] = f"desk valuation unavailable: {e}"
        return out
    desk_equity = float(desk.equity)
    out["desk_equity"] = desk_equity
    out["pinned"] = sorted(getattr(view, "excluded", []) or [])
    gross = 0.0
    for pos in (desk.positions or {}).values():
        mark = _position_mark(pos)
        value = (
            None
            if mark is None
            else _to_base(
                abs(float(pos.quantity)) * mark,
                getattr(pos.instrument, "currency", "USD"),
                base,
                rates,
            )
        )
        if value is None:
            gross = float("nan")
            break
        gross += value
    out["gross_exposure"] = (gross / desk_equity) if desk_equity > 0 and gross == gross else None
    out["cash_pct"] = (
        float(snapshot.cash) / float(snapshot.equity) if float(snapshot.equity) > 0 else None
    )
    same_book = (
        state is not None
        and (state.baseline_scope or "account") == desk.scope
        and (
            state.baseline_book_identity == desk.risk_book_identity
            or (state.baseline_book_identity is None and desk.scope == "account")
        )
    )
    out["same_book"] = bool(same_book)
    if not same_book:
        out["note"] = (
            "baseline describes a different book — returns unavailable until /baseline reset"
        )
    if same_book and state is not None:
        if state.equity_high_watermark > 0:
            out["drawdown_pct"] = desk_equity / state.equity_high_watermark - 1.0
        if state.daily_equity_open > 0:
            out["day_pct"] = desk_equity / state.daily_equity_open - 1.0
    return out


def _pin_transfers(days: list[dict[str, Any]]) -> None:
    """Split each runner day into desk and pinned books, with manual trades
    in pinned names counted as transfers, not P&L.

    ``desk = account − pinned`` is exact on any one day, but the day-to-day
    CHANGE is not a return: when the operator buys 20 more NVDA by hand,
    cash leaves the desk and the same value lands in the pinned book, and
    the naive split books it as a desk loss and a pinned gain of equal
    size. The quantity change in each pinned symbol, valued at that day's
    unit value, is that transfer. What remains is price P&L (up to the
    fill-vs-close difference on the trade day, which is small and real).

    Adds ``pinned`` (account − desk) and ``pin_transfer`` (base currency,
    positive = cash moved from the desk into pinned names) and removes the
    private ``_pins`` scratch field.
    """
    prev: dict[str, tuple[float, float]] | None = None
    for d in days:
        cur = d.pop("_pins", {})
        d["pinned"] = None if d.get("desk") is None else round(d["account"] - d["desk"], 2)
        if d.get("desk") is None:
            d["pin_transfer"] = None
            prev = None
            continue
        transfer = 0.0
        if prev is not None:
            for sym in {*cur, *prev}:
                q1, v1 = cur.get(sym, (0.0, 0.0))
                q0, v0 = prev.get(sym, (0.0, 0.0))
                if q1 == q0:
                    continue
                unit = v1 / q1 if q1 else (v0 / q0 if q0 else 0.0)
                transfer += (q1 - q0) * unit
        d["pin_transfer"] = round(transfer, 2)
        prev = cur


def equity_block(runner_db: Path, state_dir: Path) -> dict[str, Any]:
    """Daily account NetLiq, the desk book, and capital actually contributed.

    Three corrections to the old equity chart (2026-09-23):

    * The account line is mostly the operator's own pinned positions, so it
      moved with NVDA and GLD while the desk sat in cash. ``desk`` applies
      TODAY's pin list to every past snapshot — an approximation the page
      states — valued exactly as ``managed_view`` values it.
    * Deposits are stated in the flow ledger (``runtime/capital_flows``) and
      subtracted on their day, instead of guessed from >25% jumps.
    * ``flow_candidates`` lists days whose cash and equity jumped together
      by ≥1% — the signature of a transfer, not a trade (a trade moves cash
      but not equity) — that no recorded flow explains, so the operator can
      confirm them with /deposit.
    """
    from trading.core.types import AccountSnapshot
    from trading.runner.holds import load_holds
    from trading.runner.managed_account import managed_view
    from trading.runner.state import _positions_from_json
    from trading.runtime.capital_flows import flows_in_base, load_flows

    out: dict[str, Any] = {"days": [], "flows": [], "flow_candidates": [], "note": ""}
    try:
        flows = load_flows(state_dir)
        out["flows"] = [f.to_dict() for f in flows]
    except Exception as e:
        flows = []
        out["note"] = f"capital_flows.json unreadable ({e}) — returns ignore flows until repaired"
    conn = _ro_connect(runner_db)
    rows: list[Any] = []
    if conn is not None:
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(account_snapshots)")}
            fx_col = "fx_rates_json" if "fx_rates_json" in cols else "'{}' AS fx_rates_json"
            ccy_col = "base_currency" if "base_currency" in cols else "'USD' AS base_currency"
            rows = conn.execute(
                f"""SELECT ts, cash, equity, positions_json, {fx_col}, {ccy_col}
                    FROM account_snapshots WHERE id IN (
                      SELECT MAX(id) FROM account_snapshots GROUP BY CAST(ts / 86400 AS INTEGER))
                    ORDER BY ts"""
            ).fetchall()
        finally:
            conn.close()
    try:
        pins = load_holds(state_dir)
    except Exception:
        pins = set()
    today = _now().date().isoformat()
    base = "USD"
    last_rates: dict[str, float] = {}
    days: list[dict[str, Any]] = []
    for r in rows:
        ts = datetime.fromtimestamp(float(r["ts"]), tz=timezone.utc)
        day = ts.date().isoformat()
        if day == today:
            continue  # a 60s snapshot, not a close; the same rule as equity_curve
        base = str(r["base_currency"] or "USD").upper()
        rates = json.loads(r["fx_rates_json"] or "{}")
        last_rates = rates or last_rates
        desk: float | None = None
        pin_units: dict[str, tuple[float, float]] = {}
        try:
            snap = AccountSnapshot(
                ts=ts,
                cash=float(r["cash"]),
                equity=float(r["equity"]),
                positions=_positions_from_json(r["positions_json"]),
                base_currency=base,
                fx_rates=rates,
            )
            view = managed_view(snap, pins, fx_rates=rates)
            desk = float(view.account.equity)
            qty: dict[str, float] = {}
            for p in snap.positions.values():
                sym = p.instrument.symbol.upper()
                if sym in view.excluded:
                    qty[sym] = qty.get(sym, 0.0) + float(p.quantity)
            pin_units = {s: (q, view.excluded[s]) for s, q in qty.items()}
        except Exception:
            desk = None
        # The broker's own dollar rate for that day. The dashboard has no
        # USDCHF price cache on the VPS (the Live tab said "USDCHF
        # unavailable" for weeks); every snapshot already carries the rate
        # IBKR valued the book at, which is the one the account felt.
        usdchf = None
        if base == "CHF" and rates.get("USD"):
            usdchf = float(rates["USD"])
        elif base == "USD" and rates.get("CHF"):
            usdchf = 1.0 / float(rates["CHF"])
        days.append(
            {
                "t": day,
                "account": round(float(r["equity"]), 2),
                "desk": None if desk is None else round(desk, 2),
                "cash": round(float(r["cash"]), 2),
                "usdchf": None if usdchf is None else round(usdchf, 5),
                "_pins": pin_units,
            }
        )
    _pin_transfers(days)
    # Pre-bot history from IBKR Flex statements (runtime/account_history):
    # NAV for days before the first snapshot, and deposits/withdrawals as
    # the broker recorded them. Flex flows are authoritative for the days
    # the statements cover; the manual ledger fills in after that.
    from trading.runtime.account_history import load_history

    try:
        hist = load_history(state_dir)
    except Exception as e:
        hist = {"nav": {}, "flows": []}
        out["note"] = (
            out["note"] + " · " if out["note"] else ""
        ) + f"account_history.json unreadable ({e})"
    if not days:
        # No runner snapshot yet: the statement is the only source of the
        # base currency. Defaulting to USD labelled a franc account "USD".
        base = str(hist.get("base_currency") or base).upper()
    first_snap = days[0]["t"] if days else None
    pre = [
        {
            "t": d,
            "account": round(float(v), 2),
            "desk": None,
            "pinned": None,
            "pin_transfer": None,
            "cash": None,
            "source": "ibkr_flex",
        }
        for d, v in sorted(hist["nav"].items())
        if first_snap is None or d < first_snap
    ]
    flex_end = max([*hist["nav"].keys(), *(f["day"] for f in hist["flows"])], default=None)
    by_day: dict[str, float | None] = {}
    for f in hist["flows"]:
        by_day[f["day"]] = (by_day.get(f["day"]) or 0.0) + float(f.get("amount_base") or 0.0)
    for k, v in flows_in_base(flows, base, last_rates).items():
        if flex_end is None or k > flex_end:
            by_day[k] = (
                None if v is None or by_day.get(k, 0.0) is None else (by_day.get(k) or 0.0) + v
            )
    for d in days:
        d["source"] = "runner"
    days = pre + days
    # An account opened long before it was funded (13 francs from 2025-09
    # to the first deposit in 2026-02) is not history: months of a near-zero
    # NAV flatten every chart into a line on the floor. Leading statement
    # days below 1,000 with no transfer on them are dropped.
    while (
        len(days) > 1
        and days[0].get("source") == "ibkr_flex"
        and not by_day.get(days[0]["t"])
        and days[0]["account"] < 1000
    ):
        days.pop(0)
    contributed = 0.0
    for d in days:
        f = by_day.get(d["t"])
        d["flow"] = f
        contributed += f or 0.0
        d["net_flows"] = round(contributed, 2)
    out["history"] = {
        "flex_first": next(iter(sorted(hist["nav"])), None),
        "flex_last": flex_end,
        "flex_flows": len(hist["flows"]),
        "bot_first": first_snap,
    }
    unconvertible = sorted(k for k, v in by_day.items() if v is None)
    if unconvertible:
        out["note"] = f"no FX rate for flows on {', '.join(unconvertible)}"
    for prev, cur in itertools.pairwise(days):
        if prev.get("cash") is None or cur.get("cash") is None:
            continue  # Flex history rows: the broker already listed its transfers
        d_cash, d_eq = cur["cash"] - prev["cash"], cur["account"] - prev["account"]
        if prev["account"] <= 0 or abs(d_cash) < 0.01 * prev["account"]:
            continue
        if abs(d_eq - d_cash) <= 0.3 * abs(d_cash) and not cur.get("flow"):
            out["flow_candidates"].append(
                {"t": cur["t"], "amount": round(d_cash, 2), "currency": base}
            )
    out["days"] = days
    out["currency"] = base
    out["pinned_now"] = sorted(pins)
    return out


def movers_block(runner_db: Path, state_dir: Path) -> dict[str, Any]:
    """Which position carried or hurt the account since the last close.

    Replaces the Live tab's attribution, which mixed currencies: IBKR's
    ``unrealizedPNL`` is in each contract's currency, and on a CHF book
    those dollars were divided by USDCHF as if they were francs. Here every
    position is valued in the BASE currency at each snapshot's own FX rate,
    so a US stock's line includes the dollar's move — which is what the
    franc account actually made or lost on it.

    Per symbol: ``Δ value − Δ quantity × today's unit value``. The second
    term is a trade, not a result. ``residual`` is whatever the positions
    do not explain: FX on non-base cash, interest, fees, a deposit.
    """
    from trading.core.types import AccountSnapshot
    from trading.core.valuation import position_value_base
    from trading.runner.holds import load_holds
    from trading.runner.state import _positions_from_json

    out: dict[str, Any] = {"rows": [], "currency": None}
    conn = _ro_connect(runner_db)
    if conn is None:
        return out
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(account_snapshots)")}
        fx_col = "fx_rates_json" if "fx_rates_json" in cols else "'{}' AS fx_rates_json"
        ccy_col = "base_currency" if "base_currency" in cols else "'USD' AS base_currency"
        sel = f"SELECT ts, cash, equity, positions_json, {fx_col}, {ccy_col} FROM account_snapshots"
        now = conn.execute(sel + " ORDER BY ts DESC LIMIT 1").fetchone()
        if now is None:
            return out
        day0 = datetime.fromtimestamp(float(now["ts"]), tz=timezone.utc).date()
        cutoff = datetime(day0.year, day0.month, day0.day, tzinfo=timezone.utc).timestamp()
        prev = conn.execute(sel + " WHERE ts < ? ORDER BY ts DESC LIMIT 1", (cutoff,)).fetchone()
    finally:
        conn.close()
    if prev is None:
        return out

    def _book(r: Any) -> tuple[str, dict[str, tuple[float, float]], float]:
        base = str(r["base_currency"] or "USD").upper()
        rates = json.loads(r["fx_rates_json"] or "{}")
        snap = AccountSnapshot(
            ts=datetime.fromtimestamp(float(r["ts"]), tz=timezone.utc),
            cash=float(r["cash"]),
            equity=float(r["equity"]),
            positions=_positions_from_json(r["positions_json"]),
            base_currency=base,
            fx_rates=rates,
        )
        per: dict[str, tuple[float, float]] = {}
        for p in snap.positions.values():
            value, _clean = position_value_base(p, base_currency=base, fx_rates=rates)
            q0, v0 = per.get(p.instrument.symbol.upper(), (0.0, 0.0))
            per[p.instrument.symbol.upper()] = (q0 + float(p.quantity), v0 + value)
        return base, per, float(r["equity"])

    try:
        base, cur, eq1 = _book(now)
        _b, old, eq0 = _book(prev)
    except Exception as e:
        out["note"] = f"snapshots unreadable ({e})"
        return out
    try:
        pins = load_holds(state_dir)
    except Exception:
        pins = set()
    rows = []
    for sym in sorted({*cur, *old}):
        q1, v1 = cur.get(sym, (0.0, 0.0))
        q0, v0 = old.get(sym, (0.0, 0.0))
        unit = v1 / q1 if q1 else (v0 / q0 if q0 else 0.0)
        pnl = (v1 - v0) - (q1 - q0) * unit
        rows.append(
            {
                "symbol": sym,
                "pnl": round(pnl, 2),
                "pct": round(pnl / v0, 4) if v0 else None,
                "value": round(v1, 2),
                "traded": q1 != q0,
                "pinned": sym in pins,
            }
        )
    rows.sort(key=lambda r: r["pnl"])
    explained = sum(r["pnl"] for r in rows)
    out.update(
        rows=rows,
        currency=base,
        since=datetime.fromtimestamp(float(prev["ts"]), tz=timezone.utc).isoformat(),
        as_of=datetime.fromtimestamp(float(now["ts"]), tz=timezone.utc).isoformat(),
        account_change=round(eq1 - eq0, 2),
        residual=round(eq1 - eq0 - explained, 2),
    )
    return out


# ------------------------------------------------------------------ positions


def positions_block(state_dir: Path, snapshot: Any) -> list[dict[str, Any]]:
    """Every broker position with its mark, base value, weight, pin and stop."""
    from trading.runner.holds import load_holds

    if snapshot is None:
        return []
    base = str(getattr(snapshot, "base_currency", "") or "USD").upper()
    rates = dict(getattr(snapshot, "fx_rates", None) or {})
    equity = float(snapshot.equity) or 0.0
    try:
        pins = load_holds(state_dir)
    except Exception:
        pins = set()
    guards = (_read_json(state_dir / "guards.json") or {}).get("positions") or {}
    rows: list[dict[str, Any]] = []
    for pos in (snapshot.positions or {}).values():
        sym = pos.instrument.symbol.upper()
        qty = float(pos.quantity)
        mark = _position_mark(pos)
        ccy = (getattr(pos.instrument, "currency", "USD") or "USD").upper()
        value = None if mark is None else _to_base(qty * mark, ccy, base, rates)
        cost = qty * float(pos.avg_price)
        g = guards.get(sym) or {}
        stop = g.get("stop_level")
        rows.append(
            {
                "symbol": sym,
                "qty": qty,
                "avg_price": float(pos.avg_price),
                "mark": mark,
                "currency": ccy,
                "value_base": value,
                "weight": (value / equity) if value is not None and equity > 0 else None,
                "unrealized_pnl": float(pos.unrealized_pnl or 0.0),
                "unrealized_pct": (float(pos.unrealized_pnl or 0.0) / cost) if cost else None,
                "pinned": sym in pins,
                "stop_level": stop,
                "stop_distance_pct": ((mark / float(stop) - 1.0) if (stop and mark) else None),
            }
        )
    rows.sort(key=lambda r: -(r["value_base"] or 0.0))
    return rows


# --------------------------------------------------------------------- cycles


def cycles_block(runner_store: Any, limit: int = 16) -> list[dict[str, Any]]:
    rows = runner_store.recent_cycles(limit=limit)
    return [
        {
            "t": r["ts"].isoformat(),
            "status": r["status"],
            "orders": r["orders_submitted"],
            "fills": r["fills_received"],
            "error": r["error"],
            "duration_ms": r["duration_ms"],
        }
        for r in rows
    ]


def watch_block(
    state_dir: Path,
    runner_store: Any,
    *,
    cron: str,
    tz: str,
    now: datetime | None = None,
    settings: Any = None,
) -> dict[str, Any]:
    """What the watchdogs would say right now (evaluated, never alerted)."""
    from trading.runner.cadence import cadence_from
    from trading.runtime import cycle_watch

    now = now or _now()
    every, anchor = cadence_from(settings)
    findings: list[dict[str, str]] = []
    try:
        cycles = runner_store.recent_cycles(limit=200)
        if cron:
            for f in cycle_watch.evaluate(
                cycles, cron=cron, tz=tz, now=now, every_weeks=every, anchor=anchor
            ):
                findings.append({"level": f.level, "message": f.message, "key": f.key})
    except Exception as e:
        logger.bind(component="dashboard").warning(f"cycle watch failed: {e}")
    recon = _read_json(state_dir / "broker_reconcile.json") or {}
    return {
        "findings": findings,
        "reconcile_attention": list(recon.get("attention") or []),
    }


# ------------------------------------------------------------------- schedule


def schedule_block(
    settings: Any, *, cron: str, tz: str, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Next fire time of every job an operator plans around, server-computed.

    The page used to parse the cron in JavaScript as UTC and hardcode six
    other job times, which were wrong the day the schedule moved. These are
    APScheduler's own triggers, so they cannot disagree with the runner.
    """
    from apscheduler.triggers.cron import CronTrigger

    from trading.runner.cadence import cadence_from, gate
    from trading.runner.runner import _historian_trigger, _precycle_trigger

    now = now or _now()
    jobs: list[tuple[str, str, Any]] = []
    every, anchor = cadence_from(settings)

    def gated(trig: Any) -> Any:
        return gate(trig, every_weeks=every, anchor=anchor, tz=tz)

    if cron:
        try:
            label = "Rebalance cycle" + (f" (every {every} weeks)" if every > 1 else "")
            jobs.append(("cycle", label, gated(CronTrigger.from_crontab(cron, timezone=tz))))
            lead = int(getattr(settings, "pm_pre_cycle_lead_minutes", 45) or 45)
            pm = _precycle_trigger(cron, tz, lead_minutes=lead)
            if pm is not None:
                jobs.append(("pm", "Agent PM decision", gated(pm)))
            ready = _precycle_trigger(cron, tz, lead_minutes=60)
            if ready is not None:
                jobs.append(("broker_ready", "Broker readiness check", gated(ready)))
        except Exception as e:
            logger.bind(component="dashboard").warning(f"cycle schedule failed: {e}")
    jobs.extend(
        [
            (
                "committee",
                "Committee",
                CronTrigger.from_crontab(
                    os.getenv("AGENTS_COMMITTEE_CRON", "0 13 * * MON,FRI"), timezone=NY
                ),
            ),
            (
                "reconcile",
                "Broker reconciliation",
                CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=NY),
            ),
            ("grading", "Prediction grading", CronTrigger(hour=18, minute=45, timezone=NY)),
            ("curator", "Learning Curator", _historian_trigger()),
        ]
    )
    out: list[dict[str, Any]] = []
    for key, label, trig in jobs:
        try:
            nxt = trig.get_next_fire_time(None, now)
        except Exception:
            nxt = None
        if nxt is not None:
            out.append({"key": key, "label": label, "at": nxt.astimezone(timezone.utc).isoformat()})
    out.sort(key=lambda j: j["at"])
    return out


# -------------------------------------------------------------------- pending


def pending_block(state_dir: Path, order_store: Any | None) -> dict[str, Any]:
    """Anything waiting on a human: approval, a staged command, open orders."""
    from trading.bot import confirmations

    out: dict[str, Any] = {
        "approval": None,
        "staged": None,
        "open_orders": [],
        "halted_review": None,
    }
    pending = _read_json(state_dir / "cycle_approval_pending.json")
    if isinstance(pending, dict):
        plan = pending.get("plan") or {}
        raw_rows = plan.get("orders")
        rows: list[Any] = raw_rows if isinstance(raw_rows, list) else []
        out["approval"] = {
            "id": pending.get("id"),
            "ts": pending.get("ts"),
            "currency": pending.get("currency"),
            "deploy_pct": pending.get("deploy_pct"),
            "defensive": bool(pending.get("defensive_reduce_only")),
            "orders": [
                {k: r.get(k) for k in ("symbol", "side", "quantity", "notional_base", "context")}
                for r in rows
                if isinstance(r, dict)
            ][:30],
        }
    staged = confirmations.peek(state_dir)
    if staged is not None:
        out["staged"] = {
            "kind": staged.kind,
            "summary": staged.summary,
            "expires_at": (staged.staged_at + confirmations.TTL).isoformat(),
        }
    review = _read_json(state_dir / "cycle_halted_review.json")
    if isinstance(review, dict):
        out["halted_review"] = {"ts": review.get("ts"), "halt_reason": review.get("halt_reason")}
    if order_store is not None:
        try:
            for order, status, _broker_id in order_store.open_orders()[:30]:
                out["open_orders"].append(
                    {
                        "id": order.client_order_id,
                        "symbol": order.instrument.symbol,
                        "side": order.side.value,
                        "qty": order.quantity,
                        "status": status.value,
                        "created_at": order.created_at.isoformat(),
                        "age_days": round((_now() - order.created_at).total_seconds() / 86400, 1),
                    }
                )
        except Exception as e:
            logger.bind(component="dashboard").warning(f"open orders failed: {e}")
    return out


# --------------------------------------------------------------------- agents


_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def agents_block(memory_db: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """Are the agents any good? Skill vs a coin flip, calibration, trend.

    ``skill = 1 - brier / 0.25``: 0 is a coin flip (always 50%), positive is
    real skill, negative means the agent would score better by always
    saying 50%. ``overconfidence`` is stated confidence minus hit rate.
    """
    now = now or _now()
    conn = _ro_connect(memory_db)
    if conn is None:
        return {"agents": [], "bins": [], "note": "no memory database yet"}
    try:
        graded = conn.execute(
            "SELECT agent, confidence, outcome, brier, graded_ts FROM predictions "
            "WHERE outcome IN ('hit','miss') AND brier IS NOT NULL"
        ).fetchall()
        open_rows = conn.execute("SELECT due_ts FROM predictions WHERE outcome IS NULL").fetchall()
    finally:
        conn.close()
    recent_cut = (now - timedelta(days=30)).timestamp()
    by_agent: dict[str, dict[str, Any]] = {}
    for r in graded:
        a = by_agent.setdefault(
            r["agent"],
            {
                "agent": r["agent"],
                "n": 0,
                "hits": 0,
                "conf": 0.0,
                "brier": 0.0,
                "rn": 0,
                "rbrier": 0.0,
            },
        )
        a["n"] += 1
        a["hits"] += 1 if r["outcome"] == "hit" else 0
        a["conf"] += float(r["confidence"])
        a["brier"] += float(r["brier"])
        if (r["graded_ts"] or 0) >= recent_cut:
            a["rn"] += 1
            a["rbrier"] += float(r["brier"])
    agents = []
    for a in by_agent.values():
        n = a["n"]
        brier = a["brier"] / n
        rb = (a["rbrier"] / a["rn"]) if a["rn"] else None
        agents.append(
            {
                "agent": a["agent"],
                "n": n,
                "hit_rate": a["hits"] / n,
                "mean_confidence": a["conf"] / n,
                "brier": brier,
                "skill": 1.0 - brier / COIN_FLIP_BRIER,
                "overconfidence": a["conf"] / n - a["hits"] / n,
                "recent_n": a["rn"],
                "recent_skill": None if rb is None else 1.0 - rb / COIN_FLIP_BRIER,
            }
        )
    agents.sort(key=lambda x: -x["skill"])
    bins = []
    for lo, hi in _BINS:
        sel = [r for r in graded if lo <= float(r["confidence"]) < hi]
        if sel:
            bins.append(
                {
                    "lo": lo,
                    "hi": min(hi, 1.0),
                    "n": len(sel),
                    "mean_confidence": sum(float(r["confidence"]) for r in sel) / len(sel),
                    "hit_rate": sum(1 for r in sel if r["outcome"] == "hit") / len(sel),
                }
            )
    week = (now + timedelta(days=7)).timestamp()
    return {
        "agents": agents,
        "bins": bins,
        "graded": len(graded),
        "open": len(open_rows),
        "due_7d": sum(1 for r in open_rows if (r["due_ts"] or 0) <= week),
        "overall_skill": (
            1.0 - (sum(float(r["brier"]) for r in graded) / len(graded)) / COIN_FLIP_BRIER
            if graded
            else None
        ),
        "note": "",
    }


def lessons_block(memory_db: Path) -> dict[str, Any]:
    """What the system believes, and how much measured evidence backs it."""
    conn = _ro_connect(memory_db)
    if conn is None:
        return {"lessons": [], "counts": {}}
    try:
        rows = conn.execute(
            """SELECT l.id, l.status, l.statement, l.created_ts, l.support, l.contradict,
                      SUM(CASE WHEN e.evidence_kind='outcome' AND e.relation='supports' THEN 1 ELSE 0 END) AS o_sup,
                      SUM(CASE WHEN e.evidence_kind='outcome' AND e.relation='contradicts' THEN 1 ELSE 0 END) AS o_con
               FROM lessons l LEFT JOIN lesson_evidence e ON e.lesson_id = l.id
               GROUP BY l.id ORDER BY l.created_ts DESC LIMIT 40"""
        ).fetchall()
    finally:
        conn.close()
    counts: dict[str, int] = {}
    lessons = []
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        lessons.append(
            {
                "id": r["id"],
                "status": r["status"],
                "statement": r["statement"],
                "created_at": datetime.fromtimestamp(
                    float(r["created_ts"]), tz=timezone.utc
                ).isoformat(),
                "measured_support": int(r["o_sup"] or 0),
                "measured_contradict": int(r["o_con"] or 0),
                "votes_support": int(r["support"] or 0),
                "votes_contradict": int(r["contradict"] or 0),
            }
        )
    return {"lessons": lessons, "counts": counts, "promotion_net_required": 3}


def edge_block(memory_dir: Path) -> dict[str, Any]:
    """Does the selection step add anything? Picks minus passes, per origin."""
    from trading.memory.store import MemoryStore

    if not (memory_dir / "memory.db").exists():
        return {"by_origin_21d": [], "by_origin_5d": []}
    mem = MemoryStore(memory_dir)
    return {
        "by_origin_21d": mem.edge_report(leg_days=21),
        "by_origin_5d": mem.edge_report(leg_days=5),
    }


# ------------------------------------------------------------------------ llm


def llm_block(
    path: Path, *, days: int = 7, now: datetime | None = None, max_lines: int = 20000
) -> dict[str, Any]:
    """LLM calls, tokens, errors and latency over the last ``days``."""
    now = now or _now()
    if not path.exists():
        return {"days": [], "totals": {}, "by_tier": {}}
    cut = now - timedelta(days=days)
    lines: Iterable[str]
    try:
        lines = path.read_text(errors="replace").splitlines()[-max_lines:]
    except OSError:
        return {"days": [], "totals": {}, "by_tier": {}}
    per_day: dict[str, dict[str, Any]] = {}
    by_tier: dict[str, int] = {}
    lat: list[float] = []
    tot: dict[str, float | None] = {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}
    for line in lines:
        try:
            rec = json.loads(line)
            ts = datetime.fromisoformat(str(rec.get("ts")).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None or ts < cut:
            continue
        day = ts.date().isoformat()
        d = per_day.setdefault(
            day, {"day": day, "calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}
        )
        err = bool(rec.get("error_type") or (rec.get("http_status") or 200) >= 400)
        for tgt in (d, tot):
            tgt["calls"] = (tgt["calls"] or 0) + 1
            tgt["errors"] = (tgt["errors"] or 0) + (1 if err else 0)
            tgt["input_tokens"] = (tgt["input_tokens"] or 0) + int(rec.get("input_tokens") or 0)
            tgt["output_tokens"] = (tgt["output_tokens"] or 0) + int(rec.get("output_tokens") or 0)
        tier = str(rec.get("tier") or "?")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        if rec.get("latency_ms") is not None:
            with contextlib.suppress(TypeError, ValueError):
                lat.append(float(rec["latency_ms"]))
    lat.sort()

    def _q(p: float) -> float | None:
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

    tot["p50_latency_ms"] = _q(0.5)
    tot["p95_latency_ms"] = _q(0.95)
    return {
        "days": sorted(per_day.values(), key=lambda x: x["day"]),
        "totals": tot,
        "by_tier": by_tier,
    }


# --------------------------------------------------------------------- market


def market_block(data_dir: Path, market_watch: dict[str, Any] | None) -> dict[str, Any]:
    """Market temperature: how stretched or discounted the index is.

    A PREVIEW of the wave-4 cash dial (docs/ROADMAP_WAVES.md §4.2), shown so
    the operator can build intuition before anything trades on it. Each
    component maps to [-1 discount, +1 stretched]; the composite is their
    mean over the components available.
    """
    import pandas as pd

    from trading.core.types import AssetClass, Instrument
    from trading.data.cache import ParquetCache

    out: dict[str, Any] = {"components": [], "preview": True}
    closes = None
    for ac in (AssetClass.ETF, AssetClass.EQUITY):
        try:
            df = ParquetCache(data_dir).read(Instrument(symbol="SPY", asset_class=ac), "1D")
        except Exception:
            df = None
        if df is not None and not df.empty:
            col = (
                "adj_close"
                if "adj_close" in df and df["adj_close"].notna().sum() > 200
                else "close"
            )
            closes = df[col].dropna()
            break
    comps: list[dict[str, Any]] = []

    def _clip(x: float) -> float:
        return max(-1.0, min(1.0, x))

    if closes is not None and len(closes) >= 210:
        last = float(closes.iloc[-1])
        sma200 = float(closes.iloc[-200:].mean())
        peak = float(closes.max())
        dd = last / peak - 1.0
        since_high = int(len(closes) - 1 - int(closes.reset_index(drop=True).idxmax()))
        low20_recent = bool(float(closes.iloc[-5:].min()) <= float(closes.iloc[-25:-5].min()))
        dist = last / sma200 - 1.0
        out.update(
            {
                "spy": last,
                "spy_sma200": sma200,
                "spy_vs_200d": dist,
                "drawdown_from_high": dd,
                "days_since_high": since_high,
                "new_20d_low_last_week": low20_recent,
                "spy_series": [
                    {"t": pd.Timestamp(str(ts)).strftime("%Y-%m-%d"), "v": round(float(v), 2)}
                    for ts, v in closes.iloc[-260:].items()
                ],
            }
        )
        comps.append(
            {"key": "trend", "label": "SPY vs 200-day", "value": dist, "score": _clip(dist / 0.10)}
        )
        comps.append(
            {
                "key": "drawdown",
                "label": "Distance from high",
                "value": dd,
                "score": _clip(1.0 + dd / 0.10),
            }
        )
    latest = (market_watch or {}).get("latest") or {}
    breadth = latest.get("pct_above_200")
    if breadth is not None:
        comps.append(
            {
                "key": "breadth",
                "label": "Members above 200-day",
                "value": breadth,
                "score": _clip((float(breadth) - 0.55) / 0.25),
            }
        )
    vix, vix3m = latest.get("vix"), latest.get("vix3m")
    if vix and vix3m:
        ratio = float(vix) / float(vix3m)
        out["vix_term_ratio"] = ratio
        comps.append(
            {
                "key": "vol_term",
                "label": "VIX / VIX3M",
                "value": ratio,
                "score": _clip((0.92 - ratio) / 0.08),
            }
        )
    out["components"] = comps
    if comps:
        comp = sum(c["score"] for c in comps) / len(comps)
        out["temperature"] = comp
        out["label"] = (
            "Stretched"
            if comp >= 0.6
            else "Warm"
            if comp >= 0.25
            else "Neutral"
            if comp > -0.35
            else "Discount"
        )
        out["stabilised"] = (
            None
            if "new_20d_low_last_week" not in out
            else (not out["new_20d_low_last_week"] and (out.get("vix_term_ratio") or 0) < 1.0)
        )
    return out
