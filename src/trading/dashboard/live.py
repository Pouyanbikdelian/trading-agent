"""Live-tab data: per-sleeve PnL, USD-converted curves, daily attribution.

Everything here is read-only and defensive — the dashboard must render
even when a sleeve has no data yet. SQLite files are opened in
``mode=ro`` so this module can never create or migrate a database the
runner owns (the dashboard's "reads only, writes nothing" contract).

Currency model (GO_LIVE.md §1): the IBKR account is CHF-based, so the
equity curve from ``runner.db`` is CHF. Benchmarks (SPY) and the PM sim
are USD. We convert the account curve to USD with a daily USDCHF series
(USD = CHF / USDCHF) so the race isn't polluted by the franc. Per-symbol
PnL is already USD (US-listed instruments), so sleeve cards need no
conversion.

Why "exclude today from daily curves": the last point of a daily curve
is the latest 60s snapshot, not a close — plotting it makes every
intraday wobble look like a daily drop. Today lives only in the
intraday series.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def _ro_conn(path: Path) -> sqlite3.Connection | None:
    """Open a SQLite file strictly read-only; None if absent/unreadable."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


# --------------------------------------------------------------- fills


def fills_with_symbols(orders_db: Path) -> list[dict[str, Any]]:
    """Every fill joined to its order's symbol and side, oldest first."""
    conn = _ro_conn(orders_db)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """SELECT f.ts, f.quantity, f.price, f.commission,
                      o.side, o.instrument_json
               FROM fills f JOIN orders o ON o.client_order_id = f.order_id
               ORDER BY f.ts ASC"""
        ).fetchall()
    except sqlite3.Error as e:
        logger.bind(component="dashboard").warning(f"fills query failed: {e}")
        return []
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            sym = json.loads(r["instrument_json"]).get("symbol", "?")
        except Exception:
            sym = "?"
        out.append(
            {
                "ts": float(r["ts"]),
                "symbol": str(sym).upper(),
                "qty": float(r["quantity"]),
                "price": float(r["price"]),
                "commission": float(r["commission"] or 0.0),
                "side": str(r["side"]).upper(),
            }
        )
    return out


def realized_by_symbol(fills: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average-cost realized PnL + fees per symbol from a fill stream.

    Average cost (not FIFO) matches how the simulator and IBKR's default
    P&L view book it, and is insensitive to partial-fill ordering.
    Long-only in practice, but the math handles shorts symmetrically.
    """
    book: dict[str, dict[str, float]] = {}
    for f in fills:
        b = book.setdefault(f["symbol"], {"qty": 0.0, "avg": 0.0, "realized": 0.0, "fees": 0.0})
        b["fees"] += f["commission"]
        signed = f["qty"] if f["side"] == "BUY" else -f["qty"]
        q0, a0 = b["qty"], b["avg"]
        if q0 == 0 or (q0 > 0) == (signed > 0):  # opening / adding
            total = q0 + signed
            b["avg"] = (a0 * q0 + f["price"] * signed) / total if total else 0.0
            b["qty"] = total
        else:  # reducing / closing / flipping
            closed = min(abs(signed), abs(q0))
            direction = 1.0 if q0 > 0 else -1.0
            b["realized"] += direction * closed * (f["price"] - a0)
            b["qty"] = q0 + signed
            if q0 * b["qty"] < 0:  # flipped through flat
                b["avg"] = f["price"]
            elif b["qty"] == 0:
                b["avg"] = 0.0
    return {
        s: {"realized": round(v["realized"], 2), "fees": round(v["fees"], 2)}
        for s, v in book.items()
    }


# ----------------------------------------------------------- fx / curves


def fetch_usdchf(data_dir: Path) -> dict[str, float]:
    """Daily USDCHF closes keyed by ISO date. Parquet cache first, then
    yfinance; empty dict if both fail (caller falls back to unconverted)."""
    try:
        from trading.runtime.portfolio_stats import _read_close

        s = _read_close(data_dir, "USDCHF")
        if s is not None and len(s) > 20:
            return {str(ix)[:10]: float(v) for ix, v in s.items()}
    except Exception:
        pass
    try:
        import yfinance as yf

        raw = yf.download("CHF=X", period="2y", auto_adjust=True, progress=False, threads=False)
        close = raw["Close"].dropna()
        # yf returns a DataFrame column per ticker on some versions.
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        return {str(ix)[:10]: float(v) for ix, v in close.items()}
    except Exception as e:
        logger.bind(component="dashboard").warning(f"USDCHF fetch failed: {e}")
        return {}


def convert_curve_to_usd(
    points: list[dict[str, Any]], fx: dict[str, float]
) -> list[dict[str, Any]]:
    """CHF points [{t, v}] → USD via same-day (or last known) USDCHF.
    Points before the first known rate are dropped rather than guessed."""
    if not fx:
        return points
    dates = sorted(fx)
    out: list[dict[str, Any]] = []
    rate: float | None = None
    i = 0
    for p in points:
        while i < len(dates) and dates[i] <= p["t"]:
            rate = fx[dates[i]]
            i += 1
        if rate:
            out.append({"t": p["t"], "v": round(p["v"] / rate, 2)})
    return out


def daily_curve(
    curve: list[tuple[datetime, float]], *, exclude_today: bool = True
) -> list[dict[str, Any]]:
    """Last-point-of-day series; today excluded (it's a snapshot, not a
    close)."""
    daily: dict[str, float] = {}
    for ts, eq in curve:
        daily[ts.date().isoformat()] = float(eq)
    if exclude_today:
        daily.pop(datetime.now(tz=timezone.utc).date().isoformat(), None)
    return [{"t": t, "v": v} for t, v in sorted(daily.items())]


def daily_pnl_bars(points: list[dict[str, Any]], n: int = 60) -> list[dict[str, Any]]:
    """Day-over-day equity diffs for the last ``n`` closed days."""
    from itertools import pairwise

    bars = [{"t": b["t"], "v": round(b["v"] - a["v"], 2)} for a, b in pairwise(points)]
    return bars[-n:]


# ---------------------------------------------------------- attribution


def _positions_at(conn: sqlite3.Connection, day: str) -> dict[str, dict[str, float]] | None:
    """Per-symbol {unrealized, realized, qty} from the last snapshot of
    ``day`` (ISO date), or None if that day has no snapshot."""
    row = conn.execute(
        "SELECT positions_json FROM account_snapshots "
        "WHERE date(ts, 'unixepoch') = ? ORDER BY ts DESC LIMIT 1",
        (day,),
    ).fetchone()
    if row is None:
        return None
    try:
        raw = json.loads(row["positions_json"])
    except Exception:
        return None
    out: dict[str, dict[str, float]] = {}
    for _key, p in raw.items():
        sym = str(p.get("instrument", {}).get("symbol", "?")).upper()
        out[sym] = {
            "unrealized": float(p.get("unrealized_pnl", 0.0)),
            "realized": float(p.get("realized_pnl", 0.0)),
            "qty": float(p.get("quantity", 0.0)),
        }
    return out


def attribution_today(runner_db: Path, fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-symbol PnL contribution for today.

    Contribution = Δunrealized (vs the previous session's last snapshot)
    + realized booked today − fees paid today. Realized-today comes from
    the fill ledger, NOT snapshot realized_pnl deltas: a position that
    closed today drops out of the snapshot and would take its realized
    history with it. For a symbol that left the book, Δunrealized is
    −prev.unrealized (yesterday's paper gain converts into today's
    realized figure — counting both would double-book it)."""
    conn = _ro_conn(runner_db)
    if conn is None:
        return []
    try:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        days = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT date(ts,'unixepoch') AS d FROM account_snapshots "
                "WHERE d <= ? ORDER BY d DESC LIMIT 2",
                (today,),
            ).fetchall()
        ]
        if not days or days[0] != today:
            return []
        now = _positions_at(conn, today) or {}
        prev = (_positions_at(conn, days[1]) if len(days) > 1 else None) or {}
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    midnight = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0).timestamp()
    before = realized_by_symbol([f for f in fills if f["ts"] < midnight])
    total = realized_by_symbol(fills)
    realized_today = {
        s: round(total[s]["realized"] - before.get(s, {"realized": 0.0})["realized"], 2)
        for s in total
    }
    fees_today: dict[str, float] = {}
    for f in fills:
        if f["ts"] >= midnight:
            fees_today[f["symbol"]] = fees_today.get(f["symbol"], 0.0) + f["commission"]

    rows: list[dict[str, Any]] = []
    for sym in sorted({*now, *prev, *fees_today}):
        n = now.get(sym, {"unrealized": 0.0, "qty": 0.0})
        p = prev.get(sym, {"unrealized": 0.0, "qty": 0.0})
        pnl = (n["unrealized"] - p["unrealized"]) + realized_today.get(sym, 0.0)
        fee = fees_today.get(sym, 0.0)
        if abs(pnl) < 0.005 and abs(fee) < 0.005 and n["qty"] == 0:
            continue
        rows.append(
            {
                "symbol": sym,
                "pnl": round(pnl, 2),
                "fees": round(fee, 2),
                "qty": n["qty"],
            }
        )
    rows.sort(key=lambda r: r["pnl"])
    return rows


# -------------------------------------------------------------- assembly


def _momentum_sleeve(state_dir: Path, fx: dict[str, float], label: str) -> dict[str, Any] | None:
    """One momentum-runner sleeve (paper or live) from a state dir."""
    from trading.runner.state import RunnerStore

    runner_db = state_dir / "runner.db"
    if not runner_db.exists():
        return None
    try:
        store = RunnerStore(runner_db)
        curve = store.equity_curve()
        snap = store.latest_snapshot()
    except Exception as e:
        logger.bind(component="dashboard").warning(f"sleeve {label}: {e}")
        return None
    base_ccy = getattr(snap, "base_currency", None) or "USD"
    points = daily_curve(curve)
    points_usd = convert_curve_to_usd(points, fx) if base_ccy == "CHF" else points
    fills = fills_with_symbols(state_dir / "orders.db")
    per_symbol = realized_by_symbol(fills)
    realized = round(sum(v["realized"] for v in per_symbol.values()), 2)
    fees = round(sum(v["fees"] for v in per_symbol.values()), 2)
    unrealized = round(sum(p.unrealized_pnl for p in (snap.positions if snap else {}).values()), 2)
    bars = daily_pnl_bars(points_usd)
    return {
        "label": label,
        "currency": base_ccy,
        "equity": snap.equity if snap else None,
        "equity_usd": points_usd[-1]["v"] if points_usd else None,
        "curve_usd": points_usd,
        "daily_pnl_usd": bars,
        "day_pnl_usd": bars[-1]["v"] if bars else None,
        "realized_usd": realized,
        "unrealized_usd": unrealized,
        "fees_usd": fees,
        "attribution_today": attribution_today(runner_db, fills),
    }


def _pm_sleeve(state_dir: Path) -> dict[str, Any] | None:
    """The PM sim sleeve — already USD, already daily-marked."""
    path = state_dir / "agent_pm" / "portfolio.json"
    if not path.exists():
        return None
    try:
        from trading.agents.pm import performance

        book = json.loads(path.read_text())
        perf = performance(state_dir)
    except Exception:
        return None
    hist = book.get("history", [])
    today = datetime.now(tz=timezone.utc).date().isoformat()
    curve = [
        {"t": str(h["t"])[:10], "v": float(h["equity"])}
        for h in hist
        if str(h.get("t", ""))[:10] != today
    ]
    bars = daily_pnl_bars(curve)
    base = perf.get("start_equity")
    return {
        "label": "agent PM (sim)",
        "currency": "USD",
        "equity": perf.get("equity"),
        "equity_usd": perf.get("equity"),
        "curve_usd": curve,
        "daily_pnl_usd": bars,
        "day_pnl_usd": bars[-1]["v"] if bars else None,
        "realized_usd": None,  # virtual book — no fill ledger
        "unrealized_usd": round(perf["equity"] - base, 2) if perf.get("equity") and base else None,
        "fees_usd": None,
        "return_pct": perf.get("return_pct"),
        "spy_return_pct": perf.get("spy_return_pct"),
        "attribution_today": [],
    }


def build_live(state_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Everything the Live tab renders. Sleeves are ALWAYS separate series
    — live and paper books must never merge (GO_LIVE.md §1)."""
    import os

    from trading.core.config import settings

    fx = fetch_usdchf(data_dir)
    env = str(settings.trading_env)
    sleeves: list[dict[str, Any]] = []
    s = _momentum_sleeve(Path(state_dir), fx, f"momentum ({env})")
    if s:
        sleeves.append(s)
    # A second, live state dir may exist alongside the paper one (GO_LIVE
    # §3 mandates a fresh state dir for the live book). Point
    # DASHBOARD_LIVE_STATE_DIR at it and it renders as its own sleeve.
    other = os.getenv("DASHBOARD_LIVE_STATE_DIR", "")
    if other and Path(other) != Path(state_dir):
        s2 = _momentum_sleeve(Path(other), fx, "momentum (live)")
        if s2:
            sleeves.append(s2)
    pm = _pm_sleeve(Path(state_dir))
    if pm:
        sleeves.append(pm)

    # SPY benchmark reuses the PM's daily marks (free, no extra fetch).
    spy: list[dict[str, Any]] = []
    try:
        book = json.loads((Path(state_dir) / "agent_pm" / "portfolio.json").read_text())
        today = datetime.now(tz=timezone.utc).date().isoformat()
        spy = [
            {"t": str(h["t"])[:10], "v": float(h["spy"])}
            for h in book.get("history", [])
            if h.get("spy") and str(h.get("t", ""))[:10] != today
        ]
    except Exception:
        pass

    return {
        "env": env,
        "usdchf": (sorted(fx.items())[-1][1] if fx else None),
        "fx_ok": bool(fx),
        "sleeves": sleeves,
        "spy": spy,
    }
