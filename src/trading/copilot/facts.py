"""The copilot's "NOW" side — current portfolio/order/risk facts.

Everything reads the source SQLite files in ``mode=ro`` (same contract
as the dashboard: this module can never create, migrate, or mutate a
store the runner owns) and NOTHING imports the broker. Each fact
carries its own timestamp so the engine can cite data ages honestly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ro(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def positions_now(state_dir: Path, symbol: str | None = None) -> dict[str, Any]:
    """Latest account snapshot: equity, cash, positions (optionally one)."""
    conn = _ro(Path(state_dir) / "runner.db")
    if conn is None:
        return {"available": False}
    try:
        row = conn.execute(
            "SELECT ts, cash, equity, positions_json FROM account_snapshots "
            "ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"available": False}
    positions = []
    for p in json.loads(row["positions_json"]).values():
        sym = str(p.get("instrument", {}).get("symbol", "?")).upper()
        if symbol and sym != symbol.upper():
            continue
        positions.append(
            {
                "symbol": sym,
                "qty": p.get("quantity"),
                "avg_price": p.get("avg_price"),
                "unrealized_pnl": p.get("unrealized_pnl"),
                "realized_pnl": p.get("realized_pnl"),
            }
        )
    out: dict[str, Any] = {
        "available": True,
        "note": "the REAL momentum trading account (paper mode)",
        "as_of": datetime.fromtimestamp(row["ts"], tz=timezone.utc).isoformat(),
        "equity": row["equity"],
        "cash": row["cash"],
        "positions": positions,
    }
    # Precomputed so the LLM never derives its own percentages (it
    # invented a "70% deployed" from raw numbers, 2026-07-16).
    if row["equity"]:
        out["deployed_pct"] = round((1 - row["cash"] / row["equity"]) * 100, 1)
    return out


def orders_and_fills(
    state_dir: Path, symbol: str | None = None, *, limit: int = 10
) -> dict[str, Any]:
    """Recent orders (with status) and their fills, newest first."""
    conn = _ro(Path(state_dir) / "orders.db")
    if conn is None:
        return {"available": False}
    try:
        rows = conn.execute(
            "SELECT o.client_order_id, o.instrument_json, o.side, o.quantity, o.status, "
            "o.created_at, f.ts AS fill_ts, f.quantity AS fill_qty, f.price AS fill_price, "
            "f.commission FROM orders o LEFT JOIN fills f ON f.order_id = o.client_order_id "
            "ORDER BY o.created_at DESC LIMIT ?",
            (limit * 6,),
        ).fetchall()
    finally:
        conn.close()
    out = []
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    for r in rows:
        sym = str(json.loads(r["instrument_json"]).get("symbol", "?")).upper()
        if symbol and sym != symbol.upper():
            continue
        rec = {
            "order_id": r["client_order_id"],
            "symbol": sym,
            "side": r["side"],
            "qty": r["quantity"],
            "status": r["status"],
            "created": datetime.fromtimestamp(r["created_at"], tz=timezone.utc).isoformat(),
            "fill": (
                {
                    "ts": datetime.fromtimestamp(r["fill_ts"], tz=timezone.utc).isoformat(),
                    "qty": r["fill_qty"],
                    "price": r["fill_price"],
                    "commission": r["commission"],
                }
                if r["fill_ts"] is not None
                else None
            ),
        }
        # KNOWN BOOKKEEPING GAP (2026-07-16): fills are only promoted to
        # FILLED if they arrive within the submitting cycle — orders that
        # fill at the NEXT session's open stay 'submitted' in this table
        # forever. Flag it so the copilot never presents a stale status
        # as a working order.
        if r["status"] == "submitted" and r["fill_ts"] is None and now_ts - r["created_at"] > 86400:
            rec["status_note"] = (
                "STALE — recorded 'submitted' >1 day ago; almost certainly filled at the "
                "next open or expired (DAY order). Trust the position snapshot, not this status."
            )
        out.append(rec)
        if len(out) >= limit:
            break
    return {"available": True, "orders": out}


def pm_book(state_dir: Path, data_dir: Path | None = None) -> dict[str, Any]:
    """The agent PM's SIMULATED book — a separate virtual portfolio,
    never to be conflated with the real/paper momentum account.

    Raw holdings are fractional SHARE quantities (``w * equity / px``),
    which read like weights (``JPM: 0.1``) — the LLM presented them as
    portfolio weights (operator report, 2026-07-16). So when ``data_dir``
    is given each holding is marked at the last cached close and carries
    an explicit ``weight_pct``; the book carries ``deployed_pct``. The
    model is charter-bound to use these precomputed fields, never its
    own arithmetic.
    """
    path = Path(state_dir) / "agent_pm" / "portfolio.json"
    if not path.exists():
        return {"available": False}
    try:
        book = json.loads(path.read_text())
    except Exception:
        return {"available": False}
    hist = book.get("history", [])
    cash = book.get("cash")
    out: dict[str, Any] = {
        "available": True,
        "note": (
            "SIMULATED virtual book (paper-money experiment), not the trading "
            "account. Holdings are fractional SHARE quantities, NOT weights."
        ),
        "cash": cash,
        "last_marked_equity": hist[-1].get("equity") if hist else None,
        "last_mark_ts": hist[-1].get("t") if hist else None,
        "start_equity": book.get("start_equity"),
    }
    holdings: dict[str, float] = book.get("holdings", {}) or {}
    marked: dict[str, Any] = {}
    values: dict[str, float] = {}
    if data_dir is not None:
        for sym, qty in holdings.items():
            mkt = last_close(data_dir, sym)
            if mkt.get("available"):
                values[sym] = float(qty) * float(mkt["close"])
                marked[sym] = {
                    "shares": qty,
                    "last_close": mkt["close"],
                    "value": round(values[sym], 2),
                    "price_as_of": mkt["as_of"],
                }
    if values and cash is not None and len(values) == len(holdings):
        equity_now = float(cash) + sum(values.values())
        for sym, v in values.items():
            marked[sym]["weight_pct"] = round(v / equity_now * 100, 1)
        out["holdings"] = marked
        out["marked_equity_now"] = round(equity_now, 2)
        out["deployed_pct"] = round((1 - float(cash) / equity_now) * 100, 1)
    else:
        # No/partial prices: fall back to raw shares, loudly labeled.
        out["holdings_share_quantities_NOT_weights"] = holdings
    return out


def risk_now(state_dir: Path) -> dict[str, Any]:
    """Halt state + the last cycle's outcome."""
    out: dict[str, Any] = {}
    halt_path = Path(state_dir) / "halt.json"
    try:
        halt = json.loads(halt_path.read_text()) if halt_path.exists() else {}
        out["halted"] = bool(halt.get("halted"))
        out["halt_reason"] = halt.get("reason", "")
    except Exception:
        out["halted"] = None
    conn = _ro(Path(state_dir) / "runner.db")
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT ts, status, orders_submitted, fills_received FROM cycles "
                "ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                out["last_cycle"] = {
                    "ts": datetime.fromtimestamp(row["ts"], tz=timezone.utc).isoformat(),
                    "status": row["status"],
                    "orders_submitted": row["orders_submitted"],
                    "fills_received": row["fills_received"],
                }
        finally:
            conn.close()
    return out


def last_close(data_dir: Path, symbol: str) -> dict[str, Any]:
    """Most recent cached close for a symbol — cache only, no network.
    The copilot answers from what the system knows, at the age it knows it."""
    try:
        from trading.runtime.portfolio_stats import _read_close

        s = _read_close(Path(data_dir), symbol.upper())
        if s is None or s.empty:
            return {"available": False, "symbol": symbol.upper()}
        return {
            "available": True,
            "symbol": symbol.upper(),
            "close": round(float(s.iloc[-1]), 4),
            "as_of": str(s.index[-1])[:10],
            "change_5d_pct": (
                round((float(s.iloc[-1]) / float(s.iloc[-6]) - 1) * 100, 2) if len(s) > 6 else None
            ),
        }
    except Exception:
        return {"available": False, "symbol": symbol.upper()}
