"""Dashboard server — stdlib HTTP, basic auth, read-only over state.

Endpoints:
  GET /             single-page UI (static/index.html + ECharts CDN)
  GET /api/summary  everything the page renders, one JSON blob

Auth: HTTP Basic. Credentials from DASHBOARD_USER / DASHBOARD_PASS env
vars (default user 'yan', no default password — server refuses to start
without DASHBOARD_PASS so an open dashboard can't happen by accident).

Reads ONLY: runner.db (equity curve, snapshot), monitor state JSONs,
holds/k override, last committee digest, the memory store, and the
parquet cache for 52-week percentiles. Writes nothing. The order path
is untouched and untouchable from here.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trading.core.clock import artifact_age_seconds
from trading.core.logging import logger
from trading.dashboard.live import convert_curve_to_usd, fetch_usdchf


def _effective_watchlist(state_dir: Path) -> list[str]:
    """Resolve the dashboard list without letting an overlay fault hide config.

    The bot fails closed if its operator-state JSON is malformed, because it
    might otherwise overwrite the only proposal ledger.  The dashboard is
    read-only, so its safer degradation is to keep charting the versioned
    baseline and log that operator additions/removals are temporarily absent.
    """
    from trading.bot.desk import WATCHLIST_OVERRIDES_FILE, WatchlistStore
    from trading.core.config import PROJECT_ROOT

    watchlist = WatchlistStore(
        PROJECT_ROOT / "config" / "watchlist.yaml",
        state_dir / WATCHLIST_OVERRIDES_FILE,
    )
    try:
        return watchlist.items()
    except ValueError as e:
        logger.bind(component="dashboard").warning(f"operator watchlist unavailable: {e}")
        return watchlist.baseline_items()


def account_curve_usd(
    points: list[dict[str, Any]], base_currency: str, fx: dict[str, float]
) -> tuple[list[dict[str, Any]], bool]:
    """The account's daily curve in USD, and whether it really is USD.

    The Portfolio tab's race plots this line against SPY and the PM sim,
    both USD, while the IBKR account is CHF-based (GO_LIVE.md §1). Left
    unconverted, every USDCHF move renders as strategy alpha — the Live
    tab has converted since 2026-07-09, this curve had not.

    Returns the raw curve with ``False`` when conversion is needed but
    impossible. Plotting francs against dollars is only acceptable if the
    page says so, which is what the flag is for.
    """
    if (base_currency or "USD").upper() == "USD":
        return points, True
    converted = convert_curve_to_usd(points, fx) if fx else []
    # convert_curve_to_usd drops points that predate its first known rate,
    # so a short FX series can empty the curve outright. An account line
    # that silently disappears reads as "flat"; a flagged CHF one does not.
    if not converted:
        return points, False
    return converted, True


def _add_cockpit_blocks(out: dict[str, Any], state_dir: Path, data_dir: Path) -> None:
    """The operator blocks (dashboard/cockpit.py), each isolated.

    One failing source must blank one panel, never the page — the same
    contract as every other section of ``build_summary``.
    """
    from trading.dashboard import cockpit

    settings_obj: Any = None
    try:
        from trading.core.config import get_settings

        settings_obj = get_settings()
    except Exception as e:
        logger.bind(component="dashboard").warning(f"settings unavailable: {e}")
    snapshot = None
    runner_store = None
    try:
        from trading.runner.state import RunnerStore

        if (state_dir / "runner.db").exists():
            runner_store = RunnerStore(state_dir / "runner.db")
            snapshot = runner_store.latest_snapshot()
    except Exception as e:
        logger.bind(component="dashboard").warning(f"snapshot unavailable: {e}")
    cron, tz = out.get("cycle_cron") or "", out.get("cycle_tz") or "UTC"
    order_store = None
    if (state_dir / "orders.db").exists():
        try:
            from trading.execution.store import OrderStore

            order_store = OrderStore(state_dir / "orders.db")
        except Exception:
            order_store = None

    blocks: dict[str, Any] = {
        "status": lambda: cockpit.status_block(state_dir, settings_obj),
        "risk": lambda: cockpit.risk_block(state_dir, settings_obj, snapshot),
        "positions": lambda: cockpit.positions_block(state_dir, snapshot),
        "equity": lambda: cockpit.equity_block(state_dir / "runner.db", state_dir),
        "movers": lambda: cockpit.movers_block(state_dir / "runner.db", state_dir),
        "cycles": lambda: cockpit.cycles_block(runner_store) if runner_store else [],
        "watch": lambda: (
            cockpit.watch_block(state_dir, runner_store, cron=cron, tz=tz)
            if runner_store
            else {"findings": [], "reconcile_attention": []}
        ),
        "schedule": lambda: cockpit.schedule_block(settings_obj, cron=cron, tz=tz),
        "pending": lambda: cockpit.pending_block(state_dir, order_store),
        "agents": lambda: cockpit.agents_block(state_dir / "memory" / "memory.db"),
        "lessons_v2": lambda: cockpit.lessons_block(state_dir / "memory" / "memory.db"),
        "edge": lambda: cockpit.edge_block(state_dir / "memory"),
        "llm": lambda: cockpit.llm_block(state_dir / "llm_usage.jsonl"),
        "market": lambda: cockpit.market_block(data_dir, out.get("market_watch")),
    }
    for key, build in blocks.items():
        try:
            out[key] = build()
        except Exception as e:
            logger.bind(component="dashboard").warning(f"{key} block failed: {e}")
            out[key] = {} if key not in {"positions", "cycles", "schedule"} else []


def build_summary(state_dir: Path, data_dir: Path) -> dict[str, Any]:
    """One JSON blob with everything the page shows. Defensive: each
    section degrades to empty rather than failing the whole payload."""
    from trading.agents.context import build_context
    from trading.memory.store import MemoryStore
    from trading.runner.state import RunnerStore

    out: dict[str, Any] = {"generated_at": datetime.now(tz=timezone.utc).isoformat()}

    # Book, monitors, holds — reuse the committee's context pass.
    try:
        # The committee needs the live ladder; a dashboard refresh does not.
        # Building it scans hundreds of parquet files and can make the UI look
        # hung on a constrained VPS. The dashboard remains a state reader.
        out["context"] = build_context(state_dir, data_dir, include_candidate_ladder=False)
    except Exception as e:
        logger.bind(component="dashboard").warning(f"context failed: {e}")
        out["context"] = {}

    # Equity curves: one point per day for range views, plus today's
    # intraday points (snapshots land every 60s) for the "today" view.
    try:
        curve = RunnerStore(state_dir / "runner.db").equity_curve()
        daily: dict[str, float] = {}
        for ts, eq in curve:
            daily[ts.date().isoformat()] = float(eq)  # last point of each day wins
        today = datetime.now(tz=timezone.utc).date().isoformat()
        # Today's "close" is really the latest 60s snapshot — plotting it on
        # a daily curve makes intraday wobble look like a daily drop
        # (GO_LIVE.md §1). Today lives only in the intraday series below.
        daily.pop(today, None)
        out["equity_curve"] = [{"t": k, "v": v} for k, v in sorted(daily.items())]
        intraday = [(ts, eq) for ts, eq in curve if ts.date().isoformat() == today]
        step = max(1, len(intraday) // 300)
        out["equity_today"] = [{"t": ts.isoformat(), "v": float(eq)} for ts, eq in intraday[::step]]
    except Exception as e:
        logger.bind(component="dashboard").warning(f"equity curve failed: {e}")
        out["equity_curve"] = []
        out["equity_today"] = []

    # The same curve in USD, for the race that compares it with SPY and
    # the PM sim. One USDCHF fetch per payload, handed to the Live tab
    # below: two tabs converting the same book at rates fetched seconds
    # apart is its own way of inventing performance.
    fx: dict[str, float] = {}
    try:
        fx = fetch_usdchf(data_dir, allow_network=False)
        snap = RunnerStore(state_dir / "runner.db").latest_snapshot()
        base_ccy = (snap.base_currency if snap else None) or "USD"
        out["equity_currency"] = base_ccy
        out["equity_curve_usd"], out["equity_usd_ok"] = account_curve_usd(
            out["equity_curve"], base_ccy, fx
        )
    except Exception as e:
        logger.bind(component="dashboard").warning(f"usd equity curve failed: {e}")
        out["equity_currency"] = "USD"
        out["equity_curve_usd"] = out["equity_curve"]
        out["equity_usd_ok"] = False
    # The raw rate, so the Performance panel can price every line in ONE
    # currency. A franc investor comparing a CHF book with SPY in dollars
    # reads the USDCHF move as skill (or as a loss); the page converts
    # both ways from this one series, the same one the curve above used.
    out["usdchf"] = [{"t": t, "v": round(v, 5)} for t, v in sorted(fx.items())[-1300:]]

    # Agent PM (simulated sleeve): daily-marked equity history + book.
    try:
        pm_path = state_dir / "agent_pm" / "portfolio.json"
        pm = json.loads(pm_path.read_text()) if pm_path.exists() else {}
        last_path = state_dir / "agent_pm" / "last_run.json"
        out["agent_pm"] = {
            "history": pm.get("history", []),
            "holdings": pm.get("holdings", {}),
            "cash": pm.get("cash"),
            "start_equity": pm.get("start_equity"),
            "last_run": json.loads(last_path.read_text()) if last_path.exists() else {},
        }
    except Exception as e:
        logger.bind(component="dashboard").warning(f"agent_pm failed: {e}")
        out["agent_pm"] = {}

    # Last committee digest (already persisted for /detail).
    try:
        p = state_dir / "last_committee.json"
        out["committee"] = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        out["committee"] = {}

    # Market watch history (macro tab).
    try:
        mw = state_dir / "market_watch.json"
        out["market_watch"] = json.loads(mw.read_text()) if mw.exists() else {}
    except Exception:
        out["market_watch"] = {}

    # Holdings + watchlist: 6 months of closes per symbol, normalized
    # client-side. Held names from the snapshot; extras from the static
    # config watchlist plus approval-gated operator overrides in state. The
    # bot mounts config read-only, so this merge is what makes Telegram edits
    # visible without broadening the bot's container authority. Cache misses
    # remain explicit rather than triggering network I/O in an HTTP request.
    try:
        from trading.runtime.portfolio_stats import _read_close

        held = [p["symbol"] for p in out.get("context", {}).get("positions", [])]
        wl: list[str] = []
        try:
            wl = _effective_watchlist(state_dir)
        except Exception:
            wl = []
        symbols = list(dict.fromkeys([*held, *wl]))[:24]
        series: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        for sym in symbols:
            s = _read_close(data_dir, sym)
            if s is not None and len(s) > 20:
                tail = s.iloc[-126:]
                series[sym] = [
                    {"t": str(ix)[:10], "v": round(float(v), 4)} for ix, v in tail.items()
                ]
            else:
                missing.append(sym)
        out["tickers"] = {
            "held": held,
            "watchlist": wl,
            "series": series,
            "missing": missing,
        }
    except Exception as e:
        logger.bind(component="dashboard").warning(f"tickers failed: {e}")
        out["tickers"] = {}

    # Economy (FRED) — full trimmed history for the Economy tab.
    try:
        ec = state_dir / "econ_watch.json"
        out["econ"] = json.loads(ec.read_text()) if ec.exists() else {}
    except Exception:
        out["econ"] = {}

    # Live tab: per-sleeve PnL, USD curves, daily attribution.
    try:
        from trading.dashboard.live import build_live

        out["live"] = build_live(state_dir, data_dir, fx=fx)
    except Exception as e:
        logger.bind(component="dashboard").warning(f"live tab failed: {e}")
        out["live"] = {}

    # Rotation tab: RRG trails, regime ribbon, radar alerts.
    try:
        from trading.dashboard.rotation import build_rotation

        out["rotation"] = build_rotation(state_dir, data_dir, allow_network=False)
    except Exception as e:
        logger.bind(component="dashboard").warning(f"rotation failed: {e}")
        out["rotation"] = {}

    # News + sector momentum (the scout's inputs — worth eyeballing raw).
    try:
        nw = state_dir / "news.json"
        news = json.loads(nw.read_text()) if nw.exists() else {}
        out["news"] = {
            "t": news.get("t"),
            "headlines": (news.get("headlines") or [])[:12],
            "sector_momentum": news.get("sector_momentum") or {},
        }
    except Exception:
        out["news"] = {}

    # Committee posture history — the debate's trajectory over time.
    try:
        from trading.memory.store import MemoryStore

        hist = MemoryStore(state_dir / "memory").journal_tail(30, kind="committee")
        out["committee_history"] = [
            {
                "t": e["ts"].isoformat(),
                "posture": (e["payload"].get("ruling") or {}).get("posture", "neutral"),
                "dissent": e["payload"].get("disagreement", 0),
            }
            for e in reversed(hist)
        ]
    except Exception:
        out["committee_history"] = []

    # Ops: halt state + data freshness (age in minutes per state file).
    try:
        now = datetime.now(tz=timezone.utc).timestamp()

        def _age(p: Path) -> int | None:
            return int((now - p.stat().st_mtime) / 60) if p.exists() else None

        def _db_age(p: Path) -> int | None:
            secs = artifact_age_seconds(p, now=now)
            return None if secs is None else int(secs / 60)

        halt = {}
        hp = state_dir / "halt.json"
        if hp.exists():
            halt = json.loads(hp.read_text())
        out["ops"] = {
            "halted": bool(halt.get("halted")),
            "halt_reason": halt.get("reason", ""),
            "ages_min": {
                "news": _age(state_dir / "news.json"),
                "market_watch": _age(state_dir / "market_watch.json"),
                "committee": _age(state_dir / "last_committee.json"),
                "pm_book": _age(state_dir / "agent_pm" / "portfolio.json"),
                "snapshot": _db_age(state_dir / "runner.db"),
            },
        }
    except Exception:
        out["ops"] = {}

    # The environment and the real cron, so the page can stop guessing.
    # Every "(paper)" label and every NEXT UP time below used to be a
    # hardcoded string; on a live account they said "paper" and named a
    # rebalance time (Fri 21:05) the runner had not used since the cron
    # moved to 19:00. A dashboard that describes a different system than
    # the one running is worse than no dashboard.
    try:
        from trading.core.config import get_settings

        _s = get_settings()
        out["env"] = getattr(_s, "trading_env", "") or ""
        out["cycle_cron"] = os.getenv("CRON", "") or getattr(_s, "schedule_cron", "") or ""
        # The cron's own timezone. Since 2026-09-23 the live cycle is
        # expressed in New York time; parsing it as UTC showed it 4-5h early.
        out["cycle_tz"] = os.getenv("SCHEDULE_TZ", "") or "UTC"
        out["pm_pre_cycle_lead_minutes"] = _s.pm_pre_cycle_lead_minutes
    except Exception:
        out["env"] = ""
        out["cycle_cron"] = ""
        out["cycle_tz"] = "UTC"
        out["pm_pre_cycle_lead_minutes"] = 45

    _add_cockpit_blocks(out, state_dir, data_dir)

    # Memory vitals.
    try:
        mem = MemoryStore(state_dir / "memory")
        out["memory"] = {
            "stats": mem.stats(),
            "calibration": mem.calibration(),
            "trust": mem.trust_table(min_graded=1)[:10],
            "curator": mem.curator_summary(),
            "lessons": [
                {"id": r["id"], "status": r["status"], "statement": r["statement"]}
                for r in mem.lessons()[:8]
            ],
            "journal_tail": [
                {"ts": e["ts"].isoformat(), "kind": e["kind"], "actor": e["actor"]}
                for e in mem.journal_tail(12)
            ],
        }
    except Exception as e:
        logger.bind(component="dashboard").warning(f"memory failed: {e}")
        out["memory"] = {}

    return out


#: The page lives in ``static/index.html`` since the 2026-09-23 redesign: a
#: thousand lines of HTML/JS inside a Python string could not be linted,
#: previewed or diffed sensibly. Read once at import; the Docker image ships
#: everything under src/ (hatchling ``packages = ["src/trading"]``).
_PAGE = (Path(__file__).with_name("static") / "index.html").read_text(encoding="utf-8")


class _Handler(BaseHTTPRequestHandler):
    state_dir: Path
    data_dir: Path
    auth_token: str  # base64 of user:pass

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header == f"Basic {self.auth_token}"

    def do_GET(self) -> None:
        if not self._authorized():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="trading"')
            self.end_headers()
            return
        if self.path.startswith("/api/summary"):
            try:
                body = json.dumps(
                    build_summary(self.state_dir, self.data_dir), default=str
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
        elif self.path == "/" or self.path.startswith("/index"):
            body = _PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.bind(component="dashboard").debug(fmt % args)


def serve(host: str = "0.0.0.0", port: int = 8787) -> None:
    """Run the dashboard until interrupted. Refuses to start without
    DASHBOARD_PASS — an unauthenticated dashboard must be impossible."""
    from trading.core.config import settings

    user = os.getenv("DASHBOARD_USER", "yan")
    password = os.getenv("DASHBOARD_PASS", "")
    if not password:
        raise SystemExit("set DASHBOARD_PASS in .env — refusing to serve without auth")

    _Handler.state_dir = Path(settings.state_dir)
    _Handler.data_dir = Path(settings.data_dir)
    _Handler.auth_token = base64.b64encode(f"{user}:{password}".encode()).decode()

    server = ThreadingHTTPServer((host, port), _Handler)
    logger.bind(component="dashboard").info(f"dashboard listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
