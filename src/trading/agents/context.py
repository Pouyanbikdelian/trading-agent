"""Context builder — everything the committee sees, gathered locally.

One network-free pass over the system's own state: latest account
snapshot, positions with 52-week entry percentiles, operator holds,
monitor state files (macro / options / SPY-VIX / style), established
lessons, dossier list and the source-trust table. The committee judges
TODAY with the memory of every yesterday attached.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def build_context(state_dir: Path, data_dir: Path) -> dict[str, Any]:
    from trading.memory.store import MemoryStore
    from trading.runner.holds import load_holds, load_k_override
    from trading.runner.state import RunnerStore
    from trading.runtime.portfolio_stats import _read_close

    ctx: dict[str, Any] = {}

    # --- book
    try:
        snap = RunnerStore(state_dir / "runner.db").latest_snapshot()
        positions = []
        if snap:
            for pos in snap.positions.values():
                sym = pos.instrument.symbol
                row: dict[str, Any] = {
                    "symbol": sym,
                    "qty": float(pos.quantity),
                    "avg_cost": float(pos.avg_price),
                }
                s = _read_close(data_dir, sym)
                if s is not None and len(s) > 60:
                    last = float(s.iloc[-1])
                    yr = s.iloc[-252:]
                    lo, hi = float(yr.min()), float(yr.max())
                    row["last"] = last
                    row["unrealized_pct"] = last / float(pos.avg_price) - 1.0
                    if hi > lo:
                        row["entry_pctile_52w"] = round((float(pos.avg_price) - lo) / (hi - lo), 2)
                        row["now_pctile_52w"] = round((last - lo) / (hi - lo), 2)
                positions.append(row)
            ctx["account"] = {
                "equity": snap.equity,
                "cash": snap.cash,
                "base_currency": snap.base_currency,
            }
        ctx["positions"] = positions
    except Exception as e:
        logger.bind(component="agents").warning(f"context: book unavailable ({e})")

    # --- operator state
    ctx["holds"] = sorted(load_holds(state_dir))
    ctx["k_override"] = load_k_override(state_dir)

    # --- monitors (already-computed state files; no recomputation)
    ctx["macro_dial"] = _read_json(state_dir / "macro_monitor.json").get("readings", {})
    ctx["vol_surface"] = _read_json(state_dir / "options_monitor.json").get("metrics", {})
    ctx["spy_vix_triggers"] = _read_json(state_dir / "advisor.json").get("active", [])
    style = _read_json(state_dir / "style_advisor.json")
    ctx["style_leader"] = style.get("leader")

    # --- permanent memory
    try:
        mem = MemoryStore(state_dir / "memory")
        ctx["established_lessons"] = [
            {"id": r["id"], "statement": r["statement"]}
            for r in mem.lessons(status="established")[:6]
        ]
        ctx["dossiers"] = mem.dossiers()
        ctx["source_trust"] = mem.trust_table(min_graded=2)[:10]
        ctx["recent_memory"] = [
            {"kind": e["kind"], "actor": e["actor"]} for e in mem.journal_tail(10)
        ]
    except Exception as e:
        logger.bind(component="agents").warning(f"context: memory unavailable ({e})")

    # --- slow macro (FRED): CPI, claims, HY spreads etc. Compact latest
    # readings only — the dashboard owns the full history.
    try:
        from trading.runtime.econ_watch import latest_block

        econ = latest_block(state_dir)
        if econ:
            ctx["economy"] = econ
    except Exception as e:
        logger.bind(component="agents").warning(f"context: economy unavailable ({e})")

    # --- outside world, last: if the serialized context must be cut to fit
    # the prompt budget, gossip is the right thing to lose first.
    # Collected by news_watch on its own schedule; stale collections are
    # dropped so the scout never reasons over old chatter.
    try:
        from trading.runtime.news_watch import load as load_news

        news = load_news(state_dir)
        if news:
            ctx["sector_momentum_vs_spy_pct"] = news.get("sector_momentum", {})
            ctx["headlines"] = news.get("headlines", [])[:48]
    except Exception as e:
        logger.bind(component="agents").warning(f"context: news unavailable ({e})")

    return ctx
