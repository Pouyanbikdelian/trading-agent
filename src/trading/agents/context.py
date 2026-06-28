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


def _load_fundamentals(data_dir: Path) -> dict[str, Any]:
    """{symbol: Fundamentals} from the cache, or {} on any miss/error. Gives
    each position a sector tag so the committee (quant's correlation rule) and
    the deterministic guards can see when the book is concentrated. Never
    raises — sectors are a nice-to-have, never a reason to drop the context."""
    try:
        from trading.data.fundamentals_source import read_fundamentals_cache

        path = data_dir / "fundamentals.parquet"
        return read_fundamentals_cache(path) if path.exists() else {}
    except Exception:
        return {}


def _book_concentration(
    closes: dict[str, Any], *, window: int = 90, min_names: int = 3
) -> dict[str, Any] | None:
    """One interpretable concentration number for the held book: the 'effective
    number of bets' (ENB) from the correlation eigenvalues, plus average pairwise
    correlation. ENB ≈ N when names are independent and ≈ 1 when they all move
    together — so 6 holdings that act like 1.5 bets is the correlation a sector
    tag can't see (cross-sector co-movement). Reuses the closes build_context
    already read — no extra I/O. Returns None when there isn't enough clean,
    overlapping history; a missing number beats a noisy one."""
    if len(closes) < min_names:
        return None
    try:
        import numpy as np
        import pandas as pd

        recent = pd.DataFrame(closes).sort_index().pct_change().iloc[-window:]
        recent = recent.dropna(axis=1, how="any")  # keep names with full recent history
        if recent.shape[1] < min_names or recent.shape[0] < 20:
            return None
        corr = recent.corr().to_numpy()
        n = corr.shape[0]
        eig = np.linalg.eigvalsh(corr)
        eig = eig[eig > 1e-9]
        enb = float(eig.sum() ** 2 / np.square(eig).sum()) if eig.size else float(n)
        off = corr[~np.eye(n, dtype=bool)]
        avg_corr = float(off.mean()) if off.size else 0.0
        return {"n": n, "effective_bets": round(enb, 1), "avg_corr": round(avg_corr, 2)}
    except Exception:
        return None


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
            funds = _load_fundamentals(data_dir)
            close_series: dict[str, Any] = {}
            for pos in snap.positions.values():
                sym = pos.instrument.symbol
                row: dict[str, Any] = {
                    "symbol": sym,
                    "qty": float(pos.quantity),
                    "avg_cost": float(pos.avg_price),
                }
                s = _read_close(data_dir, sym)
                if s is not None and len(s) > 60:
                    close_series[sym] = s
                    last = float(s.iloc[-1])
                    yr = s.iloc[-252:]
                    lo, hi = float(yr.min()), float(yr.max())
                    row["last"] = last
                    row["unrealized_pct"] = last / float(pos.avg_price) - 1.0
                    if hi > lo:
                        row["entry_pctile_52w"] = round((float(pos.avg_price) - lo) / (hi - lo), 2)
                        row["now_pctile_52w"] = round((last - lo) / (hi - lo), 2)
                f = funds.get(sym)
                if f is not None and getattr(f, "sector", None):
                    row["sector"] = str(f.sector)
                positions.append(row)
            ctx["account"] = {
                "equity": snap.equity,
                "cash": snap.cash,
                "base_currency": snap.base_currency,
            }
            bc = _book_concentration(close_series)
            if bc:
                ctx["book_concentration"] = bc
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
            {
                "id": r["id"],
                "lesson": r["statement"],  # full elaborated text: title + 4-sentence body
                "support_vs_contradict": f"{r['support']}/{r['contradict']}",
            }
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
