"""Context builder — everything the committee sees, gathered locally.

One network-free pass over the system's own state: latest account
snapshot, positions with 52-week entry percentiles, operator holds,
monitor state files (macro / options / SPY-VIX / style), established
lessons, dossier list and the source-trust table. The committee judges
TODAY with the memory of every yesterday attached.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


MONITOR_MAX_AGE_H = 36.0
"""Hours before a monitor file stops counting as a current reading.

36h matches ``news_watch.load`` and cleanly separates the two real cases:
the runner is up and these are hours old, or the runner is down and they
are days old. Resolved per call so a change takes effect without a
rebuild — the same frozen-at-import trap that bit ``pm_signal``.
"""


def _monitor_max_age_h() -> float:
    raw = os.getenv("MONITOR_MAX_AGE_H")
    if raw is None:
        return MONITOR_MAX_AGE_H
    try:
        v = float(raw)
    except ValueError:
        logger.bind(component="agents").warning(f"MONITOR_MAX_AGE_H={raw!r} not a number; default")
        return MONITOR_MAX_AGE_H
    # 0 or negative would silently blind the desk entirely. Refuse it.
    return v if v > 0 else MONITOR_MAX_AGE_H


def _read_fresh_json(
    path: Path, label: str, *, gaps: list[str], max_age_h: float
) -> dict[str, Any]:
    """``_read_json``, but {} once the reading is older than ``max_age_h``.

    A missing file is a gap too: absent is not zero, and the desk should
    hear "no macro dial" rather than infer a calm one from an empty dict.
    """
    raw = _read_json(path)
    if not raw:
        gaps.append(f"{label}: no reading on disk")
        return {}
    stamp = raw.get("last_polled_at") or raw.get("asof") or raw.get("t")
    if not stamp:
        # Undated file: cannot be verified, so it cannot be trusted.
        gaps.append(f"{label}: reading carries no timestamp")
        return {}
    try:
        age_h = (datetime.now(tz=timezone.utc) - datetime.fromisoformat(stamp)).total_seconds() / 3600
    except Exception:
        gaps.append(f"{label}: unreadable timestamp {stamp!r}")
        return {}
    if age_h > max_age_h:
        gaps.append(f"{label}: {age_h:.0f}h old (max {max_age_h:.0f}h) — DROPPED, treat as unknown")
        return {}
    return raw


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
    # What the operator has argued recently, in his own words. A /hold is
    # an instruction the code enforces; an objection is a VIEW the
    # committee should weigh and may reasonably disagree with. Without
    # this the operator's reasoning dies in the chat scrollback while the
    # same debate repeats a week later.
    try:
        from trading.copilot.thread import recent_objections

        ctx["operator_objections"] = recent_objections(state_dir, limit=5)
    except Exception as e:
        logger.bind(component="agents").warning(f"context: objections unavailable ({e})")
    # Standing instructions left for this run ("high conviction on GS,
    # look at it next round"). Graded by the tone he used — see
    # copilot.mandates. Advisory: the risk caps still bind afterwards.
    try:
        from trading.copilot.mandates import for_context as _mandates_for_context

        ctx["operator_mandates"] = _mandates_for_context(state_dir)
    except Exception as e:
        logger.bind(component="agents").warning(f"context: mandates unavailable ({e})")

    # --- monitors (already-computed state files; no recomputation)
    #
    # Freshness-gated. These four are the desk's ONLY view of regime and
    # volatility, and every one is written by a scheduled job inside the
    # runner. Stop the runner and the files simply stop changing — they
    # do not disappear, so `_read_json` kept returning a VIX reading and
    # a macro dial from whenever the runner was last up, and the PM
    # reasoned over them as if they were current. The runner was stopped
    # 2026-08-07..08-11 and nothing anywhere said the risk picture was
    # four days old.
    #
    # Dropped rather than passed through with a label: a stale VIX
    # presented as today's VIX is worse than no VIX, because the desk
    # cannot tell it is blind. `_data_gaps` is how it is told.
    gaps: list[str] = []
    _mon = partial(_read_fresh_json, gaps=gaps, max_age_h=_monitor_max_age_h())
    ctx["macro_dial"] = _mon(state_dir / "macro_monitor.json", "macro_dial").get("readings", {})
    ctx["vol_surface"] = _mon(state_dir / "options_monitor.json", "vol_surface").get("metrics", {})
    ctx["spy_vix_triggers"] = _mon(state_dir / "advisor.json", "spy_vix_triggers").get("active", [])
    style = _mon(state_dir / "style_advisor.json", "style_leader")
    ctx["style_leader"] = style.get("leader")
    if gaps:
        ctx["_data_gaps"] = gaps
        logger.bind(component="agents").warning(f"context: stale monitors dropped: {gaps}")

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
        # Operator-authored lessons that have NOT been hardened. Only
        # established lessons used to reach the context, so a lesson Yan
        # phrased tentatively influenced precisely nothing — it sat in the
        # database waiting for graded episodes it would never accumulate,
        # because nothing trades on a lesson no agent can see. Surfaced
        # separately, and labelled unproven, so the desk weighs it as a
        # view from the operator rather than as settled desk knowledge.
        ctx["operator_lessons_under_consideration"] = [
            {"id": r["id"], "lesson": r["statement"]} for r in mem.operator_lessons("candidate")[:6]
        ]
        ctx["dossiers"] = mem.dossiers()
        ctx["source_trust"] = mem.trust_table(min_graded=2)[:10]
        ctx["recent_memory"] = [
            {"kind": e["kind"], "actor": e["actor"]} for e in mem.journal_tail(10)
        ]
    except Exception as e:
        logger.bind(component="agents").warning(f"context: memory unavailable ({e})")

    # --- the ranked candidate ladder: the ONLY channel through which a
    # name the desk doesn't already own can reach an agent. Without it the
    # context named nothing but current holdings, and every allocator
    # downstream re-picked the book it was shown (see agents/candidates.py).
    try:
        from trading.agents.candidates import build_candidate_ladder

        ladder = build_candidate_ladder(data_dir)
        if ladder:
            ctx["candidate_ladder"] = ladder
    except Exception as e:
        logger.bind(component="agents").warning(f"context: candidate ladder unavailable ({e})")

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
