"""The copilot's "NOW" side — current portfolio/order/risk facts.

Everything reads the source SQLite files in ``mode=ro`` (same contract
as the dashboard: this module can never create, migrate, or mutate a
store the runner owns) and NOTHING imports the broker. Each fact
carries its own timestamp so the engine can cite data ages honestly.
"""

from __future__ import annotations

import json
import os as _os
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


def lessons_now(state_dir: Path, limit: int = 24) -> dict[str, Any]:
    """The lesson book: what the desk believes it has learned.

    The copilot could read positions, orders, the PM book, risk and the
    tape, but nothing about lessons — so when Yan asked it to write one on
    2026-08-05 it answered that it was read-only and could not execute
    trades, which was true and beside the point. It could not discuss
    lessons because it had never been handed any.

    Established, candidate and challenged lessons are returned separately and labelled:
    conflating "the desk has evidence for this" with "someone suggested
    this last week" is the whole distinction the lifecycle exists to make.
    """
    out: dict[str, Any] = {
        "established": [],
        "candidates": [],
        "challenged": [],
        "note": (
            "established = supported by >=3 net measured outcomes, or set by the "
            "operator in a hard tone; candidate = proposed, not yet earned; "
            "challenged = withheld after contradictory measured outcomes"
        ),
    }
    try:
        from trading.memory.store import MemoryStore

        mem = MemoryStore(Path(state_dir) / "memory")
    except Exception:
        return out
    try:
        for status, key in (
            ("established", "established"),
            ("candidate", "candidates"),
            ("challenged", "challenged"),
        ):
            for row in mem.lessons(status=status)[:limit]:
                out[key].append(
                    {
                        "id": row["id"],
                        "statement": row["statement"],
                        "support_vs_contradict": f"{row['support']}/{row['contradict']}",
                        "tags": row["tags"],
                        "author": "operator" if "operator" in (row["tags"] or "") else "historian",
                    }
                )
    except Exception:
        pass
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


def config_now(state_dir: Path) -> dict[str, Any]:
    """Every setting that governs what the system will actually do.

    The copilot could cite positions, orders, lessons and halt state — and
    had NO source at all for configuration. Asked "what's our guard %" or
    "how big is the PM sleeve", it answered from the model's general
    knowledge, which is a polite way of saying it made something up. That
    is the complaint that produced this function (operator, 2026-08-10).

    Everything here is READ FROM THE LIVE OBJECTS — Settings, the guards'
    own env helpers, the mode file — never from a hand-maintained copy.
    That is deliberate and it is the lesson of the 2026-08-07 audit: this
    project's most dangerous defects were all documentation describing
    wiring that did not exist. A config summary maintained by hand would
    become exactly that, and the copilot would state it with confidence.
    """
    # get_settings() rather than the module-level `settings` singleton.
    # The singleton is bound at first import, so anything importing early
    # would report a stale snapshot — the exact defect fixed in pm_signal
    # on 2026-08-07 ("I changed the sleeve and nothing happened"). A
    # config reporter that reports the config as it was at import time
    # would be worse than useless: confidently wrong.
    from trading.core.config import get_settings
    from trading.runtime.guards import PRICE_SANITY_PCT, _env_f
    from trading.runtime.guards import enabled as guards_enabled

    s = get_settings()

    out: dict[str, Any] = {}

    out["environment"] = {
        "trading_env": s.trading_env,
        "live_armed": s.is_live_armed(),
        "_armed_means": (
            "both TRADING_ENV=live AND ALLOW_LIVE_TRADING=true. Note that "
            "docker-compose `environment:` overrides .env — trust "
            "`trading status`, not the file."
        ),
        "state_dir": str(s.state_dir),
        "broker": f"IBKR {s.ibkr_host}:{s.ibkr_port}",
        "_port_meaning": "4001 = LIVE gateway, 4002 = paper",
    }

    out["who_trades"] = {
        "agent_pm_sleeve_pct": s.agent_pm_sleeve_pct,
        "strategy_sleeve_pct": s.strategy_sleeve_pct,
        "_meaning": (
            "INDEPENDENT fractions of total account equity, one per book. "
            "strategy_sleeve_pct=0.0 means the mechanical momentum strategy "
            "still computes signals (for the benchmark and shadow ledger) "
            "but every weight is multiplied by zero, so it CANNOT place an "
            "order. At that setting the Agent PM is the only book trading."
        ),
        "pm_sleeve_capital_usd": s.pm_sleeve_capital_usd,
        "_capital_cap_meaning": (
            "Hard DOLLAR ceiling on the PM sleeve, on top of the fraction. "
            "A fraction alone grows with the account; this does not. "
            "Converted to base currency at the cycle's FX rate. 0 disables."
        ),
        "agent_pm_signal_max_age_h": s.agent_pm_signal_max_age_h,
        "_freshness_meaning": (
            "A PM decision older than this is REFUSED, not traded. So "
            "`/pm run` must precede `/cycle` within this window."
        ),
    }

    out["risk_limits"] = {
        "max_gross_exposure": s.max_gross_exposure,
        "max_position_pct": s.max_position_pct,
        "max_daily_loss_pct": s.max_daily_loss_pct,
        "max_drawdown_pct": s.max_drawdown_pct,
        "max_margin_borrowing_pct": s.max_margin_borrowing_pct,
        "_all_are": "fractions of TOTAL account equity, not of the sleeve",
        "_kill_switches_HALT_they_do_not_sell": (
            "max_daily_loss_pct and max_drawdown_pct stop trading. They "
            "never liquidate. Nothing in this system reduces exposure "
            "automatically except the position guards."
        ),
        "_config_risk_yaml_is_NOT_read": (
            "config/risk.yaml is reference only — no loader exists. .env is "
            "the single source for these numbers."
        ),
    }

    ratchet_floor = _env_f("GUARD_TRAIL_FLOOR", None)
    ratchet_tighten = _env_f("GUARD_TRAIL_TIGHTEN", None)
    ratchet_on = (
        ratchet_floor is not None
        and ratchet_tighten is not None
        and 0.0 < ratchet_floor < 1.0
        and ratchet_tighten > 0.0
    )
    out["position_guards"] = {
        "enabled": guards_enabled(),
        "_what_they_are": (
            "ATR-style TRAILING STOPS. The only mechanism in the system "
            "that exits a position on its own. They apply to EVERY position "
            "in the account regardless of who opened it — including "
            "personal holdings. `/hold SYMBOL` is the only exemption."
        ),
        "atr_mult": _env_f("GUARD_ATR_MULT", 3.0),
        "trail_min_pct": _env_f("GUARD_TRAIL_MIN_PCT", 8.0),
        "trail_max_pct": _env_f("GUARD_TRAIL_MAX_PCT", 20.0),
        "_stop_formula": (
            "stop_distance = max(trail_min, min(trail_max, atr_mult x the "
            "symbol's mean absolute daily move)). Measured DOWN FROM THE "
            "HIGH-WATER MARK since entry, not from today's open, and the "
            "stop only ever rises. Most S&P names land on the floor."
        ),
        "take_profit_pct": _env_f("GUARD_TP_PCT", None),
        "profit_ratchet_on": ratchet_on,
        "ratchet_floor": ratchet_floor,
        "ratchet_tighten": ratchet_tighten,
        "_ratchet_meaning": (
            "When on, the stop TIGHTENS as a position gains: "
            "effective = base x max(floor, 1 - tighten x gain). At "
            "floor=0.4/tighten=1.2 a +50% winner sits on 0.4x its base "
            "stop. When off, the trail stays at its full width."
        ),
        "price_sanity_pct": _env_f("GUARD_PRICE_SANITY_PCT", PRICE_SANITY_PCT),
        "_price_sanity_meaning": (
            "Guards refuse to exit when the yfinance quote disagrees with "
            "the broker's own mark by more than this — a split or bad tick "
            "otherwise reads as a crash."
        ),
        "cooldown_hours_per_symbol": 24.0,
        "_after_a_guard_exit": (
            "NOTHING is re-deployed automatically. The cash sits until the "
            "next scheduled cycle. There is no post-stop-out policy."
        ),
    }

    out["execution"] = {
        "order_type": "MARKET",
        "time_in_force": "DAY",
        "_no_algos": (
            "Every order the risk manager builds is a plain market order. "
            "No VWAP, TWAP, limit, mid-price or IBKR Adaptive algo. Only "
            "manual /buy and /sell accept an optional limit price."
        ),
        "long_only": True,
        "_long_only_meaning": (
            "Negative target weights clamp to 0 and sells clamp to the "
            "position net of working orders. The system cannot go short."
        ),
    }

    out["schedule"] = {
        "cycle_cron_utc": _os.getenv("CRON", "(compose default)"),
        "agent_pm_runs": "45 minutes before the cycle, derived from the same cron",
        "broker_readiness_check": "1 hour before the cycle",
        "_all_jobs_live_in_the_runner": (
            "Every scheduled job — snapshots, macro, rotation, historian, "
            "lessons, guards, PM, committee — runs inside the trader "
            "container's scheduler. If no runner is up, NOTHING updates "
            "and /cycle and /pm run write a flag nobody reads."
        ),
    }

    out["approval_gate"] = {
        "require_cycle_approval": s.require_cycle_approval,
        "timeout_seconds": s.cycle_approval_timeout_s,
        "_meaning": (
            "When true the cycle computes the basket, shows it, and waits "
            "for /approve, /approve N, /approve only SYM, /approve all "
            "except SYM, /approve flat or /reject. The two filtered forms "
            "are rebuilt through risk and need a separate exact-plan "
            "confirmation. No reply within the timeout auto-REJECTS — it "
            "never auto-submits. /pick remains a distinct equal-weight "
            "replacement basket."
        ),
    }

    try:
        from trading.runtime.mode import read_mode

        st = read_mode(Path(state_dir) / "mode.json")
        out["operator_mode"] = {
            "mode": st.mode.value,
            "set_by": st.set_by,
            "set_at": st.set_at,
            "_meaning": (
                "bull/neutral pass through. defense/bear scale the whole "
                "book down. flatten zeroes all targets. Applies to the PM "
                "sleeve as well as the mechanical strategy."
            ),
        }
    except Exception:
        out["operator_mode"] = {"mode": "unknown"}

    try:
        from trading.runner.holds import load_holds

        held = sorted(load_holds(Path(state_dir)))
        out["holds"] = {
            "symbols": held,
            "_meaning": (
                "Pinned symbols. The cycle will neither sell NOR buy them — "
                "frozen in both directions — and the guards skip them. "
                "Manual /buy /sell /close /flatten still work. A hold on a "
                "symbol you do not own blocks the system ever opening it."
            ),
        }
    except Exception:
        out["holds"] = {"symbols": []}

    return out


def operating_manual() -> dict[str, Any]:
    """How to operate the system — the sequences, in order.

    Mirrors ``Cycle._run_inner``. Kept next to ``config_now`` so the
    copilot can answer "how do I refresh the PM" with the actual order of
    operations rather than a plausible guess.
    """
    return {
        "what_a_cycle_does_in_order": [
            "1. Load the universe (~503 S&P symbols) and refresh prices from yfinance (~2 min)",
            "1c. Add the PM's own picks so its off-universe ETFs are priceable",
            "3. Fetch the broker account snapshot",
            "4. Intraday kill switches (daily loss, drawdown) — halts, never sells",
            "5. Mechanical strategy weights, scaled by STRATEGY_SLEEVE_PCT",
            "6b. Operator mode overlay on the strategy weights",
            "7c. Merge the Agent PM's targets, scaled by sleeve %, dollar cap and mode",
            "8. Risk manager: target weights -> per-name delta orders, all caps applied",
            "8d. Approval gate if REQUIRE_CYCLE_APPROVAL — shows the basket and waits",
            "9. Submit under the execution lock, then reconcile fills",
        ],
        "refresh_the_pm_and_trade": [
            "/pm run   — convenes the PM; it reassesses and writes a NEW decision (~1 min)",
            "/pm       — read what it decided",
            "/cycle    — turns that decision into orders (~4 min; refuses a decision older than AGENT_PM_SIGNAL_MAX_AGE_H)",
        ],
        "_pm_run_first": (
            "/cycle ALONE re-trades the EXISTING decision. It does not ask "
            "the PM to think again. /pm run is what makes it reassess."
        ),
        "_committee_is_separate": (
            "/committee is the multi-agent debate that feeds context. It is "
            "not required before /pm run and does not itself decide trades."
        ),
        "stopping_and_starting": {
            "stand_down": "docker compose stop trader-live — a stopped container is unambiguous",
            "halt": "/halt stops NEW trading. It does NOT close positions and does NOT flatten.",
            "exit_everything": "/flatten — the only command that liquidates",
            "resume": "/resume clears the halt",
        },
        "arming_order_matters": [
            "Gateway to LIVE mode FIRST",
            "THEN point STATE_DIR at the live directory",
            "The reverse order wrote paper equity into the live kill-switch baseline on 2026-08-07 and halted the account at -91.82%. State dirs are now stamped and a mismatched runner refuses to start.",
        ],
    }


def operator_interface(state_dir: Path) -> dict[str, Any]:
    """How to TALK to the desk — which door a typed message goes through.

    ``config_now`` taught the copilot the STATE of the desk. This teaches
    it the RULES OF THE INTERFACE, which is a different kind of ignorance
    and a worse one: asked "how do I make you remember I'm long-term
    bullish on TSLA", the copilot had no evidence that mandates expire,
    that tone sets strength, or that /lesson exists — so it could only
    decline or guess.

    Every number and word here is read from ``copilot.mandates`` at call
    time rather than restated. A hand-written copy of the strength
    vocabulary would drift the first time a pattern was added, and the
    copilot would confidently coach the operator into phrasing that no
    longer grades the way it says.
    """
    out: dict[str, Any] = {
        "_read_this_first": (
            "A typed message goes through exactly ONE of four doors, decided "
            "by its shape, not by intent. Getting the door wrong is the most "
            "common reason an instruction has no effect."
        ),
        "four_doors": [
            "1. Starts with '/' -> a COMMAND. Deterministic code. Commands can change their own defined state immediately; a desk-change proposal is the only non-command path that persists, and it still cannot apply until approved.",
            "2. A recognised free-text request to show or change the operator watchlist or an operator lesson -> a DESK-CHANGE PROPOSAL. It changes nothing until the operator approves that exact proposal; it never reaches trading execution.",
            "3. Other free text that parses as an instruction -> stored as a MANDATE for the next run. It never reaches the copilot, so there is no conversational reply, just a confirmation.",
            "4. Anything else -> the COPILOT (me). Read-only. I answer and the statement persists NOWHERE.",
        ],
        "_opinions_do_not_persist": (
            "This is the trap. 'I'm bullish on TSLA' is door 4: it has no "
            "instruction verb and no forward marker, so it is answered and "
            "forgotten. Only a command or a recognised instruction survives "
            "the conversation. Earlier chat turns are never evidence."
        ),
        "_the_copilot_cannot_act": (
            "I cannot place an order, set a hold, or change a trading setting "
            "— a test fails the build if this package so much as imports an "
            "execution path. The Telegram router can stage a recognised "
            "watchlist or operator-lesson proposal, but it cannot apply one "
            "without the operator's exact approval."
        ),
    }

    try:
        from trading.copilot import mandates as _m

        strengths: dict[str, list[str]] = {}
        for strength, pattern in _m._STRENGTH_PATTERNS:
            phrase = pattern.replace(r"\b", "").replace("(?:", "(").replace("'?", "'")
            strengths.setdefault(strength, []).append(phrase)
        out["how_a_mandate_is_graded"] = {
            "_by_tone_not_by_flag": (
                "Strength is read from the operator's phrasing. Unrecognised "
                "phrasing defaults to SOFT on purpose: he can always restate "
                "more firmly, but cannot un-buy a position taken on a misread."
            ),
            "trigger_phrases_by_strength": strengths,
            "what_each_strength_does": {
                "strong": "The desk acts on it unless there is a concrete reason not to.",
                "medium": "The desk weighs it seriously and reports back if it declines.",
                "soft": "The desk considers it and may drop it.",
            },
            "_fixing_a_misread": "/harden <id> or /soften <id>; /mandates to review; /mandates drop <id>",
            "_note_consider_is_soft": (
                "'consider X next round' grades SOFT — it may be dropped "
                "silently. To actually move the book, phrase it strongly: "
                "'I want X in the book next run'."
            ),
        }
        out["mandates_expire"] = {
            "default_ttl_days": _m.DEFAULT_TTL_DAYS,
            "_meaning": (
                f"A mandate is deleted after {_m.DEFAULT_TTL_DAYS} days. It is a "
                "nudge for the next run or two, NOT a standing belief. A "
                "request to remember something 'always', 'permanently' or "
                "'long term' CANNOT be expressed as a mandate — say so and "
                "point at /lesson instead."
            ),
        }
        out["_mandates_cannot_lift_a_limit"] = (
            "Mandates are advisory. They reach the PM as context, but every "
            "position, cluster and gross cap still binds afterwards."
        )
    except Exception:
        out["how_a_mandate_is_graded"] = {"unavailable": True}

    out["which_tool_for_which_intent"] = {
        "ask a question": "just type it — read-only, nothing persists",
        "nudge the next run": "plain text with a forward marker, e.g. 'consider ASML next round' (soft)",
        "make the next run act": "'I want ASML in the book next run' (strong)",
        "a DURABLE belief": (
            "/lesson <statement> — stages a tone-graded proposal: a firm "
            "instruction will land as ESTABLISHED and reach every agent from "
            "the next cycle; softer phrasing will land as a CANDIDATE. The "
            "operator must approve the exact proposal first."
        ),
        "track a research name": (
            "say 'add NVDA to our watchlist' or use /watchlist add NVDA. "
            "The bot stages an approval-bound dashboard/research change only; "
            "it does not add a tradable universe member or place an order. "
            "/watchlist undo stages an approval-bound reversal of the last change."
        ),
        "stop a position being sold": (
            "/hold <SYMBOL> — a hard mechanical block, checked before every "
            "sell including the position guards. /unhold to release. This is "
            "the mechanical half of 'keep it long term'; /lesson is the "
            "reasoning half, and they are usually wanted together."
        ),
        "review what is standing": (
            "/watchlist for research names, /mandates for instructions, "
            "/lessons for beliefs, /holds for protected names"
        ),
    }

    try:
        from trading.copilot.mandates import for_context as _for_context

        out["mandates_currently_standing"] = _for_context(Path(state_dir))
    except Exception:
        out["mandates_currently_standing"] = []
    return out
