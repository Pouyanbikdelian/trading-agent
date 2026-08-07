"""The bridge: turn the Agent PM's decision into a tradeable ``Signal``.

The PM has run a simulated USD book for months and has never been able to
touch the broker. Not because its output is the wrong shape — it already
emits ``target_weights`` (``pm.py:719``), which is exactly what
``core.types.Signal`` carries — but because nobody wrote the twenty lines
that connect them. ``_clamp_weights`` even says so: caps apply "in sim
today and through the bridge later".

This is that bridge. It reads the PM's own durable decision record and
returns a ``Signal`` the existing pipeline can consume, so the PM reaches
the market the same way every strategy does:

    Signal -> combiner -> RISK MANAGER -> Order

That ordering is the point. The PM is an LLM; it can hallucinate a ticker,
fat-finger a weight, or be talked into something by its own reasoning. It
gets no special path. The hard-blocking risk manager still sizes every
order, applies exposure and per-name caps, and can refuse the lot.

Three deliberate choices
------------------------

**Reads ``last_run.json``, not ``portfolio.json``.** The former is the
DECISION ("here is what I want to hold"); the latter is the simulated
book's bookkeeping ("here is what I pretended to buy, at what marks").
Executing the decision is right; executing the sim book would import its
accounting fictions — stale marks, simulated cash, assumed fills.

**Freshness is enforced, not assumed.** The PM currently runs Monday
14:30 UTC and the cycle Friday 21:05 — four days apart. A decision made
against Monday's tape, executed on Friday's, is not the decision the PM
made. Rather than silently accept that, this refuses anything older than
``AGENT_PM_SIGNAL_MAX_AGE_H`` and says why. The intended fix is to move
the PM run to shortly before the cycle.

**Dropped names are reported, never swallowed.** The PM may hold up to 3
ETFs, but the cycle builds its instrument map from the configured equity
universe (``cycle.py:388``), so an ETF target would vanish with no log
line and the weight would quietly become cash. Every dropped symbol comes
back in ``dropped`` for the caller to surface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.agents.pm import UNIVERSE as PM_ETF_SHELF
from trading.core.logging import logger
from trading.core.types import AssetClass, Instrument, Signal

#: Producer name, for attribution in the journal and the order ledger.
STRATEGY_NAME = "agent_pm"

def _setting(name: str, env: str, fallback: float) -> float:
    """Read a tunable from Settings, falling back to the environment.

    Resolved at CALL time rather than import time. A module-level
    ``os.getenv`` freezes the value when the module is first imported,
    which silently ignores a changed ``.env`` in any process that imported
    early — and "I changed the sleeve and nothing happened" is exactly the
    class of confusion this codebase keeps paying for.
    """
    try:
        from trading.core.config import settings

        return float(getattr(settings, name))
    except Exception:
        return float(os.getenv(env, str(fallback)))


def default_max_age_h() -> float:
    """How stale a PM decision may be and still be executed. Six hours
    assumes the PM runs shortly before the cycle; it is far below the
    four-day gap the old Monday schedule produced, so a mis-scheduled PM
    fails loudly instead of trading last week's view."""
    return _setting("agent_pm_signal_max_age_h", "AGENT_PM_SIGNAL_MAX_AGE_H", 6.0)


def default_sleeve_pct() -> float:
    """Fraction of TOTAL ACCOUNT EQUITY the PM may direct. Its own weights
    sum to at most MAX_GROSS (1.0) of its simulated book; scaling by this
    maps them onto a real sleeve. 0.0 disables the bridge."""
    return _setting("agent_pm_sleeve_pct", "AGENT_PM_SLEEVE_PCT", 0.0)


@dataclass(frozen=True)
class PMSignalResult:
    """Outcome of one bridge attempt. Never an exception — this runs on a
    scheduler, and a raising signal source is a signal source that stops
    running."""

    signal: Signal | None
    reason: str
    decided_at: datetime | None = None
    age_hours: float | None = None
    dropped: list[str] = field(default_factory=list)
    gross: float = 0.0
    #: The sleeve fraction this result was built with. The caller needs it
    #: to scale the strategy side down by the same amount, so the two
    #: books sum to one account rather than to 1 + sleeve.
    sleeve_pct: float = 0.0

    @property
    def ok(self) -> bool:
        return self.signal is not None


def pm_decision_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / "agent_pm" / "last_run.json"


def _instrument_for(symbol: str) -> Instrument:
    """ETF shelf names are ``etf:XLK``; everything else ``equity:AAPL``.

    The distinction is not cosmetic — it is the parquet cache partition
    and the key the cycle matches on, so getting it wrong makes the name
    unpriceable rather than merely mislabelled.
    """
    cls = AssetClass.ETF if symbol in PM_ETF_SHELF else AssetClass.EQUITY
    return Instrument(symbol=symbol, asset_class=cls)


def load_pm_signal(
    state_dir: Path | str,
    *,
    now: datetime | None = None,
    max_age_h: float | None = None,
    sleeve_pct: float | None = None,
    tradeable_keys: set[str] | None = None,
) -> PMSignalResult:
    """Read the PM's latest decision and express it as a ``Signal``.

    ``sleeve_pct`` scales PM weights onto a slice of real equity, so PM
    and momentum stay separately attributable and the PM's exposure is
    bounded before the risk manager ever sees it. ``tradeable_keys``, when
    given, is the set of ``Instrument.key`` the cycle can actually price;
    anything outside it is dropped and reported rather than silently lost.
    """
    now = now or datetime.now(tz=timezone.utc)
    max_age_h = default_max_age_h() if max_age_h is None else max_age_h
    sleeve_pct = default_sleeve_pct() if sleeve_pct is None else sleeve_pct

    if sleeve_pct <= 0:
        return PMSignalResult(None, "PM bridge disabled (AGENT_PM_SLEEVE_PCT=0)")

    path = pm_decision_path(state_dir)
    try:
        raw: dict[str, Any] = json.loads(path.read_text())
    except FileNotFoundError:
        return PMSignalResult(None, f"no PM decision yet at {path}")
    except Exception as e:
        logger.bind(component="pm_signal").warning(f"unreadable PM decision: {e!r}")
        return PMSignalResult(None, f"unreadable PM decision: {type(e).__name__}")

    if not raw.get("ok"):
        return PMSignalResult(None, f"last PM run failed: {raw.get('reason', 'unknown')}")

    try:
        decided_at = datetime.fromisoformat(str(raw["ts"]))
    except Exception:
        return PMSignalResult(None, f"PM decision has an unparseable ts: {raw.get('ts')!r}")
    if decided_at.tzinfo is None:
        decided_at = decided_at.replace(tzinfo=timezone.utc)

    age_h = (now - decided_at).total_seconds() / 3600.0
    if age_h > max_age_h:
        # Loud, because the likely cause is a scheduling gap rather than a
        # one-off — and a bridge that quietly does nothing every week is
        # indistinguishable from a PM that keeps choosing to hold.
        logger.bind(component="pm_signal").warning(
            f"PM decision is {age_h:.1f}h old (limit {max_age_h:.0f}h) — not trading it. "
            "Schedule the PM run shortly before the cycle."
        )
        return PMSignalResult(
            None,
            f"PM decision {age_h:.1f}h old, limit {max_age_h:.0f}h",
            decided_at=decided_at,
            age_hours=age_h,
        )

    weights = raw.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        return PMSignalResult(
            None, "PM decision carries no weights", decided_at=decided_at, age_hours=age_h
        )

    keyed: dict[str, float] = {}
    dropped: list[str] = []
    for sym, w in weights.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            dropped.append(str(sym))
            continue
        if w <= 0:
            continue
        ins = _instrument_for(str(sym).upper().strip())
        if tradeable_keys is not None and ins.key not in tradeable_keys:
            dropped.append(ins.symbol)
            continue
        keyed[ins.key] = w * sleeve_pct

    if dropped:
        logger.bind(component="pm_signal").warning(
            f"PM targets not tradeable in this universe, weight goes to cash: {sorted(dropped)}"
        )
    if not keyed:
        return PMSignalResult(
            None,
            "no PM target survived instrument mapping",
            decided_at=decided_at,
            age_hours=age_h,
            dropped=sorted(dropped),
        )

    gross = sum(keyed.values())
    signal = Signal(
        ts=decided_at,
        strategy=STRATEGY_NAME,
        target_weights=keyed,
        metadata={
            "decided_at": decided_at.isoformat(timespec="seconds"),
            "age_hours": f"{age_h:.2f}",
            "sleeve_pct": f"{sleeve_pct:.4f}",
            "gross_of_account": f"{gross:.4f}",
            "n_targets": str(len(keyed)),
            "dropped": ",".join(sorted(dropped)),
            # Carried so an order can be read back against the reasoning
            # that produced it, months later.
            "pm_equity": str(raw.get("equity", "")),
        },
    )
    logger.bind(component="pm_signal").info(
        f"PM signal: {len(keyed)} name(s), gross {gross:.2%} of account, "
        f"decision {age_h:.1f}h old"
    )
    return PMSignalResult(
        signal,
        "ok",
        decided_at=decided_at,
        age_hours=age_h,
        dropped=sorted(dropped),
        gross=gross,
        sleeve_pct=sleeve_pct,
    )


def format_bridge_note(result: PMSignalResult) -> str:
    """One line for Telegram. Silence on the happy path is wrong here —
    the operator needs to know whether the PM actually reached the market
    this cycle, because 'held everything' and 'the bridge refused' look
    identical in a position report."""
    if result.signal is None:
        return f"🧠 Agent PM → market: *not traded* — {result.reason}"
    note = (
        f"🧠 Agent PM → market: {len(result.signal.target_weights)} name(s), "
        f"{result.gross:.1%} of account"
    )
    if result.dropped:
        note += f" · dropped {', '.join(result.dropped)}"
    return note


def stale_by(result: PMSignalResult, *, max_age_h: float | None = None) -> timedelta | None:
    """How far past the freshness limit the decision is, or None."""
    max_age_h = default_max_age_h() if max_age_h is None else max_age_h
    if result.age_hours is None or result.age_hours <= max_age_h:
        return None
    return timedelta(hours=result.age_hours - max_age_h)
