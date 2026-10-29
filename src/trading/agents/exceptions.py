"""Structured creative exceptions to the mechanical candidate ladder.

The ladder is the baseline, not a prison.  An agent, committee or operator
can advance an off-ladder name, but the exception must be a falsifiable,
time-bounded research record before it can ever be considered for the broker.
This module is deliberately outside the order path: it records proposals and
answers the narrow question, "has an operator explicitly approved this exact
exception?"  The risk manager remains the only route to any order.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from trading.core.clock import UtcClock
from trading.core.file_lock import file_lock
from trading.core.logging import logger

EXCEPTIONS_FILE = "candidate_exceptions.json"
MAX_EXCEPTION_DAYS = 21
MAX_STOCK_WEIGHT = 0.10
VALID_ORIGINS = frozenset(("agent", "committee", "operator"))


def exception_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / EXCEPTIONS_FILE


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
        return (
            [dict(row) for row in payload if isinstance(row, dict)]
            if isinstance(payload, list)
            else []
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _symbols(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip() for value in values if str(value).strip()})[:8]


def normalise_proposal(
    raw: Any,
    *,
    symbol: str,
    origin: str,
    requested_weight: float,
    correlation_review: Any = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate the full exception contract; incomplete ideas remain ideas.

    `requested_weight` is included so the review card can compare model intent
    with the exception's bounded risk budget.  It never raises: malformed LLM
    JSON must degrade to a pending/rejected proposal, never a trading failure.
    """
    if not isinstance(raw, dict) or origin not in VALID_ORIGINS:
        return None
    sym = str(symbol).upper().strip()
    thesis = str(raw.get("thesis") or "").strip()
    source_ids = _symbols(raw.get("source_ids"))
    invalidation = str(raw.get("invalidation") or "").strip()
    sector_impact = str(raw.get("sector_impact") or "").strip()
    horizon_raw = raw.get("horizon_days")
    max_weight_raw = raw.get("max_weight")
    if not isinstance(horizon_raw, (int, float, str)) or not isinstance(
        max_weight_raw, (int, float, str)
    ):
        return None
    try:
        horizon_days = int(horizon_raw)
        max_weight = float(max_weight_raw)
        requested = float(requested_weight)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not sym
        or len(thesis) < 20
        or not source_ids
        or len(invalidation) < 12
        or not sector_impact
        or not 1 <= horizon_days <= 365
        or not 0 < max_weight <= MAX_STOCK_WEIGHT
        or not isfinite(requested)
        or requested <= 0
    ):
        return None
    # A model may not use the exception paperwork to enlarge a position.
    # The lower of declared and requested risk is the only meaningful budget.
    now = now or UtcClock().now()
    return {
        "id": f"ex-{uuid.uuid4().hex[:10]}",
        "symbol": sym,
        "origin": origin,
        "status": "proposed",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=MAX_EXCEPTION_DAYS)).isoformat(),
        "thesis": thesis[:1_500],
        "source_ids": source_ids,
        "horizon_days": horizon_days,
        "invalidation": invalidation[:800],
        "max_weight": min(max_weight, requested),
        "sector_impact": sector_impact[:500],
        "correlation_impact": str(raw.get("correlation_impact") or "not supplied")[:500],
        "correlation_review": correlation_review if isinstance(correlation_review, dict) else {},
    }


def record_proposals(
    state_dir: Path | str, proposals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return durable proposal records, including active retries of the same review.

    Retained simulated holdings need the same review as new ones. Dropping a
    duplicate from the return value made that pending review disappear from
    the next PM digest; expired records must also allow a fresh proposal.
    """
    if not proposals:
        return []
    path = exception_path(state_dir)
    with file_lock(path):
        rows = _read(path)
        recorded: list[dict[str, Any]] = []
        changed = False
        for proposal in proposals:
            # A retry of the same PM response should not create a stack of
            # indistinguishable approvals.  An expired/rejected record remains
            # history; a new thesis is allowed through.
            point = datetime.fromisoformat(str(proposal["created_at"]))
            review_fields = (
                "symbol",
                "origin",
                "thesis",
                "source_ids",
                "horizon_days",
                "invalidation",
                "max_weight",
                "sector_impact",
                "correlation_impact",
            )
            duplicate = next(
                (
                    row
                    for row in rows
                    if row.get("status") in {"proposed", "approved"}
                    and _not_expired(row, point)
                    and all(row.get(key) == proposal.get(key) for key in review_fields)
                ),
                None,
            )
            if duplicate is not None:
                recorded.append(duplicate)
            else:
                rows.append(proposal)
                recorded.append(proposal)
                changed = True
        if changed:
            _atomic_write(path, rows)
            logger.bind(component="candidate_exceptions").info(
                f"recorded {len(recorded)} structured exception proposal(s)"
            )
        return recorded


def _not_expired(row: dict[str, Any], now: datetime) -> bool:
    try:
        return datetime.fromisoformat(str(row["expires_at"])) > now
    except (KeyError, TypeError, ValueError):
        return False


def approved_symbols(state_dir: Path | str, *, now: datetime | None = None) -> set[str]:
    """Only active, explicitly operator-approved exceptions may reach PM bridge."""
    return set(approved_exception_limits(state_dir, now=now))


def approved_exception_limits(
    state_dir: Path | str,
    *,
    now: datetime | None = None,
    proposal_ids: set[str] | None = None,
) -> dict[str, float]:
    """Approved symbol -> the risk budget that was actually reviewed."""
    point = now or datetime.now(tz=timezone.utc)
    out: dict[str, float] = {}
    for row in _read(exception_path(state_dir)):
        if proposal_ids is not None and row.get("id") not in proposal_ids:
            continue
        if row.get("status") != "approved" or not _not_expired(row, point):
            continue
        try:
            limit = float(row["max_weight"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 < limit <= MAX_STOCK_WEIGHT:
            continue
        symbol = str(row.get("symbol")).upper()
        # Multiple approvals for a symbol must use the most conservative
        # current budget; an old broad approval cannot enlarge a new thesis.
        out[symbol] = min(out.get(symbol, limit), limit)
    return out


def set_exception_status(
    state_dir: Path | str, exception_id: str, status: str, *, now: datetime | None = None
) -> dict[str, Any] | None:
    """Operator-facing state transition helper; approvals remain time-bounded."""
    if status not in {"approved", "rejected", "expired"}:
        raise ValueError(f"unsupported exception status: {status!r}")
    path = exception_path(state_dir)
    with file_lock(path):
        rows = _read(path)
        point = now or datetime.now(tz=timezone.utc)
        for row in rows:
            if row.get("id") != exception_id or row.get("status") != "proposed":
                continue
            if status == "approved" and not _not_expired(row, point):
                return None
            row["status"] = status
            row["reviewed_at"] = point.isoformat()
            _atomic_write(path, rows)
            return row
        return None


def list_exceptions(
    state_dir: Path | str, *, active_only: bool = False, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Newest-first audit view; expired records are preserved as history."""
    rows = _read(exception_path(state_dir))
    if active_only:
        point = now or UtcClock().now()
        rows = [
            row
            for row in rows
            if row.get("status") in {"proposed", "approved"} and _not_expired(row, point)
        ]
    rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return rows


def exception_block_reason(
    state_dir: Path | str,
    symbols: set[str],
    *,
    now: datetime | None = None,
    proposal_ids: set[str] | None = None,
) -> str:
    """Missing paperwork needs a new PM review, not an impossible approval.

    Read the ledger rather than last_run's cached status: approvals, rejection
    and expiry can all occur after the PM decided.
    """
    point = now or UtcClock().now()
    blocked = symbols - set(
        approved_exception_limits(state_dir, now=point, proposal_ids=proposal_ids)
    )
    pending = {
        str(row.get("symbol", "")).upper()
        for row in list_exceptions(state_dir, active_only=True, now=point)
        if row.get("status") == "proposed"
        and (proposal_ids is None or row.get("id") in proposal_ids)
    } & blocked
    missing = blocked - pending
    parts = []
    if missing:
        parts.append(
            f"{', '.join(sorted(missing))}: exception proposal missing or expired; "
            "use `/pm run` for a fresh research decision with complete evidence before approval"
        )
    if pending:
        parts.append(
            f"{', '.join(sorted(pending))}: operator approval required; "
            "review the proposal with `/exceptions`"
        )
    return "; ".join(parts)
