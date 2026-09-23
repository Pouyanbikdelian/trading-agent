r"""Two-step confirmation for operator commands that can do real damage.

Until 2026-09-23 a single word moved the whole account: ``/flatten`` sold
everything, ``/resume`` re-armed a halted desk, and ``/buy`` of any size was
queued straight to the broker with no position, gross or notional check
(``command_processor._h_buy`` bypasses the risk manager by design, because
the operator's own long-term trades are not the desk's). A fat finger, an
autocorrected ticker, or an old message re-sent from the chat history
reached IBKR unchallenged.

The rule now:

* ``/flatten`` and ``/resume`` (while halted) always stage and need an
  explicit confirm.
* ``/buy``, ``/sell`` and ``/close`` stage when the estimated notional is at
  least ``MANUAL_ORDER_CONFIRM_PCT`` of account equity, or when the price
  cannot be estimated at all (unknown size is treated as large). Above
  ``MANUAL_ORDER_MAX_PCT`` they are refused outright.
* ``/halt`` never stages. The brake must stay one word.

One slot, one token. Staging something new replaces what was staged, and
a confirm must quote the token of the thing it confirms, so a button in
the chat history can never approve a different request than it showed.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STAGE_FILE = "bot_confirm_pending.json"
TTL = timedelta(minutes=5)
KINDS = frozenset({"buy", "sell", "close", "flatten", "resume"})


@dataclass(frozen=True)
class StagedCommand:
    token: str
    kind: str
    payload: dict[str, Any]
    summary: str
    staged_at: datetime

    def expired(self, now: datetime) -> bool:
        return now - self.staged_at > TTL


def _path(state_dir: Path) -> Path:
    return Path(state_dir) / STAGE_FILE


def _now(now: datetime | None) -> datetime:
    value = now or datetime.now(tz=timezone.utc)
    if value.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return value


def stage(
    state_dir: Path,
    kind: str,
    payload: dict[str, Any],
    summary: str,
    *,
    now: datetime | None = None,
) -> StagedCommand:
    if kind not in KINDS:
        raise ValueError(f"cannot stage {kind!r}")
    staged = StagedCommand(
        token=secrets.token_hex(3).upper(),
        kind=kind,
        payload=dict(payload),
        summary=summary,
        staged_at=_now(now),
    )
    path = _path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "token": staged.token,
                "kind": staged.kind,
                "payload": staged.payload,
                "summary": staged.summary,
                "staged_at": staged.staged_at.isoformat(),
            }
        )
    )
    os.replace(tmp, path)
    return staged


def peek(state_dir: Path) -> StagedCommand | None:
    """The staged command, or None if nothing (readable) is staged."""
    try:
        raw = json.loads(_path(state_dir).read_text())
        staged_at = datetime.fromisoformat(str(raw["staged_at"]))
        if staged_at.tzinfo is None:
            return None
        kind = str(raw["kind"])
        if kind not in KINDS:
            return None
        return StagedCommand(
            token=str(raw["token"]),
            kind=kind,
            payload=dict(raw.get("payload") or {}),
            summary=str(raw.get("summary") or ""),
            staged_at=staged_at,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def discard(state_dir: Path) -> StagedCommand | None:
    staged = peek(state_dir)
    _path(state_dir).unlink(missing_ok=True)
    return staged


def take(state_dir: Path, token: str | None, *, now: datetime | None = None) -> StagedCommand | str:
    """Consume the staged command if ``token`` matches and it is fresh.

    Returns the command, or a human-readable refusal. The slot is cleared
    on success and on expiry, never on a token mismatch — a stale button
    must not cancel the request that replaced it.
    """
    staged = peek(state_dir)
    if staged is None:
        return "nothing is waiting for confirmation."
    if token is None or token.strip().upper() != staged.token:
        return (
            f"that confirmation is not the current one. Waiting: `{staged.kind}` "
            f"(`/confirm {staged.token}`)."
        )
    if staged.expired(_now(now)):
        discard(state_dir)
        return f"⏱️ the `{staged.kind}` confirmation expired. Send the command again."
    discard(state_dir)
    return staged


def needs_confirmation(
    notional: float | None, equity: float | None, *, confirm_pct: float, max_pct: float
) -> tuple[str, float | None]:
    """Classify a manual order as ``"direct"``, ``"confirm"`` or ``"refuse"``.

    Returns the decision and the order's share of equity when known. An
    unknown price or equity is ``"confirm"``: the size cannot be proven
    small, so it is treated as large.
    """
    if notional is None or equity is None or equity <= 0:
        return "confirm", None
    share = abs(notional) / equity
    if share > max_pct:
        return "refuse", share
    if share >= confirm_pct:
        return "confirm", share
    return "direct", share
