r"""Operator-set portfolio mode.

The runner has *one* settable mode at any time, persisted to
``state/mode.json``. Telegram bot, CLI, and the runner all read/write
this file via the helpers below. Atomic-write semantics so a half-
flushed file is never visible to a concurrent reader.

Modes — what they mean
----------------------

``BULL``     — maximum offense. Pass strategy weights through unchanged.
``NEUTRAL``  — default. Same as BULL today, but separated so a future
               build can introduce mild de-risking here without breaking
               the explicit "bull" semantic.
``DEFENSE``  — soft de-risk: scale the strategy sleeve to 70% gross, fill
               the other 30% with a defensive ETF basket
               (``defensive_sleeve``). This is the mode you want during
               a slow grind down or when realised vol picks up.
``BEAR``     — hard de-risk: 50% to defensive ETFs, 50% cash. Roughly the
               same risk as a 60/40 balanced portfolio.
``FLATTEN``  — close everything. Equivalent to setting halt.json but
               without the "refuses to trade until manually cleared"
               semantic — flatten just sets target weights to zero so the
               runner's next cycle rebalances to cash.

Mode selection vs the halt flag
-------------------------------
``halt.json`` is the kill switch — it stops the runner from acting at
all and force-flattens. ``mode.json`` is the *style* of acting — it
shapes the target weights but the runner still runs. The two are
orthogonal: a halted system stays halted even in BULL mode; an unhalted
system runs whatever mode is active.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class Mode(str, Enum):
    BULL = "bull"
    NEUTRAL = "neutral"
    DEFENSE = "defense"
    BEAR = "bear"
    FLATTEN = "flatten"

    @classmethod
    def parse(cls, raw: str) -> Mode:
        try:
            return cls(raw.strip().lower())
        except ValueError as e:
            raise ValueError(f"unknown mode {raw!r}; choose from {[m.value for m in cls]}") from e


@dataclass(frozen=True)
class ModeState:
    """The persisted record. Frozen — we mutate by writing a new file."""

    mode: Mode = Mode.NEUTRAL
    set_at: str = ""  # ISO 8601 UTC
    set_by: str = "default"  # "telegram" | "cli" | "auto" | "default"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModeState:
        return cls(
            mode=Mode.parse(payload.get("mode", "neutral")),
            set_at=str(payload.get("set_at", "")),
            set_by=str(payload.get("set_by", "default")),
            reason=str(payload.get("reason", "")),
        )


def read_mode(path: Path) -> ModeState:
    """Read the persisted mode. Returns NEUTRAL default on missing/corrupt file."""
    if not path.exists():
        return ModeState()
    try:
        return ModeState.from_dict(json.loads(path.read_text()))
    except Exception:
        # Corrupt file — fail safe to NEUTRAL. The next write will
        # repair it. We never throw here because the runner cycle must
        # not be killed by a bad mode file.
        return ModeState()


def write_mode(path: Path, mode: Mode, *, set_by: str = "cli", reason: str = "") -> ModeState:
    """Atomically write a new mode. Returns the persisted state."""
    state = ModeState(
        mode=mode,
        set_at=datetime.now(tz=timezone.utc).isoformat(),
        set_by=set_by,
        reason=reason,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return state


# ---------------------------------------------------------------------------
# Pending-change ticket — used by the bot for the preview/confirm flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingModeChange:
    """A staged mode change waiting for user CONFIRM.

    Stored separately from ``mode.json`` so a preview never affects the
    runner. Expires after ``ttl_seconds`` so a forgotten preview can't
    later be confirmed by accident.
    """

    new_mode: Mode
    requested_at: str
    requested_by: str
    reason: str = ""
    ttl_seconds: int = 600  # 10 minutes
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["new_mode"] = self.new_mode.value
        return d

    @classmethod
    def from_dict(cls, p: dict[str, Any]) -> PendingModeChange:
        return cls(
            new_mode=Mode.parse(p.get("new_mode", "neutral")),
            requested_at=str(p.get("requested_at", "")),
            requested_by=str(p.get("requested_by", "unknown")),
            reason=str(p.get("reason", "")),
            ttl_seconds=int(p.get("ttl_seconds", 600)),
            extra=dict(p.get("extra", {})),
        )

    def is_expired(self, *, now: datetime | None = None) -> bool:
        if not self.requested_at:
            return True
        now = now or datetime.now(tz=timezone.utc)
        try:
            then = datetime.fromisoformat(self.requested_at)
        except ValueError:
            return True
        return (now - then).total_seconds() > self.ttl_seconds


def read_pending(path: Path) -> PendingModeChange | None:
    if not path.exists():
        return None
    try:
        return PendingModeChange.from_dict(json.loads(path.read_text()))
    except Exception:
        return None


def write_pending(path: Path, pending: PendingModeChange) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(pending.to_dict(), f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def clear_pending(path: Path) -> None:
    if path.exists():
        path.unlink()
