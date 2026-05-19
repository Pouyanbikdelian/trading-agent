r"""File-based command queue: Telegram bot ↔ runner.

Why file-based instead of in-process
------------------------------------
The Telegram bot and the trader live in *separate* docker containers.
They can't share Python objects. We could:

  * Run a small HTTP/socket server in the trader → bot makes API calls.
    Works, but adds a network surface + auth burden + a dependency on
    the trader being responsive.

  * Give the bot its own ``IbkrBroker`` connection (clientId 18).
    Cleanest API, but two clients on the same gateway sometimes confuse
    IBKR's session state, and the bot would race the runner.

  * **File queue (this module)**. Bot drops a JSON command into a
    shared docker volume; the runner has a fast watcher that picks it
    up, executes it, and pushes a Telegram result alert. Zero races
    with the cycle thread (the runner serialises all broker work
    through one event loop), and the volume is already shared. The
    cost is a few hundred ms of latency, which is invisible at our
    weekly-rebalance cadence.

Lifecycle of one command
------------------------

  bot                          runner
   │ writes pending/<uuid>.json
   │ -----------------------> │ watcher (every 5s) picks it up
   │                           │ moves it to running/<uuid>.json
   │                           │ executes via broker
   │ Telegram                  │ sends Telegram result
   │ <----------------------- │
   │                           │ moves to executed/<uuid>.json

State transitions are atomic file renames so a crash/restart between
steps never loses a command — the watcher just resumes whichever
state file it finds.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Sub-directory names. The watcher polls ``pending`` and is the only
# writer of ``running`` / ``executed``; the bot is the only writer of
# ``pending``. Each command flows pending → running → executed.
PENDING_DIR = "pending"
RUNNING_DIR = "running"
EXECUTED_DIR = "executed"


class CommandType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    FLATTEN = "flatten"
    CANCEL_ORDER = "cancel_order"
    FX_CONVERT = "fx_convert"
    REFRESH_DATA = "refresh_data"
    RECONNECT_BROKER = "reconnect_broker"


@dataclass(frozen=True)
class Command:
    """One queued command. ``args`` is type-specific (see handlers)."""

    id: str
    type: CommandType
    args: dict[str, Any] = field(default_factory=dict)
    requested_by: str = "telegram"
    requested_at: str = ""  # ISO 8601

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Command:
        return cls(
            id=str(payload["id"]),
            type=CommandType(payload["type"]),
            args=dict(payload.get("args", {})),
            requested_by=str(payload.get("requested_by", "telegram")),
            requested_at=str(payload.get("requested_at", "")),
        )

    @classmethod
    def new(
        cls,
        cmd_type: CommandType,
        *,
        args: dict[str, Any] | None = None,
        requested_by: str = "telegram",
    ) -> Command:
        return cls(
            id=uuid.uuid4().hex,
            type=cmd_type,
            args=args or {},
            requested_by=requested_by,
            requested_at=datetime.now(tz=timezone.utc).isoformat(),
        )


def _state_root(state_dir: Path) -> Path:
    return state_dir / "commands"


def _ensure_dirs(state_dir: Path) -> None:
    root = _state_root(state_dir)
    for sub in (PENDING_DIR, RUNNING_DIR, EXECUTED_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)


def submit(cmd: Command, state_dir: Path) -> Path:
    r"""Write a pending command. Returns the path written.

    Atomic — we write to a temp file inside the same directory and
    rename into place so the watcher never reads a half-written file.
    """
    _ensure_dirs(state_dir)
    target = _state_root(state_dir) / PENDING_DIR / f"{cmd.id}.json"
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f"{cmd.id}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(cmd.to_dict(), f, indent=2)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return target


def pending_commands(state_dir: Path) -> list[Command]:
    r"""Return all currently-pending commands, oldest first.

    The watcher calls this every poll. It only lists *pending* — any
    command already moved to ``running`` is being executed by a prior
    poll and we should not pick it up again.
    """
    pending_dir = _state_root(state_dir) / PENDING_DIR
    if not pending_dir.exists():
        return []
    out: list[tuple[float, Path]] = []
    for p in pending_dir.iterdir():
        if not p.name.endswith(".json") or p.name.endswith(".tmp"):
            continue
        try:
            out.append((p.stat().st_mtime, p))
        except OSError:
            continue
    out.sort(key=lambda x: x[0])
    cmds: list[Command] = []
    for _, p in out:
        try:
            cmds.append(Command.from_dict(json.loads(p.read_text())))
        except Exception:
            # Corrupt file — move it aside so we don't keep retrying.
            with contextlib.suppress(OSError):
                p.rename(p.with_suffix(".corrupt"))
    return cmds


def mark_running(cmd: Command, state_dir: Path) -> Path:
    r"""Atomically move pending/<id>.json → running/<id>.json.

    If the move fails because pending/ no longer has the file (another
    watcher beat us to it), raises FileNotFoundError so the caller can
    skip the command.
    """
    src = _state_root(state_dir) / PENDING_DIR / f"{cmd.id}.json"
    dst = _state_root(state_dir) / RUNNING_DIR / f"{cmd.id}.json"
    os.replace(src, dst)
    return dst


def mark_executed(
    cmd: Command,
    state_dir: Path,
    *,
    status: str,
    result: dict[str, Any] | str | None = None,
) -> Path:
    r"""Move running/<id>.json → executed/<id>.json with an embedded
    result/error blob. Operator can grep the executed/ dir to audit."""
    src = _state_root(state_dir) / RUNNING_DIR / f"{cmd.id}.json"
    dst = _state_root(state_dir) / EXECUTED_DIR / f"{cmd.id}.json"
    # Re-write with the result baked in.
    payload = cmd.to_dict()
    payload["status"] = status
    payload["completed_at"] = datetime.now(tz=timezone.utc).isoformat()
    if result is not None:
        payload["result"] = result
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f"{cmd.id}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    # Best-effort cleanup of the running/ file. If the move-to-dst above
    # was successful but src still exists (shouldn't happen — we wrote
    # dst directly, not from src), remove it.
    if src.exists():
        with contextlib.suppress(OSError):
            src.unlink()
    return dst
