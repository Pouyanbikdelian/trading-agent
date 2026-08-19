"""The one way to write ``halt.json``.

Four places used to write this file — ``/halt`` and ``/resume`` in the
bot, ``trading halt`` and ``trading resume`` in the CLI, and the
runner's auto-halt — and all four wrote a fresh four-key dict:
``halted``, ``reason``, ``halted_at``, ``flatten_on_next_cycle``.

``HaltState`` carries three more fields: ``equity_high_watermark``,
``daily_equity_open`` and ``last_day``. It neither forbids extra keys
nor requires the missing ones, so every one of those writes silently
erased the kill-switch baselines and the next load quietly defaulted
them to ``0.0 / 0.0 / None``.

The daily-loss switch skips a zero baseline entirely
(``if daily_equity_open > 0``), and the drawdown switch restarts its
peak from wherever equity happens to be. So halting and resuming gave
both kill switches amnesia — and the drawdown switch could never fire on
a decline that spanned a halt, which is the exact situation in which
somebody halted.

It is also, by accident, what cleared the poisoned CHF 1,068,862
watermark on 2026-08-07. Being saved by a bug is not the same as being
safe.

So: read, modify, write. The halt trio is the only thing any caller may
change; everything else on disk survives untouched. Unrecognised keys
are dropped deliberately — see ``flatten_on_next_cycle`` below.

**On ``flatten_on_next_cycle``**: it was written by ``/halt`` and by the
CLI, and read by nothing, anywhere. The only force-flatten in the system
comes from a playbook regime rule. ``/halt`` replied "Next cycle will
force-flatten positions" and no cycle ever did. The flag is gone rather
than implemented: halting is reversible and liquidating is not, and one
word should not market-sell a book. ``/flatten`` remains the explicit,
separate way to exit — and it, unlike a halt, is meant to be irreversible.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from trading.core.file_lock import file_lock
from trading.core.logging import logger
from trading.risk.limits import HaltState

FILENAME = "halt.json"

#: Fields a halt/resume may touch. Everything else on disk is preserved.
_HALT_TRIO = ("halted", "reason", "halted_at")


def halt_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / FILENAME


def _read_halt_state_unlocked(path: Path) -> HaltState:
    """Best-effort parse while the caller owns ``file_lock(path)``."""
    if not path.exists():
        return HaltState()
    try:
        return HaltState.model_validate_json(path.read_text())
    except Exception as e:
        logger.bind(component="halt_file").warning(
            f"{path} unreadable ({type(e).__name__}); treating as fresh state"
        )
        return HaltState()


def _write_halt_state_unlocked(path: Path, state: HaltState) -> None:
    """Atomically publish a full state while the caller owns its file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(state.model_dump_json(indent=2))
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_halt_state(state_dir: Path | str) -> HaltState:
    """Load the persisted state, defaulting on anything unreadable.

    Unlike ``RiskManager._load_state`` this does NOT fail closed — it is
    used by writers that are about to overwrite the file anyway, and by
    read-only callers who want a best-effort view.
    """
    path = halt_path(state_dir)
    with file_lock(path):
        return _read_halt_state_unlocked(path)


def write_halt_state(state_dir: Path | str, state: HaltState) -> Path:
    """Atomically persist a full ``HaltState``, under the cross-process lock."""
    path = halt_path(state_dir)
    with file_lock(path):
        _write_halt_state_unlocked(path, state)
    return path


def set_halted(
    state_dir: Path | str,
    *,
    halted: bool,
    reason: str = "",
    halted_at: datetime | None = None,
) -> HaltState:
    """Flip the halt flag WITHOUT disturbing the kill-switch baselines.

    ``halted_at`` defaults to now when halting and to None when clearing.
    Always timezone-aware: the runner's auto-halt used a naive
    ``datetime.now()``, which on a UTC-scheduled system in a non-UTC
    container is a timestamp nobody can interpret later.
    """
    path = halt_path(state_dir)
    with file_lock(path):
        # This is one transaction, not a read followed by a separately
        # locked write.  The 60-second risk monitor concurrently updates
        # the high-water mark, and an operator /halt or /resume must never
        # clobber those counters (or be clobbered by them).
        current = _read_halt_state_unlocked(path)
        if halted and halted_at is None:
            halted_at = datetime.now(tz=timezone.utc)
        if not halted:
            halted_at = None
            reason = ""
        new = current.replace(halted=halted, reason=reason, halted_at=halted_at)
        _write_halt_state_unlocked(path, new)
    logger.bind(component="halt_file").warning(
        f"halt.json: halted={halted} reason={reason!r} "
        f"(baselines preserved: open={new.daily_equity_open:,.0f} "
        f"hwm={new.equity_high_watermark:,.0f})"
    )
    return new


def describe_exposure(state_dir: Path | str) -> str:
    """One line on what a halt does NOT do, for the operator's reply.

    A halt stops trading; it does not reduce exposure. Neither do the
    daily-loss and drawdown kill switches — they halt too. The position
    guards are the only thing that exits on their own, and they are off
    by default. Saying so at halt time is the difference between an
    operator who has decided to sit exposed and one who thinks they are
    covered.
    """
    from trading.runtime.guards import enabled as guards_enabled

    if guards_enabled():
        return (
            "Positions are unchanged and still fully exposed. Trailing-stop "
            "guards remain active and may still exit a position; nothing else "
            "will trade. `/flatten` to exit everything."
        )
    return (
        "Positions are unchanged and still fully exposed. Guards are OFF, so "
        "*nothing* will reduce this book while halted. `/flatten` to exit."
    )
