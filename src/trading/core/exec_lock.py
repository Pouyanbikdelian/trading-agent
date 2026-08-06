r"""The single execution lock — one mutex over every path to the broker.

Flagged by external review 2026-07-15 and open until 2026-08-06. Four
independent paths can submit orders:

1. the scheduled cycle (``runner`` container, APScheduler),
2. the on-demand cycle (flag file, 30s poll),
3. the approval flow (operator ``/approve`` → command processor),
4. manual commands (``/buy``, ``/close``, ``/flatten``),

plus the position guards, which route exits through the command pipeline.
They ran under *per-job* locks and a 10-second cooldown — which prevents
a job racing itself, and does nothing about two different jobs racing
each other. The system already shipped one order-stacking incident from
this family (2026-07-14/15: stacked after-hours orders took two names
short on paper).

Why a file lock rather than an asyncio lock
-------------------------------------------
The trader and the bot are **separate containers**. An in-process lock
cannot see across them. ``fcntl.flock`` on the shared ``state/`` volume
can, and it releases automatically when a process dies — so a crashed
runner cannot leave the operator unable to flatten.

Why non-blocking, unlike ``core.file_lock``
-------------------------------------------
``file_lock`` blocks indefinitely, which is right for a quick
read-modify-write of ``halt.json`` and badly wrong here. **A lock that
can deadlock the operator's ``/flatten`` is worse than no lock at all.**
This one polls with a deadline and then gives up loudly: the caller
decides whether to skip a cycle or tell the operator to retry, and
either beats hanging.

Scope: hold it around the BROKER MUTATION only — submit plus fill
reconciliation, which is seconds. Holding it across a whole cycle would
block the operator for the five minutes a price refresh can take.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from trading.core.logging import logger

LOCK_FILENAME = "execution.lock"

#: Default patience. Long enough to sit behind a normal submit burst
#: (a basket of ~10 orders is a couple of seconds), short enough that a
#: human waiting on a Telegram reply gets an answer.
DEFAULT_TIMEOUT_S = 30.0

#: Poll interval while waiting. Cheap syscall; 50ms keeps latency low
#: without spinning a core on a 1-vCPU box.
_POLL_S = 0.05


class ExecutionBusyError(RuntimeError):
    """Raised when the execution lock could not be acquired in time.

    Carries whatever the current holder wrote about itself, so the
    operator is told *what* is busy rather than just that something is.
    """

    def __init__(self, holder: str, waited_s: float) -> None:
        self.holder = holder
        self.waited_s = waited_s
        super().__init__(f"execution locked by {holder} (waited {waited_s:.1f}s)")


def lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / LOCK_FILENAME


def current_holder(state_dir: Path) -> str:
    """Best-effort description of whoever holds the lock. Never raises."""
    try:
        raw = json.loads(lock_path(state_dir).read_text())
        return (
            f"{raw.get('holder', '?')} (pid {raw.get('pid', '?')}, since {raw.get('since', '?')})"
        )
    except Exception:
        return "unknown"


@contextlib.contextmanager
def execution_lock(
    state_dir: Path,
    *,
    holder: str,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Iterator[None]:
    """Exclusive, cross-process, non-blocking-with-deadline.

    ``holder`` is a short label written into the lock file for
    diagnostics — "cycle", "command:/flatten", "approval". Raises
    :class:`ExecutionBusyError` rather than waiting forever.

    The lock file is truncated and rewritten on each acquisition, so its
    contents always describe the CURRENT holder. Stale contents from a
    crashed process are harmless: flock already released, and the next
    holder overwrites them.
    """
    path = lock_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, timeout)
    started = time.monotonic()

    # Append mode so concurrent creation is race-free; we seek+truncate
    # only once the lock is actually ours.
    with open(path, "a+") as fh:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    waited = time.monotonic() - started
                    other = current_holder(state_dir)
                    logger.bind(component="exec_lock").warning(
                        f"{holder} could not acquire execution lock: held by {other}"
                    )
                    raise ExecutionBusyError(other, waited) from None
                time.sleep(_POLL_S)

        try:
            fh.seek(0)
            fh.truncate()
            fh.write(
                json.dumps(
                    {
                        "holder": holder,
                        "pid": os.getpid(),
                        "since": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                    }
                )
            )
            fh.flush()
            waited = time.monotonic() - started
            if waited > 1.0:
                # Contention is not an error, but it IS the signal that two
                # paths tried to trade at once — worth seeing in the log.
                logger.bind(component="exec_lock").info(
                    f"{holder} waited {waited:.1f}s for the execution lock"
                )
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
