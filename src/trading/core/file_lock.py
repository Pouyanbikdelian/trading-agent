r"""Cross-process advisory file lock for state files shared between processes.

The trader and the Telegram bot run in separate containers but share
``state/`` via a Docker volume. A few state files (``halt.json``) are
read-modify-written by both. Without locking, simultaneous ops can
silently drop changes — the audit (May 2026, item #9) flagged this.

Why ``fcntl.flock`` and not a sentinel/PID file
-----------------------------------------------
``fcntl.flock`` is OS-level advisory locking that all POSIX processes
respect when they participate. Process death automatically releases the
lock (the file descriptor closes), so a crashed trader can't leave the
bot deadlocked. Limitations:

* POSIX only (we run on Linux; macOS dev also works).
* Doesn't work across NFS in the general case. We're on a local volume
  so this is moot.
* Advisory — non-participating writers can still clobber. All our own
  code uses this helper, so cooperation is enforced.

Usage
-----
::

    with file_lock(state_dir / "halt.json"):
        payload = read_or_default(...)
        new_payload = mutate(payload)
        atomic_write(...)

The lock is on a sibling ``.lock`` file so the data file itself can
still be atomically replaced via ``os.replace`` inside the locked block.
"""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive lock keyed to ``path`` for the duration of the
    context. Blocks until granted; released automatically on exit or if
    the process dies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    # Open in append mode so we never truncate an existing lock file, and
    # so file creation works without races (O_CREAT semantics).
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            # Explicit unlock — also happens on close, but explicit is clearer.
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
