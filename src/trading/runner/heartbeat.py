"""Heartbeat file — atomic JSON write each cycle.

External health checks (Docker healthcheck, systemd watchdog, cron monitor)
read this file and confirm:
  * It exists.
  * It was updated within the last ``max_age`` seconds.
  * Its ``status`` is not ``"error"``.

We write atomically (tmp + rename) so a concurrent reader never sees a
half-written file.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path


def write_heartbeat(
    path: Path,
    *,
    ts: datetime,
    status: str,
    cycle_no: int,
    extra: dict[str, object] | None = None,
) -> None:
    """Atomically write a heartbeat JSON payload to ``path``."""
    if ts.tzinfo is None:
        raise ValueError("heartbeat ts must be timezone-aware")
    payload: dict[str, object] = {
        "ts": ts.isoformat(),
        "status": status,
        "cycle": cycle_no,
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def read_heartbeat(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def heartbeat_age_seconds(path: Path) -> float | None:
    """Wall-clock age of the heartbeat file in seconds, or None if absent."""
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def heartbeat_is_stale(path: Path, max_age_seconds: float) -> bool:
    """True if the heartbeat is missing or older than ``max_age_seconds``."""
    age = heartbeat_age_seconds(path)
    return age is None or age > max_age_seconds
