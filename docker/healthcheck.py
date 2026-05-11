"""Docker HEALTHCHECK script.

Exit 0 if the heartbeat file is present, fresh, and reports a non-error
status. Exit 1 otherwise. Never raises — Docker only reads the exit code.

Usage::

    python healthcheck.py <heartbeat-path> <max-age-seconds>

The Dockerfile invokes us with 300s (5 min) which is generous for a daily
runner; tighten when scheduling intraday cycles.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main(path: str, max_age_seconds: float) -> int:
    p = Path(path)
    if not p.exists():
        print(f"heartbeat missing: {p}", file=sys.stderr)
        return 1

    age = time.time() - p.stat().st_mtime
    if age > max_age_seconds:
        print(f"heartbeat stale: {age:.0f}s > {max_age_seconds:.0f}s", file=sys.stderr)
        return 1

    try:
        payload = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"heartbeat unreadable: {e!r}", file=sys.stderr)
        return 1

    status = str(payload.get("status", "")).lower()
    if status == "error":
        print(f"heartbeat reports error: {payload!r}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 3:
        print("usage: healthcheck.py <path> <max-age-seconds>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1], float(sys.argv[2])))
