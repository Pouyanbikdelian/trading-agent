"""External dead-man's switch.

Every alert this system can send originates inside the trader container.
If the VPS dies, the container is OOM-killed, or Docker stops, the silence
is indistinguishable from a quiet market. A dead-man's switch inverts that:
the runner pings an external monitor (e.g. a free healthchecks.io check) on
a fixed interval, and the *monitor* alerts when the pings stop.

Configured by ``HEALTHCHECK_PING_URL``; absent means disabled. A failed
ping is logged and never raised — the monitor's own missing-ping alarm is
the failure signal, and a flaky HTTPS call must not disturb the runner.
"""

from __future__ import annotations

from typing import Any

import httpx

from trading.core.logging import logger

PING_TIMEOUT_S = 5.0
PING_INTERVAL_MINUTES = 5


def ping(url: str | None, *, client: Any = None) -> bool:
    """GET ``url``; True on a 2xx. Never raises."""
    if not url:
        return False
    try:
        if client is None:
            resp = httpx.get(url, timeout=PING_TIMEOUT_S)
        else:
            resp = client.get(url, timeout=PING_TIMEOUT_S)
        ok = 200 <= int(resp.status_code) < 300
        if not ok:
            logger.bind(component="deadman").warning(f"dead-man ping HTTP {resp.status_code}")
        return ok
    except Exception as e:
        logger.bind(component="deadman").warning(f"dead-man ping failed: {type(e).__name__}")
        return False
