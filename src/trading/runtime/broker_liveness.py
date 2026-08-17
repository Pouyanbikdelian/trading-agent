"""Authenticated IBKR-session liveness, separate from cached account state.

An open Gateway TCP port and a recently-written ``runner.db`` only prove
that the local process is alive.  They do not prove that Gateway still has
an authenticated IBKR session: account data is subscription-backed and can
remain readable after a 2FA/login failure.  The IBKR adapter's
``probe_liveness`` performs ``reqCurrentTime``, a read-only request that
requires a fresh Gateway response.

This module records that result on every snapshot-refresh tick.  It never
asks a broker to connect, reconnect, or restart Gateway; observability must
report an outage rather than turn it into a recovery storm.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from trading.core.logging import logger

FILENAME = "broker_liveness.json"
PROBE_NAME = "reqCurrentTime"


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    """Publish a complete JSON observation, never a partially-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _previous_last_success(path: Path) -> str | None:
    """Read only a well-shaped prior success value; corrupt state is harmless."""
    try:
        payload = json.loads(path.read_text())
        value = payload.get("last_success_at") if isinstance(payload, dict) else None
        return value if isinstance(value, str) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _observed_at(now: datetime | None) -> datetime:
    """Return a UTC observation time without guessing a naive timezone."""
    checked = now or datetime.now(tz=timezone.utc)
    if checked.tzinfo is None:
        raise ValueError("broker liveness timestamp must be timezone-aware")
    return checked.astimezone(timezone.utc)


def record_broker_liveness_failure(
    state_dir: Path,
    detail: str,
    *,
    probe: str,
    now: datetime | None = None,
) -> dict[str, object]:
    """Atomically record a failed read-only broker observation.

    Startup has a distinct failure mode: ``connect()`` can fail before an
    :class:`IbkrBroker` has an API socket for ``probe_liveness`` to inspect.
    Recording that original failure lets the watchdog alert while the runner
    stays in its no-orders bootstrap loop. The helper never calls the broker
    and preserves the prior verified response time for useful diagnostics.
    """
    checked = _observed_at(now)
    path = Path(state_dir) / FILENAME
    payload: dict[str, object] = {
        "checked_at": checked.isoformat(),
        "ready": False,
        "probe": probe,
        "detail": detail[:240] or "unknown failure",
        "last_success_at": _previous_last_success(path),
    }
    _atomic_write(path, payload)
    logger.bind(component="broker_liveness").warning(
        f"broker API probe failed: {payload['detail']}"
    )
    return payload


def record_broker_liveness(
    broker: object,
    state_dir: Path,
    *,
    now: datetime | None = None,
) -> dict[str, object] | None:
    """Probe a supported broker and atomically persist its session state.

    ``Simulator`` and third-party test brokers intentionally have no
    ``probe_liveness`` method, so they return ``None`` and do not create an
    artifact.  An IBKR failure is still recorded as ``ready: false``; the
    ops watchdog can therefore alert immediately and retain the prior
    successful response time for age-based diagnostics.
    """
    probe = getattr(broker, "probe_liveness", None)
    if not callable(probe):
        return None

    checked = _observed_at(now)
    checked_at = checked.isoformat()
    path = Path(state_dir) / FILENAME

    try:
        gateway_time = probe()
        if not isinstance(gateway_time, datetime):
            raise TypeError("reqCurrentTime returned a non-datetime response")
        if gateway_time.tzinfo is None:
            gateway_time = gateway_time.replace(tzinfo=timezone.utc)
        else:
            gateway_time = gateway_time.astimezone(timezone.utc)
        payload: dict[str, object] = {
            "checked_at": checked_at,
            "ready": True,
            "probe": PROBE_NAME,
            "server_time": gateway_time.isoformat(),
            "last_success_at": checked_at,
        }
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"[:240]
        return record_broker_liveness_failure(
            state_dir,
            detail,
            probe=PROBE_NAME,
            now=checked,
        )

    _atomic_write(path, payload)
    return payload
