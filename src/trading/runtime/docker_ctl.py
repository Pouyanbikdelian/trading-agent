"""Talk to the docker daemon over its unix socket. Stdlib only.

The IBKR gateway holds the account's ONE session (IBC is configured
``ExistingSessionDetectedAction=primary``), so while the container is up
the operator cannot log into TWS or the mobile app to trade by hand. The
only way to take manual control is to stop the container — which, from a
phone, previously meant an SSH client.

``IbkrBroker`` already restarted the gateway this way to recover a wedged
API session; the transport is lifted here so the command pipeline can use
it without going through a broker instance, and so stop/start exist
alongside restart.

Why not shell out to the docker CLI: it is not in the image, and adding
it to trade would be a large dependency for four HTTP calls.
"""

from __future__ import annotations

import contextlib
import http.client
import socket as _socket

from trading.core.logging import logger

DOCKER_SOCKET_PATH = "/var/run/docker.sock"
GATEWAY_CONTAINER = "ibkr-gateway"

#: Only these may be targeted from a chat command. The trader must never
#: be able to stop the bot that supervises it, and an operator who can
#: name any container from Telegram is one typo from stopping the VPS.
ALLOWED_CONTAINERS = frozenset({GATEWAY_CONTAINER})

ACTIONS = ("stop", "start", "restart")


class _UDSConnection(http.client.HTTPConnection):
    """HTTPConnection over a unix domain socket instead of TCP."""

    def __init__(self, sock_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._sock_path = sock_path

    def connect(self) -> None:  # type: ignore[override]
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._sock_path)
        self.sock = s


def docker_action(
    container: str,
    action: str,
    *,
    timeout: float = 30.0,
    socket_path: str = DOCKER_SOCKET_PATH,
) -> str:
    """POST /containers/<name>/<action>. Returns a human-readable result.

    Raises ``ValueError`` for a container or action outside the allow
    list, and ``RuntimeError`` on transport or non-2xx failure — the
    caller reports either to the operator. A silent failure here would
    leave someone believing the session was released when it was not,
    and they would find out from a rejected login rather than from us.
    """
    if container not in ALLOWED_CONTAINERS:
        raise ValueError(f"{container!r} is not a container this system may control")
    if action not in ACTIONS:
        raise ValueError(f"{action!r} is not one of {ACTIONS}")

    conn = _UDSConnection(socket_path, timeout=timeout)
    try:
        # t=10 → SIGTERM, wait 10s, then SIGKILL. The gateway handles TERM
        # cleanly, and IBC writes nothing we need to preserve on exit.
        suffix = "?t=10" if action in ("stop", "restart") else ""
        conn.request("POST", f"/containers/{container}/{action}{suffix}")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")[:200]
        # 304 = already in the requested state. Not an error: asking a
        # stopped gateway to stop is exactly what a nervous operator does.
        if resp.status == 304:
            return f"{container} was already {'stopped' if action == 'stop' else 'running'}"
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"docker {action} {container}: HTTP {resp.status} {body}")
        return f"{container} {action} ok"
    except (OSError, http.client.HTTPException) as e:
        raise RuntimeError(f"docker socket unreachable ({type(e).__name__}: {e})") from e
    finally:
        conn.close()


def container_state(
    container: str = GATEWAY_CONTAINER,
    *,
    timeout: float = 10.0,
    socket_path: str = DOCKER_SOCKET_PATH,
) -> str:
    """``running`` / ``exited`` / ``unknown: <reason>``. Never raises.

    Used to answer "is my session free?" without the operator having to
    infer it from a failed login.
    """
    conn = _UDSConnection(socket_path, timeout=timeout)
    try:
        conn.request("GET", f"/containers/{container}/json")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        if not (200 <= resp.status < 300):
            return f"unknown: HTTP {resp.status}"
        import json

        return str(json.loads(body).get("State", {}).get("Status", "unknown"))
    except Exception as e:
        logger.bind(component="docker_ctl").warning(f"state check failed: {e}")
        return f"unknown: {type(e).__name__}"
    finally:
        with contextlib.suppress(Exception):
            conn.close()
