"""Pre-cycle broker readiness — ask for the human BEFORE it matters.

IBKR requires two-factor authentication for every user; there is no
API-only exemption. The gateway re-authenticates after IBKR's daily
auto-restart and its mandatory weekly shutdown, and each of those can
raise an IBKR Mobile prompt. IBC retries a missed prompt rather than
dying, so a missed tap is recoverable — but only if somebody taps.

Every other consequence of a dead gateway is non-fatal: a monitor
degrades, a snapshot is skipped, the dashboard shows stale numbers. The
ONE moment it matters is the cycle, because a cycle that cannot reach
the broker submits nothing, and a silently skipped rebalance looks
exactly like a rebalance that decided to do nothing.

So this runs an hour ahead of the cycle and, if the broker is not
usable, sends one Telegram message while there is still time to act.
That converts "notice on Monday that Friday never traded" into "tap
your phone at 20:05".

Deliberately a REAL broker call, not a TCP probe. The gateway's port
accepts connections the whole time it is sitting at a login screen —
`docker compose ps` says healthy and nothing works. Only asking for the
account proves the session is authenticated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from trading.core.logging import logger


def check_broker_ready(broker: Any) -> dict[str, Any]:
    """Is the broker usable right now? Never raises.

    ``{"ready": bool, "detail": str, "equity": float | None}``.
    """
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    try:
        snap = broker.get_account()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"[:200]
        logger.bind(component="broker_ready").warning(f"broker not ready: {detail}")
        return {"ready": False, "detail": detail, "equity": None, "checked_at": now}

    equity = getattr(snap, "equity", None)
    if equity is None:
        # A snapshot with no equity means the session answered but has no
        # account data — the shape a half-authenticated gateway returns.
        return {
            "ready": False,
            "detail": "broker returned a snapshot with no equity — session may be half-open",
            "equity": None,
            "checked_at": now,
        }
    return {"ready": True, "detail": "ok", "equity": float(equity), "checked_at": now}


def format_not_ready_alert(result: dict[str, Any], *, minutes_to_cycle: int) -> str:
    """The message that has to make someone pick up their phone.

    Leads with the deadline, because the only thing that matters is how
    long there is left to fix it.
    """
    return (
        f"🔴 *Broker not ready — cycle in {minutes_to_cycle} min*\n"
        f"`{result.get('detail', 'unknown')}`\n\n"
        "The gateway is not answering with account data. Most likely it is "
        "waiting on an IBKR Mobile 2FA prompt after a restart.\n\n"
        "*Do now:*\n"
        "1. Open IBKR Mobile and approve any pending login.\n"
        "2. If there is no prompt, force a fresh login:\n"
        "   `docker compose restart ib-gateway` — then watch for the prompt.\n"
        "3. Re-check with `/health`.\n\n"
        "_If this is not fixed, the cycle will submit nothing and say so._"
    )


def format_ready_note(result: dict[str, Any], *, minutes_to_cycle: int) -> str:
    """Quiet confirmation. Sent only when it had previously failed —
    a green message every week is a message nobody reads."""
    eq = result.get("equity")
    return f"🟢 Broker ready — cycle in {minutes_to_cycle} min" + (
        f" · equity {eq:,.0f}" if isinstance(eq, (int, float)) else ""
    )
