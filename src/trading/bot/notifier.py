r"""Outbound Telegram alerts.

Single-purpose: POST a message to the Bot API ``sendMessage`` endpoint
using the token and chat ID from settings. No polling loop, no command
dispatch — that lives in ``trading.bot.telegram``.

Used by the runner to push:

* halt notifications (manual or auto-triggered by the risk manager)
* cycle failure tracebacks
* daily/weekly reports
* heartbeat-stale alerts

The functions degrade gracefully: if no token or chat ID is configured,
they log a debug line and return without raising. This means a runner
deployed without Telegram credentials still works — alerts just go
nowhere.
"""

from __future__ import annotations

import httpx

from trading.core.config import settings
from trading.core.logging import logger

BOT_API_BASE = "https://api.telegram.org"

# Telegram limits a single message to 4096 characters. We truncate
# (rather than splitting) to keep the surface simple — reports get
# chunked at a higher level if needed.
MAX_MESSAGE_LENGTH = 4000


def _enabled() -> tuple[str, str] | None:
    """Return (token, chat_id) if both are configured, else None."""
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        return None
    return token, chat_id


def _truncate(text: str) -> str:
    if len(text) <= MAX_MESSAGE_LENGTH:
        return text
    return text[: MAX_MESSAGE_LENGTH - 32] + "\n\n... [truncated]"


async def send_message(text: str, *, parse_mode: str | None = "Markdown") -> bool:
    """POST a message to the configured Telegram chat.

    Returns True on a 2xx response, False on any failure (network, auth,
    or missing config). Never raises — callers are typically inside a
    runner loop where one bad alert mustn't kill the cycle.
    """
    creds = _enabled()
    if creds is None:
        logger.debug("telegram not configured; skipping send")
        return False
    token, chat_id = creds
    url = f"{BOT_API_BASE}/bot{token}/sendMessage"
    payload: dict[str, str | bool] = {
        "chat_id": chat_id,
        "text": _truncate(text),
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, json=payload)
        if r.status_code >= 400:
            logger.warning(f"telegram sendMessage failed: {r.status_code} {r.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"telegram sendMessage error: {e}")
        return False


def send_message_sync(text: str, *, parse_mode: str | None = "Markdown") -> bool:
    """Synchronous variant — useful from CLI commands and tests."""
    creds = _enabled()
    if creds is None:
        logger.debug("telegram not configured; skipping send")
        return False
    token, chat_id = creds
    url = f"{BOT_API_BASE}/bot{token}/sendMessage"
    payload: dict[str, str | bool] = {
        "chat_id": chat_id,
        "text": _truncate(text),
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json=payload)
        if r.status_code >= 400:
            logger.warning(f"telegram sendMessage failed: {r.status_code} {r.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"telegram sendMessage error: {e}")
        return False
