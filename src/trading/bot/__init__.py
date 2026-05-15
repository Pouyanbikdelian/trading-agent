"""Telegram bot — outbound alerts + inbound command surface for the runner.

Two surfaces:

* :func:`trading.bot.notifier.send_message` — fire-and-forget outbound
  message used by the runner to push alerts (halts, fills, weekly
  reports). No bot loop required; pure HTTP POST.
* :func:`trading.bot.telegram.run_bot` — long-polling command bot
  (``/status``, ``/halt``, ``/resume``, ``/positions``, ``/report``).
  Run as a separate systemd service alongside the runner.

Both share the same authorization model: only the chat ID stored in
``settings.telegram_chat_id`` is allowed to issue commands or receive
alerts. Unknown chats are ignored.
"""

from __future__ import annotations

from trading.bot.notifier import send_message, send_message_sync
from trading.bot.telegram import run_bot

__all__ = ["run_bot", "send_message", "send_message_sync"]
