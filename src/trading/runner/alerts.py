"""Telegram alerts via the bot HTTP API.

We use stdlib ``urllib`` instead of importing ``python-telegram-bot`` — that
library targets bots that *receive* messages, which we don't. A one-way
``sendMessage`` is a tiny POST.

Behavior
--------
* If ``token`` or ``chat_id`` is missing, ``enabled`` is forced ``False``
  and every send is a no-op. That makes test setup trivial.
* All network calls are wrapped in ``try/except`` and log on failure but
  never raise — a flaky chat shouldn't crash the trading loop.
* The runner uses ``info`` / ``warning`` / ``error`` / ``critical`` levels.
  Critical messages get a ``🚨`` prefix so they stand out in mobile
  notifications.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Literal

from trading.core.logging import logger

Level = Literal["info", "warning", "error", "critical"]


class TelegramAlerts:
    """One-way alert sink for the runner."""

    def __init__(
        self,
        *,
        token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self.enabled = bool(enabled and token and chat_id)
        self._sent: list[tuple[Level, str]] = []   # in-memory log for tests

    def info(self, msg: str) -> None:
        self._send("info", msg)

    def warning(self, msg: str) -> None:
        self._send("warning", msg)

    def error(self, msg: str) -> None:
        self._send("error", msg)

    def critical(self, msg: str) -> None:
        self._send("critical", f"🚨 {msg}")

    @property
    def sent(self) -> list[tuple[Level, str]]:
        """Returned in order. Useful for tests that want to assert on what
        the runner alerted on without mocking ``urllib``."""
        return list(self._sent)

    def _send(self, level: Level, msg: str) -> None:
        self._sent.append((level, msg))
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": str(self.chat_id),
            "text": msg,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if resp.status != 200:
                    logger.bind(component="alerts").warning(
                        f"telegram returned status={resp.status} body={body}"
                    )
        except Exception as e:  # noqa: BLE001 — alerts must not raise
            logger.bind(component="alerts").exception(f"telegram send failed: {e!r}")


class NullAlerts(TelegramAlerts):
    """Convenience: explicit no-op sink for tests and dry-runs."""

    def __init__(self) -> None:
        super().__init__(token=None, chat_id=None, enabled=False)
