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
* Messages are sent as Telegram Markdown, retried as plain text if Telegram
  rejects the formatting, and split on line boundaries past 3,800 chars
  (2026-09-26). Until then every runner, PM, committee and cycle message
  went out as plain text, so each ``*bold*``, ``_italic_`` and backtick
  arrived on the phone as a literal symbol — and a message over Telegram's
  4,096-char limit was rejected and lost.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Literal

from trading.core.logging import logger

Level = Literal["info", "warning", "error", "critical"]

#: Telegram rejects messages over 4,096 characters; split below it.
CHUNK_LIMIT = 3_800
MAX_CHUNKS = 5


def split_message(text: str, limit: int = CHUNK_LIMIT, max_chunks: int = MAX_CHUNKS) -> list[str]:
    """Split on line boundaries into chunks of at most ``limit`` characters."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single monster line: hard-wrap it
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        chunks[-1] += "\n…(truncated)"
    return chunks


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
        self._sent: list[tuple[Level, str]] = []  # in-memory log for tests

    def info(self, msg: str, *, buttons: dict[str, Any] | None = None) -> None:
        self._send("info", msg, buttons=buttons)

    def warning(self, msg: str, *, buttons: dict[str, Any] | None = None) -> None:
        self._send("warning", msg, buttons=buttons)

    def error(self, msg: str) -> None:
        self._send("error", msg)

    def critical(self, msg: str) -> None:
        self._send("critical", f"🚨 {msg}")

    @property
    def sent(self) -> list[tuple[Level, str]]:
        """Returned in order. Useful for tests that want to assert on what
        the runner alerted on without mocking ``urllib``."""
        return list(self._sent)

    def _send(self, level: Level, msg: str, *, buttons: dict[str, Any] | None = None) -> None:
        self._sent.append((level, msg))
        if not self.enabled:
            return
        chunks = split_message(msg)
        for i, chunk in enumerate(chunks):
            # Buttons ride on the last chunk, next to the compose box.
            self._post(chunk, buttons if i == len(chunks) - 1 else None)

    def _post(self, text: str, buttons: dict[str, Any] | None) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        fields: dict[str, str] = {
            "chat_id": str(self.chat_id),
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }
        if buttons is not None:
            # Form-encoded endpoint: reply_markup must be a JSON *string*.
            fields["reply_markup"] = json.dumps(buttons)
        try:
            self._request(url, fields)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                logger.bind(component="alerts").exception(f"telegram send failed: {e!r}")
                return
            # 400 is almost always "can't parse entities": an underscore in a
            # ticker or env name. The words matter more than the styling.
            fields.pop("parse_mode", None)
            try:
                self._request(url, fields)
            except Exception as e2:
                logger.bind(component="alerts").exception(f"telegram send failed: {e2!r}")
        except Exception as e:
            logger.bind(component="alerts").exception(f"telegram send failed: {e!r}")

    def _request(self, url: str, fields: dict[str, str]) -> None:
        data = urllib.parse.urlencode(fields).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                logger.bind(component="alerts").warning(
                    f"telegram returned status={resp.status} body={body}"
                )


class NullAlerts(TelegramAlerts):
    """Convenience: explicit no-op sink for tests and dry-runs."""

    def __init__(self) -> None:
        super().__init__(token=None, chat_id=None, enabled=False)
