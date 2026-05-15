r"""Long-polling Telegram command bot.

Runs as a separate process from the trading runner. Receives commands
from the operator's phone, mutates the on-disk state (halt.json), and
responds with status snapshots pulled from the same SQLite stores the
runner writes to.

Why long polling and not webhooks
---------------------------------
Webhooks require a publicly reachable HTTPS endpoint with a valid TLS
cert. That's needless complexity for a single-operator bot — long
polling works through any NAT, no inbound ports, no TLS. The trade-off
is a single open HTTP connection 24/7, which is trivial.

Authorization
-------------
Only the chat ID configured in ``settings.telegram_chat_id`` is
accepted. Every other message is silently ignored. Bots can be added
to groups; the authorization check prevents anyone else in such a
group (or a typo'd chat) from issuing commands.

Commands
--------
``/start``       — greeting + help
``/help``        — command list
``/status``      — env, halted, heartbeat age, last cycle outcome
``/positions``   — current positions and weights
``/report``      — generate a fresh weekly report and send the summary
``/halt``        — set halt.json. Optional reason after the slash.
``/resume``      — clear halt.json
``/heartbeat``   — runner heartbeat age in seconds
"""

from __future__ import annotations

import asyncio
import json
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from trading.core.config import settings
from trading.core.logging import logger

BOT_API_BASE = "https://api.telegram.org"
POLL_TIMEOUT = 25  # seconds — long-poll
HELP_TEXT = (
    "*Trading bot — commands*\n"
    "/status — env, halted, heartbeat, last cycle\n"
    "/positions — current positions and weights\n"
    "/report — generate and send the weekly report\n"
    "/halt [reason] — force-flatten on next cycle\n"
    "/resume — clear halt\n"
    "/heartbeat — last cycle age\n"
)


# ---------------------------------------------------------------------------
# Low-level Bot API helpers
# ---------------------------------------------------------------------------


async def _send(client: httpx.AsyncClient, token: str, chat_id: str, text: str) -> None:
    """POST sendMessage. Never raises — bot loop must keep running."""
    try:
        r = await client.post(
            f"{BOT_API_BASE}/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10.0,
        )
        if r.status_code >= 400:
            logger.warning(f"telegram send failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.warning(f"telegram send error: {e}")


async def _get_updates(client: httpx.AsyncClient, token: str, offset: int) -> list[dict[str, Any]]:
    """Long-poll for the next batch of updates."""
    try:
        r = await client.get(
            f"{BOT_API_BASE}/bot{token}/getUpdates",
            params={"offset": offset, "timeout": POLL_TIMEOUT},
            timeout=POLL_TIMEOUT + 5,
        )
        if r.status_code >= 400:
            logger.warning(f"telegram getUpdates failed: {r.status_code}")
            return []
        data = r.json()
        return data.get("result", []) if data.get("ok") else []
    except httpx.TimeoutException:
        return []  # normal — no new messages
    except Exception as e:
        logger.warning(f"telegram getUpdates error: {e}")
        await asyncio.sleep(2)
        return []


# ---------------------------------------------------------------------------
# Command handlers — each returns the reply text
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _cmd_halt(args: list[str]) -> str:
    reason = " ".join(args) if args else "telegram"
    halt_path = settings.state_dir / "halt.json"
    _atomic_write_json(
        halt_path,
        {
            "halted": True,
            "reason": reason,
            "halted_at": datetime.now(tz=timezone.utc).isoformat(),
            "flatten_on_next_cycle": True,
        },
    )
    logger.bind(reason=reason).warning("telegram halt")
    return f"🛑 *HALTED* — reason: `{reason}`\nNext cycle will force-flatten positions."


def _cmd_resume() -> str:
    halt_path = settings.state_dir / "halt.json"
    _atomic_write_json(
        halt_path,
        {
            "halted": False,
            "reason": "",
            "halted_at": None,
            "flatten_on_next_cycle": False,
        },
    )
    logger.info("telegram resume")
    return "✅ *RESUMED* — halt cleared."


def _cmd_status() -> str:
    halt_path = settings.state_dir / "halt.json"
    halted = False
    halt_reason = ""
    if halt_path.exists():
        try:
            payload = json.loads(halt_path.read_text())
            halted = bool(payload.get("halted", False))
            halt_reason = str(payload.get("reason", ""))
        except Exception:
            pass

    hb_age = _heartbeat_age()
    hb_line = "unknown" if hb_age is None else f"{hb_age:.0f}s ago"
    halt_line = f"🛑 *HALTED* — `{halt_reason}`" if halted else "🟢 running"
    return (
        "*Trading status*\n"
        f"env: `{settings.trading_env}`\n"
        f"live armed: `{settings.is_live_armed()}`\n"
        f"state: {halt_line}\n"
        f"heartbeat: {hb_line}\n"
    )


def _heartbeat_age() -> float | None:
    hb_path = settings.state_dir / "heartbeat.json"
    if not hb_path.exists():
        return None
    age = datetime.now(tz=timezone.utc).timestamp() - hb_path.stat().st_mtime
    return float(age)


def _cmd_heartbeat() -> str:
    age = _heartbeat_age()
    if age is None:
        return "_no heartbeat file yet — runner hasn't completed a cycle._"
    return f"heartbeat updated `{age:.0f}s` ago"


def _cmd_positions() -> str:
    # Pull the latest snapshot from the runner store. Lazily imported so
    # the bot doesn't require sqlite to be initialised at import time.
    try:
        from trading.runner.state import RunnerStore

        store = RunnerStore(settings.state_dir / "runner.db")
        snap = store.latest_snapshot()
    except Exception as e:
        return f"could not read positions: `{e}`"
    if snap is None:
        return "_no snapshot yet — runner hasn't completed a cycle._"
    if not snap.positions:
        return "_no open positions._"

    lines = [f"*Equity:* `${snap.equity:,.2f}`  *Cash:* `${snap.cash:,.2f}`", ""]
    for _key, pos in sorted(snap.positions.items()):
        mv = pos.quantity * pos.avg_price + pos.unrealized_pnl
        weight = mv / snap.equity if snap.equity > 0 else 0.0
        lines.append(
            f"`{pos.instrument.symbol:<6}` qty `{pos.quantity:>8.2f}` "
            f"avg `${pos.avg_price:>8.2f}` w `{weight:>6.2%}`"
        )
    return "\n".join(lines)


def _cmd_report() -> str:
    """Generate a fresh weekly report and return its executive summary."""
    try:
        import contextlib

        from trading.reporting import (
            fetch_news_for_symbols,
            gather_weekly_report,
            summarise,
        )

        report = gather_weekly_report(fetch_vix=True)
        if report.positions:
            with contextlib.suppress(Exception):
                # News fetch is best-effort; never let a network blip
                # kill the report.
                report.news_by_symbol = fetch_news_for_symbols(
                    list(report.positions.keys()), max_per_symbol=3
                )
        return summarise(report)
    except Exception as e:
        return f"report generation failed: `{e}`"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _dispatch(text: str) -> str | None:
    """Parse a command and return a reply, or None if not a command."""
    if not text or not text.startswith("/"):
        return None
    parts = shlex.split(text)
    cmd = parts[0].lower().split("@")[0]  # strip "@botname" suffix
    args = parts[1:]

    if cmd in ("/start", "/help"):
        return HELP_TEXT
    if cmd == "/status":
        return _cmd_status()
    if cmd == "/heartbeat":
        return _cmd_heartbeat()
    if cmd == "/positions":
        return _cmd_positions()
    if cmd == "/halt":
        return _cmd_halt(args)
    if cmd == "/resume":
        return _cmd_resume()
    if cmd == "/report":
        return _cmd_report()
    return f"unknown command `{cmd}` — try /help"


async def run_bot() -> None:
    """Run the long-poll loop. Blocks until cancelled.

    Authorization: only messages from ``settings.telegram_chat_id`` are
    processed. Everything else is silently ignored.
    """
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set in .env")

    logger.info("telegram bot starting (long-poll)")
    offset = 0
    async with httpx.AsyncClient() as client:
        # Greet on startup so the operator knows the bot is up.
        await _send(
            client,
            token,
            chat_id,
            "🤖 *Bot online.* Use /help for commands.",
        )
        while True:
            updates = await _get_updates(client, token, offset)
            for upd in updates:
                offset = max(offset, upd["update_id"] + 1)
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                # Authorization: ignore anything not from the configured chat.
                msg_chat = str(msg.get("chat", {}).get("id"))
                if msg_chat != str(chat_id):
                    logger.warning(f"telegram unauthorized chat {msg_chat}")
                    continue
                text = msg.get("text", "")
                reply = await _dispatch(text)
                if reply is not None:
                    await _send(client, token, chat_id, reply)
