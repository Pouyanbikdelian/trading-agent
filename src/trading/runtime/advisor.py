r"""Advisory alerter — informs the operator when the risk monitor fires,
*without* changing anything in the trading pipeline.

The detector fires far too often to be trusted with auto-execution (see
``experiments/backtest_modes.py``: -$2.5M over 11 years vs the raw
strategy). What it IS good at is flagging market regime *changes* so a
human gets a heads-up before the news cycle catches up. That's this
module's only job: poll SPY + VIX on a schedule, debounce repeated
firings of the same trigger, and push a Telegram alert when something
genuinely new shows up.

State persistence
-----------------
We persist the *set of currently-active trigger names* to
``state/advisor.json``. On the next poll we diff against the new set
and only alert on:

  * a name that wasn't active last time (re-arm / first firing)
  * a severity change (LIGHT → HEAVY etc.)
  * a recovery — all triggers cleared after at least one was active

Whether the operator acts on the alert is entirely up to them. The
mode system stays in NEUTRAL until the operator manually flips it. We
do not touch ``mode.json`` from here.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading.core.config import settings
from trading.core.logging import logger
from trading.runtime.risk_monitor import MonitorConfig, Trigger, evaluate

STATE_FILENAME = "advisor.json"


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
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


def _format_alert(triggers: list[Trigger]) -> str:
    """One Telegram-ready message summarising the active triggers."""
    if not triggers:
        return "✅ *RECOVERY signal* — all risk triggers cleared.\nConsider `/mode neutral`."
    top = max(triggers, key=lambda t: int(t.severity))
    lines = [
        f"⚠️ *RISK signal* — `{top.name}` (severity: `{top.severity.name}`)",
        f"_{top.detail}_",
        "",
        f"Suggested mode: `{top.suggested_mode.value}`",
        "",
        "This is **advisory only** — no automatic action. Run `/mode "
        f"{top.suggested_mode.value}` if you want to act.",
    ]
    other = [t for t in triggers if t is not top]
    if other:
        lines.append("")
        lines.append("*Other active triggers:*")
        for t in other:
            lines.append(f"  • `{t.name}` ({t.severity.name}): {t.detail}")
    return "\n".join(lines)


async def poll_and_alert(
    *,
    spy: pd.Series,
    vix: pd.Series | None = None,
    state_path: Path | None = None,
    cfg: MonitorConfig | None = None,
) -> dict[str, Any]:
    r"""Run a single poll cycle.

    1. Run the risk monitor over the supplied SPY/VIX series.
    2. Diff against ``state/advisor.json``.
    3. Send a Telegram alert for any genuinely-new event.
    4. Persist the new state.

    Returns a dict describing what happened (useful in tests).
    """
    cfg = cfg or MonitorConfig()
    state_path = state_path or (settings.state_dir / STATE_FILENAME)
    prior = _read_state(state_path)
    prior_active: set[str] = set(prior.get("active", []))
    prior_severities: dict[str, int] = dict(prior.get("severities", {}))

    triggers = evaluate(spy, vix, cfg=cfg)
    now_active = {t.name for t in triggers}
    now_severities = {t.name: int(t.severity) for t in triggers}

    new_or_escalated = [
        t
        for t in triggers
        if t.name not in prior_active or now_severities[t.name] > prior_severities.get(t.name, 0)
    ]
    cleared = bool(prior_active) and not now_active

    sent = None
    if new_or_escalated:
        text = _format_alert(triggers)
        sent = await _send_telegram(text)
    elif cleared:
        sent = await _send_telegram(_format_alert([]))

    _write_state(
        state_path,
        {
            "active": sorted(now_active),
            "severities": now_severities,
            "last_polled_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return {
        "triggers": [t.to_dict() for t in triggers],
        "new_or_escalated": [t.name for t in new_or_escalated],
        "cleared": cleared,
        "alert_sent": bool(sent),
    }


async def _send_telegram(text: str) -> bool:
    """Best-effort outbound. Lazily imported so the advisor can be unit-
    tested without Telegram configured."""
    try:
        from trading.bot.notifier import send_message
    except Exception:
        logger.warning("advisor: cannot import telegram notifier; alert dropped")
        return False
    return await send_message(text)
