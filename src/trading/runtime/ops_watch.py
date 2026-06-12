"""Ops watchdog — infrastructure health, no LLM, separate channel.

Hourly mechanical checks on the box and the data plumbing:

* disk usage and available memory on the host (as seen from the container);
* freshness of every state artifact (broker snapshot, news, econ, macro,
  committee, PM book) against per-artifact tolerances;
* trading halt state (a halted runner at 2am is news the operator wants).

Issues go to a dedicated ops Telegram channel when ``OPS_TELEGRAM_BOT_TOKEN``
/ ``OPS_TELEGRAM_CHAT_ID`` are set — keeping infrastructure noise out of the
trading chat — and fall back to the main channel otherwise. Each distinct
issue alerts at most once per ``DEBOUNCE_HOURS``; a recovery message is sent
when a previously-reported issue clears. Silence means healthy.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

STATE_FILENAME = "ops_watch.json"
DEBOUNCE_HOURS = 6.0
DISK_WARN_PCT = 85.0
MEM_WARN_AVAILABLE_MB = 150.0

# artifact -> (relative path, max age in hours before it counts as stale)
_FRESHNESS: dict[str, tuple[str, float]] = {
    "broker snapshot": ("runner.db", 80.0),  # survives weekends
    "news watch": ("news.json", 80.0),
    "econ watch": ("econ_watch.json", 100.0),
    "macro watch": ("market_watch.json", 80.0),
    "committee": ("last_committee.json", 80.0),
    "PM book": ("agent_pm/portfolio.json", 200.0),  # weekly cadence
}


def _mem_available_mb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return float(ln.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def check_health(state_dir: Path, *, now: datetime | None = None) -> list[str]:
    """Mechanical pass. Returns human-readable issue strings; [] = healthy."""
    now = now or datetime.now(tz=timezone.utc)
    issues: list[str] = []

    try:
        du = shutil.disk_usage("/")
        pct = du.used / du.total * 100.0
        if pct >= DISK_WARN_PCT:
            issues.append(f"disk {pct:.0f}% full ({du.free / 1e9:.1f} GB free)")
    except Exception:
        pass

    mem = _mem_available_mb()
    if mem is not None and mem < MEM_WARN_AVAILABLE_MB:
        issues.append(f"memory low: {mem:.0f} MB available")

    for label, (rel, max_h) in _FRESHNESS.items():
        p = state_dir / rel
        if not p.exists():
            issues.append(f"{label}: missing ({rel})")
            continue
        age_h = (now.timestamp() - p.stat().st_mtime) / 3600.0
        if age_h > max_h:
            issues.append(f"{label}: stale ({age_h:.0f}h old, limit {max_h:.0f}h)")

    try:
        halt = json.loads((state_dir / "halt.json").read_text())
        if halt.get("halted"):
            issues.append(f"trading HALTED: {halt.get('reason', 'no reason recorded')}")
    except Exception:
        pass

    return issues


def _send_ops(text: str) -> bool:
    """Send to the ops channel; fall back to the main trading channel."""
    import httpx

    token = os.getenv("OPS_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("OPS_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "disable_web_page_preview": True},
            timeout=10.0,
        )
        return r.status_code == 200
    except Exception as e:
        logger.bind(component="ops_watch").warning(f"ops telegram send failed: {e}")
        return False


def _load(state_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(state_dir) / STATE_FILENAME).read_text())
    except Exception:
        return {"reported": {}}


def _save(state_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def run_ops_watch(state_dir: Path, *, now: datetime | None = None) -> dict[str, Any]:
    """One watchdog cycle: check, debounce per issue, alert, track recovery."""
    now = now or datetime.now(tz=timezone.utc)
    issues = check_health(state_dir, now=now)
    state = _load(state_dir)
    reported: dict[str, str] = dict(state.get("reported", {}))

    # Issue identity = text before the first ':' — values change, kind doesn't.
    current = {i.split(":")[0].split("(")[0].strip(): i for i in issues}
    new_alerts: list[str] = []
    for key, text in current.items():
        last = reported.get(key)
        if last:
            try:
                age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600
                if age_h < DEBOUNCE_HOURS:
                    continue
            except Exception:
                pass
        new_alerts.append(text)
        reported[key] = now.isoformat()

    recovered = [k for k in list(reported) if k not in current]
    for k in recovered:
        del reported[k]

    if new_alerts:
        _send_ops("⚠️ Ops watchdog\n" + "\n".join(f"• {a}" for a in new_alerts))
    if recovered and not new_alerts:
        _send_ops("✅ Ops watchdog: recovered — " + ", ".join(recovered))

    _save(state_dir, {"reported": reported, "last_run": now.isoformat()})
    if new_alerts:
        logger.bind(component="ops_watch").warning(f"issues: {new_alerts}")
    return {"issues": issues, "alerted": new_alerts, "recovered": recovered}
