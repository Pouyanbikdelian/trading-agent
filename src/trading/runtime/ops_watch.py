"""Ops watchdog — infrastructure health, no LLM, separate channel.

Hourly mechanical checks on the box and the data plumbing:

* disk usage and available memory on the host (as seen from the container);
* freshness of every state artifact (broker snapshot, news, econ, macro,
  committee, PM book) against per-artifact tolerances;
* trading halt state (a halted runner at 2am is news the operator wants);
* **liveness of the learning loops** — did the committee, the PM, the
  historian and the nightly memory pass actually produce anything;
* **errors logged in the last hour**, grouped by call site.

The last two exist because the first three were all green on 2026-08-06
while three separate loops had been running-and-achieving-nothing for
weeks. Freshness asks whether a file was touched. It cannot ask whether
the work inside succeeded, and every one of those failures wrote a
perfectly ordinary log line and updated a perfectly ordinary mtime. A
watchdog that only checks that things RAN will keep missing this class of
bug; these checks ask what was PRODUCED.

Issues go to a dedicated ops Telegram channel when ``OPS_TELEGRAM_BOT_TOKEN``
/ ``OPS_TELEGRAM_CHAT_ID`` are set — keeping infrastructure noise out of the
trading chat — and fall back to the main channel otherwise. Each distinct
issue alerts at most once per ``DEBOUNCE_HOURS``; a recovery message is sent
when a previously-reported issue clears. Silence means healthy.
"""

from __future__ import annotations

import json
import os
import re
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


# --------------------------------------------------------------- liveness
#
# Everything above this line checks that a FILE is fresh. That is not the
# same as checking that the work succeeded, and on 2026-08-06 the gap
# turned out to be the whole ballgame: the nightly grader, the shadow
# ledger and the weekly historian had each been running-and-achieving-
# nothing for as long as they had existed, while every mtime check stayed
# green. One raised a TypeError swallowed by a broad `except`; one turned
# the same error into a `None` indistinguishable from "not due yet"; one
# was never scheduled at all because of an indentation accident.
#
# So these checks ask a different question: did the loop PRODUCE anything?
# A loop that is supposed to write a journal row and has not written one
# in twice its cadence is broken, whatever its logs say.
#
# journal kind -> (label, max age in hours before it counts as dead)
_JOURNAL_CADENCE: dict[str, tuple[str, float]] = {
    "committee": ("committee debate", 96.0),  # 2x/week
    "agent_pm": ("agent PM run", 240.0),  # weekly
    "historian": ("historian distillation", 264.0),  # weekly, Friday
    "daily": ("nightly memory pass", 48.0),  # nightly
}


def check_learning_loops(state_dir: Path, *, now: datetime | None = None) -> list[str]:
    """Did the learning machinery actually do anything lately?

    Returns issue strings; [] means the loops are alive. Never raises —
    a watchdog that can crash is a watchdog that stops watching.
    """
    now = now or datetime.now(tz=timezone.utc)
    issues: list[str] = []
    db = Path(state_dir) / "memory" / "memory.db"
    if not db.exists():
        return []  # nothing to say before the first run creates it

    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return ["memory database unreadable"]

    try:
        for kind, (label, max_h) in _JOURNAL_CADENCE.items():
            try:
                row = conn.execute(
                    "SELECT MAX(ts) AS ts FROM journal WHERE kind = ?", (kind,)
                ).fetchone()
            except sqlite3.Error:
                continue
            last = row["ts"] if row else None
            if last is None:
                issues.append(f"{label}: has NEVER run")
                continue
            age_h = (now.timestamp() - float(last)) / 3600.0
            if age_h > max_h:
                issues.append(f"{label}: last ran {age_h / 24:.1f}d ago (limit {max_h / 24:.0f}d)")

        # Predictions that came due and were never scored. This is the
        # exact signature of the tz bug: they accumulate forever while the
        # grader logs a cheerful nothing every night.
        try:
            overdue = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE graded_ts IS NULL AND due_ts <= ?",
                (now.timestamp() - 48 * 3600,),
            ).fetchone()[0]
            graded = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE graded_ts IS NOT NULL"
            ).fetchone()[0]
            if overdue >= 5:
                issues.append(
                    f"{overdue} prediction(s) overdue and ungraded — the scorecard is not scoring"
                )
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if total >= 20 and graded == 0:
                issues.append(
                    f"{total} predictions recorded, ZERO ever graded — agent calibration is empty"
                )
        except sqlite3.Error:
            pass

        # Shadow legs matured but never filled — the counterfactual ledger
        # failing quietly looks exactly like a ledger patiently waiting.
        try:
            stale_legs = conn.execute(
                "SELECT COUNT(*) FROM shadow WHERE ts <= ? AND ret_21d IS NULL",
                (now.timestamp() - 45 * 86400,),
            ).fetchone()[0]
            if stale_legs >= 10:
                issues.append(f"{stale_legs} shadow leg(s) matured but ungraded")
        except sqlite3.Error:
            pass  # older schema without the shadow table
    finally:
        conn.close()
    return issues


_ERROR_LINE_RE = re.compile(r"^\S+ \S+ \| (ERROR|CRITICAL)\s+\| ([^\s]+) - (.*)$")
_ERROR_ALERT_THRESHOLD = 1  # one unhandled error is worth knowing about


def check_recent_errors(log_dir: Path, *, hours: float = 1.5) -> list[str]:
    """ERROR/CRITICAL lines from the last ``hours`` of today's log.

    The catch-all. Most failures in this system are caught by a broad
    ``except`` that calls ``logger.exception`` and continues — correct
    behaviour for a trading loop that must not die over a bad symbol, but
    it means a real bug produces a log line nobody reads and nothing
    else. Every silent failure found on 2026-08-06 had been printing to
    this file, nightly, for weeks.

    Grouped by call site so a loop that throws two hundred times is one
    alert, not two hundred.
    """
    now = datetime.now(tz=timezone.utc)
    path = Path(log_dir) / f"trading.{now.date().isoformat()}.log"
    if not path.exists():
        return []
    cutoff = now.timestamp() - hours * 3600.0
    counts: dict[str, int] = {}
    try:
        # Tail only: these files reach tens of MB and the watchdog runs
        # hourly on a 1-vCPU box.
        with path.open("rb") as f:
            f.seek(0, 2)
            start = max(0, f.tell() - 400_000)
            f.seek(start)
            lines = f.read().decode("utf-8", "replace").splitlines()
            # Only discard the first line when we actually seeked into the
            # middle of the file and it may therefore be a fragment. A
            # short log starts at byte 0 and its first line is complete —
            # dropping it there loses a real error.
            tail = lines[1:] if start > 0 else lines
    except OSError:
        return []
    for line in tail:
        m = _ERROR_LINE_RE.match(line)
        if not m:
            continue
        try:
            stamp = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if stamp.timestamp() < cutoff:
            continue
        where, msg = m.group(2), m.group(3)[:120]
        counts[f"{where} — {msg}"] = counts.get(f"{where} — {msg}", 0) + 1
    if not counts:
        return []
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    return [f"error in {k}" + (f" (x{n})" if n > 1 else "") for k, n in top]


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

    # Outcome checks, not mtime checks — see _JOURNAL_CADENCE above.
    try:
        issues.extend(check_learning_loops(state_dir, now=now))
    except Exception:
        logger.bind(component="ops_watch").exception("learning-loop check failed")

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


def run_ops_watch(
    state_dir: Path,
    *,
    now: datetime | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """One watchdog cycle: check, debounce per issue, alert, track recovery.

    ``log_dir`` opts into the error-log scan and is passed explicitly by
    the runner. It is NOT read from settings by default, and that is the
    point: an ambient read made this function depend on whatever any
    other component had logged, so "is the system healthy" answered
    differently on every call. State freshness and error recency are two
    different questions — keep them separate, merge at the reporting
    layer, and let the caller say which it wants.
    """
    now = now or datetime.now(tz=timezone.utc)
    issues = check_health(state_dir, now=now)
    if log_dir is not None:
        try:
            issues.extend(check_recent_errors(log_dir))
        except Exception:
            logger.bind(component="ops_watch").exception("error-log scan failed")

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
