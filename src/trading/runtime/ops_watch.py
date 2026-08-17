"""Ops watchdog — infrastructure health, no LLM, separate channel.

Five-minute mechanical checks on the box and the data plumbing:

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

from trading.core.clock import artifact_age_seconds
from trading.core.logging import logger

STATE_FILENAME = "ops_watch.json"
DEBOUNCE_HOURS = 6.0
DISK_WARN_PCT = 85.0
MEM_WARN_AVAILABLE_MB = 150.0
# Snapshot/account data can be a stale IBKR subscription. A separate
# reqCurrentTime response must arrive at least this often to call the
# authenticated broker session healthy.
BROKER_LIVENESS_MAX_AGE_SECONDS = 5 * 60.0

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
    "historian": ("historian distillation", 120.0),  # Tuesday + Friday
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

        # The latest daily pass explains whether individual due predictions
        # are a normal next-session wait. Read it before counting overdue
        # rows: a Friday/Saturday/Sunday daily close needs Monday's cache
        # bar, and those exact IDs are not evidence that grading is broken.
        # Old journal payloads have no IDs and therefore receive no
        # exemption — conservative compatibility is intentional.
        latest_daily_payload: dict[str, Any] | None = None
        latest_daily_at: datetime | None = None
        latest_daily_is_fresh = False
        try:
            latest_daily = conn.execute(
                """SELECT ts, payload FROM journal WHERE kind = 'daily'
                   ORDER BY ts DESC LIMIT 1"""
            ).fetchone()
            if latest_daily:
                daily_ts = float(latest_daily["ts"])
                daily_age = now.timestamp() - daily_ts
                # Do not let a clock-skewed future row suppress an alert.
                # Unlike the status payload below, the exact-ID exemption
                # deliberately survives a long market holiday: it expires
                # on the next settled NYSE session, not after 48 hours.
                if daily_age >= 0:
                    payload = json.loads(latest_daily["payload"])
                    if isinstance(payload, dict):
                        latest_daily_payload = payload
                        latest_daily_at = datetime.fromtimestamp(daily_ts, tz=timezone.utc)
                        latest_daily_is_fresh = daily_age <= 48 * 3600
        except (sqlite3.Error, TypeError, ValueError, OverflowError, json.JSONDecodeError):
            pass

        # Predictions that came due and were never scored. This is the
        # exact signature of the tz bug: they accumulate forever while the
        # grader logs a cheerful nothing every night.
        try:
            overdue_rows = conn.execute(
                "SELECT id FROM predictions WHERE graded_ts IS NULL AND due_ts <= ?",
                (now.timestamp() - 48 * 3600,),
            ).fetchall()
            awaiting_ids: set[str] = set()
            if latest_daily_payload is not None and latest_daily_at is not None:
                raw_awaiting = latest_daily_payload.get(
                    "awaiting_next_daily_bar_prediction_ids", []
                )
                if isinstance(raw_awaiting, list):
                    # A journal row is evidence of the state at its own
                    # timestamp, not a 48-hour blanket exemption.  Once a
                    # later NYSE session has settled, a stale row must not
                    # hide a cache or grader failure that prevented the next
                    # daily pass from replacing it.
                    try:
                        from trading.runtime.portfolio_stats import nyse_session_settled_since

                        still_expected = not nyse_session_settled_since(latest_daily_at, now)
                    except Exception:
                        # This watchdog promises never to raise. Fail closed
                        # to an overdue alert if the local calendar cannot
                        # establish that a stale journal is still legitimate.
                        still_expected = False
                    if still_expected:
                        awaiting_ids = {str(prediction_id) for prediction_id in raw_awaiting}
            overdue = sum(str(row["id"]) not in awaiting_ids for row in overdue_rows)
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

        # The latest daily pass is the current scorecard state. Reading an
        # old one-off failure record kept an alert alive for 48 hours even
        # after a later clean grading pass had recovered it.
        if latest_daily_payload is not None and latest_daily_is_fresh:
            blocked_parts: list[str] = []
            for key, label in (
                ("unpriced_subjects", "missing prices"),
                ("cache_behind_subjects", "cache behind"),
                ("grading_failed_subjects", "grading errors"),
            ):
                subjects = [str(s) for s in latest_daily_payload.get(key, [])][:8]
                if subjects:
                    blocked_parts.append(f"{label}: {', '.join(subjects)}")
            if blocked_parts:
                issues.append("scorecard data blocked — " + "; ".join(blocked_parts))

        # A fresh historian journal row with ``ok=false`` is not liveness;
        # it is an explicit failed permanent-memory review and deserves an
        # alert rather than being hidden by the timestamp alone.
        try:
            latest_hist = conn.execute(
                "SELECT payload FROM journal WHERE kind = 'historian' ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if latest_hist:
                payload = json.loads(latest_hist["payload"])
                if payload.get("ok") is False:
                    issues.append(
                        "historian distillation failed: "
                        + str(payload.get("reason", "unknown"))[:160]
                    )
        except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
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
        # every five minutes on a 1-vCPU box.
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


def _parse_aware_timestamp(value: object) -> datetime | None:
    """Read a timestamp from state without silently treating naive time as UTC."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(timezone.utc)


def check_broker_liveness(state_dir: Path, *, now: datetime | None = None) -> list[str]:
    """Return authenticated-IBKR-session issues, if this runner supports it.

    ``broker_liveness.json`` is intentionally optional: simulator and
    third-party broker deployments do not expose a wire-level probe. Once an
    IBKR-capable runner writes it, however, its boolean result is meaningful
    immediately and its last successful request is a stronger freshness
    signal than a file mtime or cached account snapshot.
    """
    from trading.runtime.broker_liveness import FILENAME

    path = Path(state_dir) / FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return [f"broker API liveness: unreadable ({FILENAME})"]
    if not isinstance(payload, dict) or not isinstance(payload.get("ready"), bool):
        return [f"broker API liveness: invalid ({FILENAME})"]

    now = now or datetime.now(tz=timezone.utc)
    last_success = _parse_aware_timestamp(payload.get("last_success_at"))
    if payload["ready"] is False:
        detail = str(payload.get("detail", "unknown failure"))[:180]
        if last_success is None:
            success_text = "no authenticated response recorded"
        else:
            age_s = max(0.0, (now - last_success).total_seconds())
            success_text = f"last authenticated response {age_s / 60.0:.1f}m ago"
        return [f"broker API liveness: unavailable ({detail}; {success_text})"]

    if last_success is None:
        return ["broker API liveness: no authenticated response timestamp"]
    age_s = max(0.0, (now - last_success).total_seconds())
    if age_s > BROKER_LIVENESS_MAX_AGE_SECONDS:
        return [
            "broker API liveness: last authenticated response "
            f"{age_s / 60.0:.1f}m ago (limit {BROKER_LIVENESS_MAX_AGE_SECONDS / 60.0:.0f}m)"
        ]
    return []


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
        # WAL-aware: runner.db's own mtime is the last checkpoint, not
        # the last write. See core.clock.artifact_age_seconds.
        _secs = artifact_age_seconds(p, now=now.timestamp())
        if _secs is None:
            continue
        age_h = _secs / 3600.0
        if age_h > max_h:
            issues.append(f"{label}: stale ({age_h:.0f}h old, limit {max_h:.0f}h)")

    issues.extend(check_broker_liveness(state_dir, now=now))

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
    new_alerts: list[tuple[str, str]] = []
    for key, text in current.items():
        last = reported.get(key)
        if last:
            try:
                age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600
                if age_h < DEBOUNCE_HOURS:
                    continue
            except Exception:
                pass
        new_alerts.append((key, text))

    recovered = [k for k in list(reported) if k not in current]
    alerted: list[str] = []
    recovered_reported: list[str] = []

    if new_alerts:
        message = "⚠️ Ops watchdog\n" + "\n".join(f"• {text}" for _, text in new_alerts)
        if _send_ops(message):
            for key, _ in new_alerts:
                reported[key] = now.isoformat()
            alerted = [text for _, text in new_alerts]
    # Recovery is independently useful information. Suppressing it merely
    # because a different issue appeared in the same pass leaves an outage
    # looking active long after its authenticated probe has recovered.
    if recovered and _send_ops("✅ Ops watchdog: recovered — " + ", ".join(recovered)):
        for key in recovered:
            del reported[key]
        recovered_reported = recovered

    _save(state_dir, {"reported": reported, "last_run": now.isoformat()})
    if alerted:
        logger.bind(component="ops_watch").warning(f"issues: {alerted}")
    return {"issues": issues, "alerted": alerted, "recovered": recovered_reported}
