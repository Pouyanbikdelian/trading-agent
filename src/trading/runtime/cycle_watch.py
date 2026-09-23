r"""Watch what the desk achieved, not whether the process is alive.

Until 2026-09-23 every health signal measured liveness. ``snapshot_refresh``
rewrote ``heartbeat.json`` every 60 s, so the runner's watchdog, ``/heartbeat``
and the Docker healthcheck all answered "is the broker talking?". Nothing
answered "did the Friday cycle run, and could it trade?". The desk then sat
in halted reviews and no-order cycles from mid-August to late September with
every light green — a subsystem that reports attempts, not achievements.

Three questions, each answered from ``runner.db``'s ``cycles`` table, which
every cycle (scheduled, ``/cycle``, ``/review``) writes exactly once:

* **Missed cycle.** A scheduled fire time passed (plus a grace period) and
  no cycle row exists after it. The scheduler, the container or the VPS was
  down at the one moment of the week that matters.
* **Stuck desk.** The last N cycles all ended in a status that cannot trade
  (halted, halted review, error, lock contention). A desk that is
  structurally unable to trade is an incident even when nothing crashes.
* **No trades.** The last N cycles all submitted zero orders, whatever the
  reason. Weaker than "stuck" — a momentum book can legitimately hold — so
  it is a warning with a larger N.

Each finding carries a dedupe key persisted in ``cycle_watch.json``, so an
hourly watchdog alerts once per incident rather than once per hour. A
streak is keyed on the last HEALTHY cycle before it (its anchor), which
does not move as the streak grows; keying on the streak's oldest visible
row re-alerted every cycle once the streak outgrew the read window. A
stuck desk implies no trades, so the weaker warning is suppressed then.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

STATE_FILE = "cycle_watch.json"

#: Cycle statuses that cannot have traded. ``no_orders`` is deliberately
#: absent: an executable cycle that decided nothing needs changing is healthy.
STUCK_STATUSES = frozenset({"halted", "halted_review", "error", "skipped_locked"})

DEFAULT_GRACE = timedelta(hours=2)
DEFAULT_STUCK_STREAK = 2
DEFAULT_NO_TRADE_STREAK = 4
_MAX_REMEMBERED_KEYS = 64


@dataclass(frozen=True)
class CycleWatchFinding:
    key: str
    level: Literal["warning", "critical"]
    message: str


def last_scheduled_fire(
    cron: str, tz: str, before: datetime, *, horizon_days: int = 8
) -> datetime | None:
    """Latest fire time of ``cron`` (in ``tz``) at or before ``before``.

    Uses APScheduler's own trigger arithmetic so this can never disagree
    with the scheduler about when a cycle was due, DST included.
    """
    from apscheduler.triggers.cron import CronTrigger

    if before.tzinfo is None:
        raise ValueError("before must be timezone-aware")
    trigger = CronTrigger.from_crontab(cron, timezone=tz)
    cursor = before - timedelta(days=horizon_days)
    fire = trigger.get_next_fire_time(None, cursor)
    latest: datetime | None = None
    for _ in range(10_000):  # bounded: a pathological cron must not hang the watchdog
        if fire is None or fire > before:
            break
        latest = fire
        fire = trigger.get_next_fire_time(fire, fire + timedelta(seconds=1))
    return latest


def _ts(row: Mapping[str, object]) -> datetime:
    ts = row["ts"]
    if not isinstance(ts, datetime) or ts.tzinfo is None:
        raise ValueError("cycle rows must carry a timezone-aware 'ts'")
    return ts


def check_missed_cycle(
    cycles: Sequence[Mapping[str, object]],
    *,
    cron: str,
    tz: str,
    now: datetime,
    grace: timedelta = DEFAULT_GRACE,
) -> CycleWatchFinding | None:
    fire = last_scheduled_fire(cron, tz, now - grace)
    if fire is None:
        return None
    # Five minutes of slack: the cycle row is stamped at cycle start, and
    # the scheduler may fire a hair early relative to the recorded clock.
    if any(_ts(c) >= fire - timedelta(minutes=5) for c in cycles):
        return None
    return CycleWatchFinding(
        key=f"missed:{fire.isoformat()}",
        level="critical",
        message=(
            f"⏰ *Missed cycle.* The cycle scheduled for {fire:%a %Y-%m-%d %H:%M %Z} "
            f"has no record {grace.total_seconds() / 3600:.0f}h later: the runner "
            "was down, the scheduler did not fire, or the cycle died before it "
            "could record itself. Check `/health`; `/cycle` runs one now if "
            "appropriate."
        ),
    )


def _streak(
    rows: Sequence[Mapping[str, object]], pred: Callable[[Mapping[str, object]], bool]
) -> tuple[list[Mapping[str, object]], str]:
    """The newest-first run matching ``pred``, and a stable incident anchor.

    The anchor is the first row that breaks the run (the last healthy
    cycle). If the run fills the whole window there is none in view; the
    constant "window" is used, which changes the key at most once.
    """
    out: list[Mapping[str, object]] = []
    anchor = "window"
    for row in rows:  # newest first
        if not pred(row):
            anchor = _ts(row).isoformat()
            break
        out.append(row)
    return out, anchor


def check_stuck_desk(
    cycles: Sequence[Mapping[str, object]], *, streak: int = DEFAULT_STUCK_STREAK
) -> CycleWatchFinding | None:
    run, anchor = _streak(cycles, lambda c: str(c.get("status")) in STUCK_STATUSES)
    if len(run) < streak:
        return None
    first = run[-1]
    statuses = ", ".join(sorted({str(c.get("status")) for c in run}))
    return CycleWatchFinding(
        key=f"stuck:{anchor}",
        level="critical",
        message=(
            f"🧱 *Desk cannot trade.* The last {len(run)} cycles all ended "
            f"without the ability to execute ({statuses}), since "
            f"{_ts(first):%Y-%m-%d %H:%M UTC}. Check `/status` and `/baseline`."
        ),
    )


def _orders(row: Mapping[str, object]) -> int:
    raw = row.get("orders_submitted")
    return int(raw) if isinstance(raw, (int, float)) else 0


def check_no_trades(
    cycles: Sequence[Mapping[str, object]], *, streak: int = DEFAULT_NO_TRADE_STREAK
) -> CycleWatchFinding | None:
    run, anchor = _streak(cycles, lambda c: _orders(c) == 0)
    if len(run) < streak:
        return None
    first = run[-1]
    return CycleWatchFinding(
        key=f"notrade:{anchor}",
        level="warning",
        message=(
            f"💤 *No trades in {len(run)} cycles* (since {_ts(first):%Y-%m-%d}). "
            "This can be correct for a holding book; confirm it is a decision, "
            "not a blockage."
        ),
    )


def evaluate(
    cycles: Sequence[Mapping[str, object]],
    *,
    cron: str,
    tz: str,
    now: datetime,
    grace: timedelta = DEFAULT_GRACE,
    stuck_streak: int = DEFAULT_STUCK_STREAK,
    no_trade_streak: int = DEFAULT_NO_TRADE_STREAK,
) -> list[CycleWatchFinding]:
    """All current findings, most severe first. ``cycles`` is newest first."""
    ordered = sorted(cycles, key=_ts, reverse=True)
    stuck = check_stuck_desk(ordered, streak=stuck_streak)
    found = [
        check_missed_cycle(ordered, cron=cron, tz=tz, now=now, grace=grace),
        stuck,
        None if stuck is not None else check_no_trades(ordered, streak=no_trade_streak),
    ]
    return [f for f in found if f is not None]


def _load_keys(path: Path) -> list[str]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    keys = raw.get("alerted") if isinstance(raw, dict) else None
    return [str(k) for k in keys] if isinstance(keys, list) else []


def unalerted(state_dir: Path, findings: Sequence[CycleWatchFinding]) -> list[CycleWatchFinding]:
    """Findings not yet alerted; records them as alerted (atomic write).

    A corrupt or missing state file re-alerts rather than suppresses:
    one duplicate message is cheaper than a silenced incident.
    """
    path = Path(state_dir) / STATE_FILE
    seen = _load_keys(path)
    fresh = [f for f in findings if f.key not in seen]
    if fresh:
        keys = (seen + [f.key for f in fresh])[-_MAX_REMEMBERED_KEYS:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"alerted": keys}, indent=2))
        os.replace(tmp, path)
    return fresh
