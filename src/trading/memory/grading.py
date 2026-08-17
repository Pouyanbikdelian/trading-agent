"""Prediction grading — extracted from the runner so it can be tested.

This lived as a loop inside ``Runner._run_memory_grader_async``, an async
method needing a whole Runner to exercise. No test ever covered it, and
it was broken from the day it was written: a tz-aware/naive comparison
raised on the first row of every nightly pass, inside one broad
``except``. No prediction was ever graded; agent calibration was
permanently empty; the entire "weight the voices by track record" design
was reading an empty table.

The bug was trivial. What let it survive was that the WIRING had no
seam — helpers were testable, the loop was not, and the loop was where
the bug lived. So the loop is a plain function now, and the runner is a
four-line caller.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def grade_due_predictions(
    mem: Any,
    data_dir: Path,
    *,
    asof: datetime | None = None,
) -> dict[str, Any]:
    """Grade every matured prediction and expose why any row was blocked.

    Grades at the DUE date, not at the newest available bar: scoring a
    14-day call over however many days happen to have passed measures a
    different prediction from the one the agent made.

    Per-row isolation, because one unpriceable symbol is not a reason to
    stop scoring the other seven agents — the original single try block
    around the whole batch is what turned one TypeError into a total
    outage of the scorecard.

    ``asof`` makes the cache-timing decision reproducible and tells
    session-aware coverage when a missing next-session bar is genuinely
    stale rather than a normal weekend wait.
    """
    from trading.runtime.portfolio_stats import (
        _read_close,
        completed_session_close,
        coverage_status,
    )

    now = asof or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        raise ValueError("scorecard grading asof must be timezone-aware")
    graded = skipped = 0
    unpriced_subjects: set[str] = set()
    awaiting_next_daily_bar_subjects: set[str] = set()
    awaiting_next_daily_bar_prediction_ids: set[str] = set()
    cache_behind_subjects: set[str] = set()
    failed_subjects: set[str] = set()
    for row in mem.due_predictions(asof=now):
        subject = str(row["subject"]).upper()
        try:
            series = _read_close(data_dir, subject)
            if series is None or len(series) < 5:
                skipped += 1
                unpriced_subjects.add(subject)
                continue
            made = datetime.fromtimestamp(row["ts"], tz=timezone.utc)
            due = datetime.fromtimestamp(row["due_ts"], tz=timezone.utc)
            # Validate the entry endpoint before classifying a weekend or
            # holiday expiry as a normal pending session. Otherwise an old
            # prediction outside cache history could receive an exact-ID
            # watchdog exemption despite already being ungradeable.
            base = completed_session_close(series, made)
            if not base:
                skipped += 1
                cache_behind_subjects.add(subject)
                continue
            status = coverage_status(series, due, asof=now)
            if status != "covered":
                skipped += 1  # not matured in our data yet; retry tomorrow
                if status == "awaiting_next_daily_bar":
                    awaiting_next_daily_bar_subjects.add(subject)
                    awaiting_next_daily_bar_prediction_ids.add(str(row["id"]))
                else:
                    cache_behind_subjects.add(subject)
                continue
            # Predictions can be created or expire mid-session.  The cache's
            # midnight date label is not the session close, so selecting by
            # timestamp alone would retrospectively use that day's eventual
            # close.  Score only completed NYSE sessions at both endpoints.
            end = completed_session_close(series, due)
            if end is None:
                skipped += 1
                cache_behind_subjects.add(subject)
                continue
            mem.grade_prediction(row["id"], end / base - 1.0)
            graded += 1
        except Exception:
            skipped += 1
            failed_subjects.add(subject)
            logger.bind(component="memory", subject=subject).exception(
                "grading one prediction failed"
            )
    if skipped:
        logger.bind(component="memory").warning(
            f"{skipped} prediction(s) not graded this pass (unpriced or unmatured)"
        )
    if graded:
        logger.bind(component="memory").info(f"graded {graded} due prediction(s)")
    if (
        unpriced_subjects
        or awaiting_next_daily_bar_subjects
        or cache_behind_subjects
        or failed_subjects
    ):
        logger.bind(component="memory").warning(
            "scorecard status: "
            f"unpriced={','.join(sorted(unpriced_subjects)) or '-'} "
            f"awaiting_next_daily_bar={','.join(sorted(awaiting_next_daily_bar_subjects)) or '-'} "
            f"cache_behind={','.join(sorted(cache_behind_subjects)) or '-'} "
            f"failed={','.join(sorted(failed_subjects)) or '-'}"
        )
    return {
        "graded": graded,
        "skipped": skipped,
        "unpriced_subjects": sorted(unpriced_subjects),
        "awaiting_next_daily_bar_subjects": sorted(awaiting_next_daily_bar_subjects),
        "awaiting_next_daily_bar_prediction_ids": sorted(awaiting_next_daily_bar_prediction_ids),
        "cache_behind_subjects": sorted(cache_behind_subjects),
        "failed_subjects": sorted(failed_subjects),
    }
