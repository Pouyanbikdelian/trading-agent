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


def grade_due_predictions(mem: Any, data_dir: Path) -> dict[str, int]:
    """Grade every matured prediction. Returns ``{graded, skipped}``.

    Grades at the DUE date, not at the newest available bar: scoring a
    14-day call over however many days happen to have passed measures a
    different prediction from the one the agent made.

    Per-row isolation, because one unpriceable symbol is not a reason to
    stop scoring the other seven agents — the original single try block
    around the whole batch is what turned one TypeError into a total
    outage of the scorecard.
    """
    from trading.runtime.portfolio_stats import _read_close, close_at, covers

    graded = skipped = 0
    for row in mem.due_predictions():
        try:
            series = _read_close(data_dir, row["subject"])
            if series is None or len(series) < 5:
                skipped += 1
                continue
            made = datetime.fromtimestamp(row["ts"], tz=timezone.utc)
            due = datetime.fromtimestamp(row["due_ts"], tz=timezone.utc)
            if not covers(series, due):
                skipped += 1  # not matured in our data yet; retry tomorrow
                continue
            base = close_at(series, made)
            end = close_at(series, due)
            if not base or end is None:
                skipped += 1
                continue
            mem.grade_prediction(row["id"], end / base - 1.0)
            graded += 1
        except Exception:
            skipped += 1
            logger.bind(component="memory", subject=row.get("subject")).exception(
                "grading one prediction failed"
            )
    if skipped:
        logger.bind(component="memory").warning(
            f"{skipped} prediction(s) not graded this pass (unpriced or unmatured)"
        )
    if graded:
        logger.bind(component="memory").info(f"graded {graded} due prediction(s)")
    return {"graded": graded, "skipped": skipped}
