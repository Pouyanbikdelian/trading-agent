"""The grading loop itself — the thing that had no test and no seam.

`close_at` and `covers` are unit-tested elsewhere. That is not the same
as testing that grading WORKS: the bug lived in the loop, and the loop
was an async Runner method nobody could exercise. These tests drive the
real `MemoryStore`, real parquet files and the real loop end to end, so
"is calibration populated afterwards" is an assertion rather than a hope.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.core.types import AssetClass, Instrument
from trading.data.cache import ParquetCache
from trading.memory.grading import grade_due_predictions
from trading.memory.store import MemoryStore

NOW = datetime.now(tz=timezone.utc)


def _write_prices(
    data_dir: Path,
    symbol: str,
    *,
    bars: int = 400,
    drift: float = 0.001,
    end: datetime | None = None,
) -> None:
    """Bars ending TODAY, tz-aware UTC — exactly the shape the live cache
    has, which is what made the naive comparison raise in production."""
    idx = pd.date_range(end=(end or NOW).date(), periods=bars, freq="B", tz="UTC")
    # `end` on a weekend makes pandas return one bar fewer than asked for,
    # so every column has to be sized from the index rather than from
    # `bars`. Without this the whole file errors out on Saturdays.
    close = 100.0 * np.exp(np.arange(len(idx)) * drift)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(len(idx), 1e6),
            "adj_close": close,
        },
        index=idx,
    )
    df.index.name = "ts"
    ParquetCache(data_dir).write(Instrument(symbol=symbol, asset_class=AssetClass.EQUITY), "1D", df)


def _write_daily_closes(data_dir: Path, symbol: str, closes: dict[str, float]) -> None:
    """Write a deliberately shaped daily cache for endpoint-timing tests."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(closes), utc=True))
    close = np.array(list(closes.values()), dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 1e6),
            "adj_close": close,
        },
        index=idx,
    )
    df.index.name = "ts"
    ParquetCache(data_dir).write(Instrument(symbol=symbol, asset_class=AssetClass.EQUITY), "1D", df)


@pytest.fixture
def desk(tmp_path: Path):
    data = tmp_path / "parquet"
    _write_prices(data, "AAPL")
    return MemoryStore(tmp_path / "memory"), data


def _predict(mem: MemoryStore, *, agent: str, horizon: int, days_ago: int, subject="AAPL") -> str:
    pid = mem.add_prediction(
        agent=agent,
        subject=subject,
        direction="up",
        horizon_days=horizon,
        confidence=0.7,
        statement="test",
    )
    # Backdate so the prediction is genuinely due — add_prediction stamps
    # "now", and a prediction made now is never due now.
    made = (NOW - timedelta(days=days_ago)).timestamp()
    mem.conn.execute(
        "UPDATE predictions SET ts = ?, due_ts = ? WHERE id = ?",
        (made, made + horizon * 86400, pid),
    )
    mem.conn.commit()
    return pid


class TestGradingActuallyGrades:
    def test_a_matured_prediction_is_graded(self, desk) -> None:
        """The assertion that would have caught the outage on day one."""
        mem, data = desk
        _predict(mem, agent="quant", horizon=14, days_ago=40)
        out = grade_due_predictions(mem, data)
        assert out["graded"] == 1 and out["skipped"] == 0
        assert out["unpriced_subjects"] == []
        assert mem.due_predictions() == []

    def test_calibration_is_populated_afterwards(self, desk) -> None:
        """Calibration being empty is the symptom the operator would see;
        assert on the symptom, not just the mechanism."""
        mem, data = desk
        for i in range(4):
            _predict(mem, agent="quant", horizon=7, days_ago=30 + i)
        assert mem.calibration() == []  # nothing graded yet
        grade_due_predictions(mem, data)
        cal = mem.calibration()
        assert len(cal) == 1 and cal[0]["agent"] == "quant" and cal[0]["n"] == 4

    def test_an_unmatured_prediction_waits(self, desk) -> None:
        """Not yet due in OUR data — must be retried, not scored early."""
        mem, data = desk
        _predict(mem, agent="scout", horizon=400, days_ago=1)
        out = grade_due_predictions(mem, data)
        assert out["graded"] == 0

    def test_same_day_expiry_waits_for_next_daily_bar_not_a_cache_repair(self, desk) -> None:
        mem, data = desk
        _predict(mem, agent="scout", horizon=1, days_ago=1, subject="AAPL")
        out = grade_due_predictions(mem, data)
        assert out["graded"] == 0 and out["skipped"] == 1
        assert out["awaiting_next_daily_bar_subjects"] == ["AAPL"]
        assert out["cache_behind_subjects"] == []

    def test_weekend_expiry_is_pending_with_exact_ids_for_the_watchdog(self, desk) -> None:
        """Friday's valid bar must not look stale merely because the due
        timestamp lands on Sunday; ops needs the exact pending row to avoid
        suppressing an unrelated genuinely overdue prediction."""
        mem, data = desk
        _write_prices(data, "AAPL", end=datetime(2026, 8, 14, tzinfo=timezone.utc))
        pid = _predict(mem, agent="scout", horizon=14, days_ago=30, subject="AAPL")
        made = datetime(2026, 8, 2, 12, tzinfo=timezone.utc)
        due = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        mem.conn.execute(
            "UPDATE predictions SET ts = ?, due_ts = ? WHERE id = ?",
            (made.timestamp(), due.timestamp(), pid),
        )
        mem.conn.commit()

        out = grade_due_predictions(mem, data, asof=datetime(2026, 8, 16, 18, tzinfo=timezone.utc))

        assert out["graded"] == 0 and out["skipped"] == 1
        assert out["awaiting_next_daily_bar_subjects"] == ["AAPL"]
        assert out["awaiting_next_daily_bar_prediction_ids"] == [pid]
        assert out["cache_behind_subjects"] == []

    def test_weekend_expiry_grades_once_mondays_bar_arrives(self, desk) -> None:
        mem, data = desk
        _write_prices(data, "AAPL", end=datetime(2026, 8, 17, tzinfo=timezone.utc))
        pid = _predict(mem, agent="scout", horizon=14, days_ago=30, subject="AAPL")
        made = datetime(2026, 8, 2, 12, tzinfo=timezone.utc)
        due = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        mem.conn.execute(
            "UPDATE predictions SET ts = ?, due_ts = ? WHERE id = ?",
            (made.timestamp(), due.timestamp(), pid),
        )
        mem.conn.commit()

        out = grade_due_predictions(mem, data, asof=datetime(2026, 8, 17, 23, tzinfo=timezone.utc))

        assert out["graded"] == 1 and out["skipped"] == 0
        assert out["awaiting_next_daily_bar_prediction_ids"] == []
        assert mem.due_predictions(asof=datetime(2026, 8, 17, 23, tzinfo=timezone.utc)) == []

    def test_missing_entry_close_is_not_mislabeled_as_session_pending(self, desk) -> None:
        """Only a row that can eventually grade earns an awaiting-ID exemption."""
        mem, data = desk
        _write_prices(data, "AAPL", end=datetime(2026, 8, 14, tzinfo=timezone.utc))
        pid = _predict(mem, agent="scout", horizon=14, days_ago=30, subject="AAPL")
        made = datetime(2020, 8, 2, 12, tzinfo=timezone.utc)
        due = datetime(2026, 8, 16, 12, tzinfo=timezone.utc)
        mem.conn.execute(
            "UPDATE predictions SET ts = ?, due_ts = ? WHERE id = ?",
            (made.timestamp(), due.timestamp(), pid),
        )
        mem.conn.commit()

        out = grade_due_predictions(mem, data, asof=datetime(2026, 8, 16, 18, tzinfo=timezone.utc))

        assert out["graded"] == 0 and out["skipped"] == 1
        assert out["awaiting_next_daily_bar_prediction_ids"] == []
        assert out["cache_behind_subjects"] == ["AAPL"]

    def test_mid_session_endpoints_exclude_the_same_sessions_future_close(self, desk) -> None:
        """A later cache read must not rewrite a 10:00 ET forecast.

        The deliberately huge Monday close demonstrates the old failure:
        midnight-label slicing would have scored Friday-to-Monday instead of
        Thursday-to-Friday. Tuesday merely proves the Monday horizon bar is
        final and allows the conservative coverage gate to release it.
        """
        mem, data = desk
        _write_daily_closes(
            data,
            "AAPL",
            {
                "2026-01-07": 90.0,
                "2026-01-08": 100.0,
                "2026-01-09": 120.0,
                "2026-01-12": 1_000.0,
                "2026-01-13": 130.0,
            },
        )
        pid = _predict(mem, agent="quant", horizon=3, days_ago=30, subject="AAPL")
        made = datetime(2026, 1, 9, 15, tzinfo=timezone.utc)  # Friday 10:00 ET
        due = datetime(2026, 1, 12, 15, tzinfo=timezone.utc)  # Monday 10:00 ET
        mem.conn.execute(
            "UPDATE predictions SET ts = ?, due_ts = ? WHERE id = ?",
            (made.timestamp(), due.timestamp(), pid),
        )
        mem.conn.commit()

        out = grade_due_predictions(mem, data, asof=datetime(2026, 1, 13, 23, tzinfo=timezone.utc))
        row = mem.conn.execute(
            "SELECT realized_move FROM predictions WHERE id = ?", (pid,)
        ).fetchone()

        assert out["graded"] == 1 and out["skipped"] == 0
        assert row["realized_move"] == pytest.approx(0.20)

    def test_an_unpriceable_symbol_does_not_stop_the_batch(self, desk) -> None:
        """One bad symbol used to kill every remaining prediction AND the
        shadow grading below it."""
        mem, data = desk
        _predict(mem, agent="quant", horizon=14, days_ago=40, subject="NOSUCHTICKER")
        _predict(mem, agent="scout", horizon=14, days_ago=40, subject="AAPL")
        out = grade_due_predictions(mem, data)
        assert out["graded"] == 1 and out["skipped"] == 1
        assert out["unpriced_subjects"] == ["NOSUCHTICKER"]
        assert [c["agent"] for c in mem.calibration()] == ["scout"]

    def test_horizon_is_respected_not_time_since(self, desk) -> None:
        """A 7-day call graded 90 days late must still be scored over 7
        days. The old code used the newest bar, scoring 90 days of drift
        as though the agent had predicted it."""
        mem, data = desk
        pid = _predict(mem, agent="quant", horizon=7, days_ago=90)
        grade_due_predictions(mem, data)
        row = mem.conn.execute(
            "SELECT realized_move FROM predictions WHERE id = ?", (pid,)
        ).fetchone()
        # ~5 business days of +0.1%/bar compounding, nowhere near 90 days'.
        assert row["realized_move"] == pytest.approx(0.005, abs=0.004)

    def test_grading_is_not_repeated(self, desk) -> None:
        mem, data = desk
        _predict(mem, agent="quant", horizon=14, days_ago=40)
        grade_due_predictions(mem, data)
        out = grade_due_predictions(mem, data)
        assert out["graded"] == 0 and out["skipped"] == 0
        assert out["unpriced_subjects"] == []

    def test_empty_book_is_quiet(self, desk) -> None:
        mem, data = desk
        out = grade_due_predictions(mem, data)
        assert out["graded"] == 0 and out["skipped"] == 0
        assert out["unpriced_subjects"] == []
