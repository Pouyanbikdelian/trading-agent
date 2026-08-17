"""tz-safe price lookup — the helper that unbroke the scorecard.

Found 2026-08-06. The nightly memory grader compared a tz-AWARE cache
index against a tz-NAIVE datetime::

    hist = s[s.index <= ts0.replace(tzinfo=None)]

The parquet index is ``datetime64[ms, UTC]``, so pandas raised
``TypeError`` on the FIRST prediction of every pass. The loop sat inside
one broad ``except Exception``, so the grader logged once a night and
graded nothing — for as long as it had existed. Consequences: no
prediction ever scored, ``calibration()`` permanently empty, per-agent
skill weighting reasoning over an empty table, source trust never
updated, and the shadow ledger never filled a single leg.

No test covered the grader, which is exactly why it could run broken
indefinitely. These are those tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading.runtime.portfolio_stats import (
    _read_close,
    cache_symbol_for_subject,
    close_at,
    completed_session_close,
    coverage_status,
    covers,
    nyse_session_settled_since,
)


def _series(tz: str | None, days: int = 30) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=days, freq="D", tz=tz)
    return pd.Series(range(100, 100 + days), index=idx, dtype="float64")


def _nyse_daily_series(last: str) -> pd.Series:
    """A realistic daily cache through one US trading-session label."""
    idx = pd.date_range("2026-08-10", last, freq="B", tz="UTC")
    return pd.Series(range(100, 100 + len(idx)), index=idx, dtype="float64")


class TestCloseAt:
    def test_aware_index_with_aware_timestamp(self) -> None:
        s = _series("UTC")
        assert close_at(s, datetime(2026, 1, 10, tzinfo=timezone.utc)) == 109.0

    def test_aware_index_with_naive_timestamp(self) -> None:
        """The exact combination that raised in production."""
        s = _series("UTC")
        assert close_at(s, datetime(2026, 1, 10)) == 109.0

    def test_naive_index_with_aware_timestamp(self) -> None:
        """And the mirror image, so a naive cache cannot break it either."""
        s = _series(None)
        assert close_at(s, datetime(2026, 1, 10, tzinfo=timezone.utc)) == 109.0

    def test_returns_last_bar_at_or_before(self) -> None:
        s = _series("UTC")
        # A weekend/holiday date must fall back to the prior bar, not fail.
        assert close_at(s, datetime(2026, 1, 10, 23, 59, tzinfo=timezone.utc)) == 109.0

    def test_before_history_is_none_not_zero(self) -> None:
        """None means 'cannot grade'; zero would score a real return."""
        s = _series("UTC")
        assert close_at(s, datetime(2020, 1, 1, tzinfo=timezone.utc)) is None

    def test_empty_and_missing_series(self) -> None:
        assert close_at(None, datetime(2026, 1, 10)) is None
        assert close_at(pd.Series(dtype="float64"), datetime(2026, 1, 10)) is None


class TestCompletedSessionClose:
    def test_mid_session_timestamp_uses_the_prior_completed_close(self) -> None:
        """The midnight label for Friday must not leak Friday's later close
        into a forecast made at 10:00 ET."""
        s = pd.Series(
            [100.0, 110.0],
            index=pd.DatetimeIndex(["2026-01-08", "2026-01-09"], tz="UTC"),
        )

        assert completed_session_close(s, datetime(2026, 1, 9, 15, tzinfo=timezone.utc)) == 100.0

    def test_after_close_timestamp_can_use_that_sessions_close(self) -> None:
        s = pd.Series(
            [100.0, 110.0],
            index=pd.DatetimeIndex(["2026-01-08", "2026-01-09"], tz="UTC"),
        )

        assert completed_session_close(s, datetime(2026, 1, 9, 22, tzinfo=timezone.utc)) == 110.0


class TestCovers:
    def test_true_when_the_cache_reaches_the_date(self) -> None:
        s = _series("UTC")
        assert covers(s, datetime(2026, 1, 20, tzinfo=timezone.utc)) is True

    def test_false_when_the_horizon_has_not_elapsed_in_our_data(self) -> None:
        """Grading a 14-day call against the newest available bar is not
        grading it over 14 days — it scores a different prediction from
        the one the agent made."""
        s = _series("UTC", days=10)
        assert covers(s, datetime(2026, 1, 20, tzinfo=timezone.utc)) is False

    def test_naive_and_aware_both_work(self) -> None:
        assert covers(_series("UTC"), datetime(2026, 1, 5)) is True
        assert covers(_series(None), datetime(2026, 1, 5, tzinfo=timezone.utc)) is True

    def test_empty_never_covers(self) -> None:
        assert covers(None, datetime(2026, 1, 5)) is False

    def test_same_day_daily_bar_waits_without_claiming_the_cache_is_stale(self) -> None:
        s = _series("UTC", days=20)
        due_after_close_label = datetime(2026, 1, 20, 17, tzinfo=timezone.utc)
        assert coverage_status(s, due_after_close_label) == "awaiting_next_daily_bar"
        assert covers(s, due_after_close_label) is False


class TestDailySessionCoverage:
    @pytest.mark.parametrize(
        "due",
        [
            datetime(2026, 8, 14, 22, tzinfo=timezone.utc),  # Friday after NYSE close
            datetime(2026, 8, 15, 12, tzinfo=timezone.utc),  # Saturday
            datetime(2026, 8, 16, 12, tzinfo=timezone.utc),  # Sunday
        ],
    )
    def test_friday_close_is_a_normal_wait_through_the_weekend(self, due: datetime) -> None:
        friday_cache = _nyse_daily_series("2026-08-14")

        assert (
            coverage_status(
                friday_cache,
                due,
                asof=datetime(2026, 8, 16, 18, tzinfo=timezone.utc),
            )
            == "awaiting_next_daily_bar"
        )

    @pytest.mark.parametrize(
        "due",
        [
            datetime(2026, 8, 14, 22, tzinfo=timezone.utc),
            datetime(2026, 8, 15, 12, tzinfo=timezone.utc),
            datetime(2026, 8, 16, 12, tzinfo=timezone.utc),
        ],
    )
    def test_due_is_covered_by_the_following_monday_bar(self, due: datetime) -> None:
        monday_cache = _nyse_daily_series("2026-08-17")

        assert (
            coverage_status(
                monday_cache,
                due,
                asof=datetime(2026, 8, 17, 23, tzinfo=timezone.utc),
            )
            == "covered"
        )

    def test_missing_monday_bar_is_stale_after_the_post_close_cache_window(self) -> None:
        friday_cache = _nyse_daily_series("2026-08-14")

        assert (
            coverage_status(
                friday_cache,
                datetime(2026, 8, 16, 12, tzinfo=timezone.utc),
                asof=datetime(2026, 8, 17, 23, tzinfo=timezone.utc),
            )
            == "cache_behind"
        )

    def test_missing_due_session_is_not_hidden_by_a_later_cache_bar(self) -> None:
        """A file can have a hole: Monday is not proof Friday was cached.

        If Friday is absent but Monday exists, ``close_at`` would otherwise
        fall back to Thursday and silently grade the wrong interval.
        """
        gap_cache = pd.Series(
            [100.0, 110.0],
            index=pd.DatetimeIndex(["2026-08-13", "2026-08-17"], tz="UTC"),
        )

        assert (
            coverage_status(
                gap_cache,
                datetime(2026, 8, 16, 12, tzinfo=timezone.utc),
                asof=datetime(2026, 8, 17, 23, tzinfo=timezone.utc),
            )
            == "cache_behind"
        )
        assert covers(gap_cache, datetime(2026, 8, 16, 12, tzinfo=timezone.utc)) is False

    def test_standard_time_grader_detects_missing_monday_bar_that_night(self) -> None:
        """The NY-local evening grader must not wait until Tuesday in winter."""
        friday_cache = pd.Series(
            range(5),
            index=pd.date_range("2026-01-05", "2026-01-09", freq="B", tz="UTC"),
            dtype="float64",
        )

        assert (
            coverage_status(
                friday_cache,
                datetime(2026, 1, 11, 12, tzinfo=timezone.utc),
                asof=datetime(2026, 1, 12, 23, 45, tzinfo=timezone.utc),
            )
            == "cache_behind"
        )

    def test_missing_friday_bar_is_not_hidden_by_the_weekend(self) -> None:
        thursday_cache = _nyse_daily_series("2026-08-13")

        assert (
            coverage_status(
                thursday_cache,
                datetime(2026, 8, 16, 12, tzinfo=timezone.utc),
                asof=datetime(2026, 8, 16, 18, tzinfo=timezone.utc),
            )
            == "cache_behind"
        )

    def test_market_holiday_is_also_a_normal_wait(self) -> None:
        # Labor Day 2026 is Monday, September 7.  A weekday-only heuristic
        # would falsely ask for a bar that cannot exist.
        friday_cache = _nyse_daily_series("2026-09-04")

        assert (
            coverage_status(
                friday_cache,
                datetime(2026, 9, 7, 22, tzinfo=timezone.utc),
                asof=datetime(2026, 9, 7, 23, tzinfo=timezone.utc),
            )
            == "awaiting_next_daily_bar"
        )

    def test_early_close_waits_for_the_regular_cache_and_grader_slots(self) -> None:
        """A 13:00 ET close must not pre-empt jobs scheduled for that evening."""
        before_early_close = pd.Series(
            [100.0, 101.0],
            index=pd.DatetimeIndex(["2026-11-24", "2026-11-25"], tz="UTC"),
        )
        thanksgiving_due = datetime(2026, 11, 26, 22, tzinfo=timezone.utc)
        journal_at = datetime(2026, 11, 26, 23, tzinfo=timezone.utc)

        assert (
            coverage_status(
                before_early_close,
                thanksgiving_due,
                asof=datetime(2026, 11, 27, 22, tzinfo=timezone.utc),  # 17:00 ET
            )
            == "awaiting_next_daily_bar"
        )
        assert (
            nyse_session_settled_since(
                journal_at, datetime(2026, 11, 27, 23, 44, tzinfo=timezone.utc)
            )
            is False
        )
        assert (
            nyse_session_settled_since(
                journal_at, datetime(2026, 11, 28, 0, 1, tzinfo=timezone.utc)
            )
            is True
        )


def test_horizon_is_respected_end_to_end() -> None:
    """A 14-day prediction graded late must still be scored over 14 days."""
    s = _series("UTC", days=60)
    made = datetime(2026, 1, 1, tzinfo=timezone.utc)
    due = made + timedelta(days=14)

    base, end = close_at(s, made), close_at(s, due)
    assert base is not None and end is not None
    realized = end / base - 1.0

    # 14 daily bars of +1 each, from 100.
    assert realized == pytest.approx(14 / 100)
    # The naive "use the latest bar" version would have scored 59/100.
    assert float(s.iloc[-1]) / base - 1.0 == pytest.approx(59 / 100)


def test_index_subject_uses_explicit_tradeable_proxy(tmp_path) -> None:
    assert cache_symbol_for_subject("NDX") == "QQQ"
    frame = pd.DataFrame({"close": _series("UTC", days=6)})
    path = tmp_path / "etf" / "QQQ"
    path.mkdir(parents=True)
    frame.to_parquet(path / "1D.parquet")

    out = _read_close(tmp_path, "NDX")
    assert out is not None and len(out) == 6
