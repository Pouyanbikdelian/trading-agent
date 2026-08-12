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

from trading.runtime.portfolio_stats import _read_close, cache_symbol_for_subject, close_at, covers


def _series(tz: str | None, days: int = 30) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=days, freq="D", tz=tz)
    return pd.Series(range(100, 100 + days), index=idx, dtype="float64")


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
