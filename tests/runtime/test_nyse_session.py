"""NYSE session calendar facts used by continuous risk monitoring."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from trading.runtime.nyse_session import (
    OPENING_CAPTURE_DURATION,
    current_nyse_session,
    current_nyse_session_label,
    is_opening_capture_window,
    nyse_session_on,
    opening_capture_bounds,
)


def test_current_session_uses_new_york_date_not_utc_date() -> None:
    """UTC midnight is still the preceding NYSE calendar day in New York."""
    session = current_nyse_session(datetime(2026, 3, 10, 1, 30, tzinfo=timezone.utc))

    assert session is not None
    assert session.label == date(2026, 3, 9)
    assert current_nyse_session_label(datetime(2026, 3, 10, 1, 30, tzinfo=timezone.utc)) == date(
        2026, 3, 9
    )


@pytest.mark.parametrize(
    ("label", "expected_open", "expected_close"),
    [
        # The Monday after the March DST shift opens at 13:30 UTC (09:30 EDT).
        (
            date(2026, 3, 9),
            datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc),
            datetime(2026, 3, 9, 20, 0, tzinfo=timezone.utc),
        ),
        # After the November DST shift the same local session opens at 14:30 UTC.
        (
            date(2026, 11, 2),
            datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2026, 11, 2, 21, 0, tzinfo=timezone.utc),
        ),
    ],
)
def test_session_boundaries_follow_dst(
    label: date,
    expected_open: datetime,
    expected_close: datetime,
) -> None:
    session = nyse_session_on(label)

    assert session is not None
    assert session.market_open == expected_open
    assert session.market_close == expected_close


@pytest.mark.parametrize(
    "label",
    [
        date(2026, 3, 14),  # Saturday
        date(2026, 4, 3),  # Good Friday
        date(2026, 7, 3),  # Independence Day observed
    ],
)
def test_weekends_and_holidays_have_no_session(label: date) -> None:
    assert nyse_session_on(label) is None
    assert (
        # Noon UTC is safely inside the same New York calendar date in both
        # standard and daylight time; midnight UTC can still be the prior ET day.
        current_nyse_session(
            datetime.combine(label, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=12)
        )
        is None
    )


def test_early_close_comes_from_exchange_calendar() -> None:
    """The day after Thanksgiving closes at 13:00 EST, not the normal 16:00."""
    session = nyse_session_on(date(2026, 11, 27))

    assert session is not None
    assert session.market_open == datetime(2026, 11, 27, 14, 30, tzinfo=timezone.utc)
    assert session.market_close == datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc)


def test_opening_capture_window_is_narrow_and_half_open() -> None:
    opens_at = datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)

    assert opening_capture_bounds(opens_at) == (
        opens_at,
        opens_at + OPENING_CAPTURE_DURATION,
    )
    assert is_opening_capture_window(opens_at)
    assert is_opening_capture_window(
        opens_at + OPENING_CAPTURE_DURATION - timedelta(microseconds=1)
    )
    assert not is_opening_capture_window(opens_at + OPENING_CAPTURE_DURATION)
    assert not is_opening_capture_window(opens_at - timedelta(microseconds=1))


def test_opening_capture_window_is_not_open_on_holidays() -> None:
    assert opening_capture_bounds(datetime(2026, 4, 3, 13, 30, tzinfo=timezone.utc)) is None
    assert not is_opening_capture_window(datetime(2026, 4, 3, 13, 30, tzinfo=timezone.utc))


def test_naive_instants_and_invalid_capture_durations_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        current_nyse_session(datetime(2026, 3, 9, 9, 30))
    with pytest.raises(ValueError, match="positive"):
        opening_capture_bounds(
            datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc),
            duration=timedelta(0),
        )
