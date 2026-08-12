"""The Historian must run often enough to notice a broken learning loop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading.runner.runner import _add_scorecard_backfill_targets, _historian_trigger


def test_historian_runs_tuesday_then_friday() -> None:
    trigger = _historian_trigger()
    monday = datetime(2026, 8, 10, 23, 0, tzinfo=timezone.utc)

    tuesday = trigger.get_next_fire_time(None, monday)
    assert tuesday == datetime(2026, 8, 11, 22, 45, tzinfo=timezone.utc)

    friday = trigger.get_next_fire_time(tuesday, tuesday + timedelta(seconds=1))
    assert friday == datetime(2026, 8, 14, 22, 45, tzinfo=timezone.utc)


def test_scorecard_backfill_fetches_an_out_of_universe_index_proxy() -> None:
    default_start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    earliest = datetime(2026, 1, 15, tzinfo=timezone.utc)
    symbols = {"AAPL"}
    starts = {"AAPL": default_start}

    _add_scorecard_backfill_targets(
        symbols,
        starts,
        [
            {"subject": "NDX", "earliest_ts": earliest.timestamp()},
            {"subject": "PORTFOLIO", "earliest_ts": earliest.timestamp()},
        ],
        default_start=default_start,
    )

    assert symbols == {"AAPL", "QQQ"}
    assert starts["QQQ"] == earliest - timedelta(days=3)
