"""Halting must not give the kill switches amnesia.

Four writers used to overwrite halt.json with a four-key dict, dropping
``equity_high_watermark``, ``daily_equity_open`` and ``last_day``.
``HaltState`` neither forbids extras nor requires those fields, so they
reloaded as ``0.0 / 0.0 / None`` — silently. The daily-loss switch skips
a zero baseline outright, and the drawdown switch restarts its peak from
whatever equity happens to be, so neither could fire on a decline
spanning a halt.

Also pinned here: ``flatten_on_next_cycle`` is gone. It was written by
/halt and by the CLI and read by nothing, so /halt replied "Next cycle
will force-flatten positions" and no cycle ever did.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from trading.risk.halt_file import (
    halt_path,
    read_halt_state,
    set_halted,
    write_halt_state,
)
from trading.risk.limits import HaltState

SEEDED = HaltState(
    halted=False,
    equity_high_watermark=94_120.0,
    daily_equity_open=87_413.0,
    last_day=date(2026, 8, 7),
)


def test_halting_preserves_both_baselines(tmp_path):
    write_halt_state(tmp_path, SEEDED)

    new = set_halted(tmp_path, halted=True, reason="telegram")

    assert new.halted and new.reason == "telegram"
    assert new.daily_equity_open == 87_413.0
    assert new.equity_high_watermark == 94_120.0
    assert new.last_day == date(2026, 8, 7)


def test_resuming_preserves_both_baselines(tmp_path):
    write_halt_state(tmp_path, SEEDED)
    set_halted(tmp_path, halted=True, reason="telegram")

    new = set_halted(tmp_path, halted=False)

    assert not new.halted and new.reason == ""
    assert new.halted_at is None
    assert new.daily_equity_open == 87_413.0
    assert new.equity_high_watermark == 94_120.0


def test_a_halt_resume_round_trip_survives_on_disk(tmp_path):
    """The regression that matters: read it back the way the risk manager
    does, not just the in-memory return value."""
    write_halt_state(tmp_path, SEEDED)
    set_halted(tmp_path, halted=True, reason="x")
    set_halted(tmp_path, halted=False)

    reloaded = HaltState.model_validate_json(halt_path(tmp_path).read_text())

    assert reloaded.equity_high_watermark == 94_120.0
    assert reloaded.daily_equity_open == 87_413.0


def test_halted_at_is_timezone_aware(tmp_path):
    """The runner's auto-halt used a naive datetime.now() — uninterpretable
    from a container whose local time is not UTC."""
    new = set_halted(tmp_path, halted=True, reason="auto")

    assert new.halted_at is not None
    assert new.halted_at.tzinfo is not None


def test_resume_clears_the_reason_and_timestamp(tmp_path):
    set_halted(tmp_path, halted=True, reason="daily loss")

    new = set_halted(tmp_path, halted=False)

    assert new.reason == "" and new.halted_at is None


def test_the_flatten_flag_is_not_written_anymore(tmp_path):
    set_halted(tmp_path, halted=True, reason="telegram")

    payload = json.loads(halt_path(tmp_path).read_text())

    assert "flatten_on_next_cycle" not in payload


def test_a_legacy_file_with_the_flatten_flag_still_loads(tmp_path):
    """Deployed state dirs have one on disk right now."""
    halt_path(tmp_path).write_text(
        json.dumps(
            {
                "halted": True,
                "reason": "telegram",
                "halted_at": "2026-08-07T15:21:50.692536+00:00",
                "flatten_on_next_cycle": True,
            }
        )
    )

    state = read_halt_state(tmp_path)

    assert state.halted and state.reason == "telegram"


def test_a_corrupt_file_does_not_wedge_the_writer(tmp_path):
    halt_path(tmp_path).write_text("{not json")

    new = set_halted(tmp_path, halted=True, reason="x")

    assert new.halted


def test_missing_file_starts_from_defaults(tmp_path):
    assert read_halt_state(tmp_path) == HaltState()


def test_explicit_halted_at_is_respected(tmp_path):
    ts = datetime(2026, 8, 7, 15, 21, tzinfo=timezone.utc)

    new = set_halted(tmp_path, halted=True, reason="x", halted_at=ts)

    assert new.halted_at == ts
