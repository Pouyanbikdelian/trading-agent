"""State-directory env stamping — the guard against 2026-08-07's first incident.

The bug being pinned: a process running one TRADING_ENV wrote equity
figures into another environment's state directory, and nothing noticed
until the kill switch measured a live account against a paper baseline.
"""

from __future__ import annotations

import json

import pytest

from trading.core.state_env import (
    STAMP_FILENAME,
    StateEnvMismatchError,
    assert_state_dir_env,
    read_stamp,
    verify_state_dir,
    write_stamp,
)


def test_first_open_adopts_and_stamps(tmp_path):
    check = verify_state_dir(tmp_path, "paper")
    assert check.ok and check.adopted
    assert check.stamped_env is None
    assert read_stamp(tmp_path) == "paper"


def test_second_open_same_env_is_a_no_op(tmp_path):
    verify_state_dir(tmp_path, "paper")
    check = verify_state_dir(tmp_path, "paper")
    assert check.ok and not check.adopted
    assert check.stamped_env == "paper"


def test_mismatch_is_refused(tmp_path):
    """The exact 2026-08-07 sequence: paper stamps it, live opens it."""
    verify_state_dir(tmp_path, "paper")
    check = verify_state_dir(tmp_path, "live")
    assert not check.ok
    assert check.stamped_env == "paper"
    assert "paper" in check.detail and "live" in check.detail


def test_mismatch_in_either_direction(tmp_path):
    """Live-stamped dir opened by a paper process is equally wrong — this
    is the direction the incident actually took (paper wrote into live)."""
    verify_state_dir(tmp_path, "live")
    assert not verify_state_dir(tmp_path, "paper").ok


def test_assert_raises_with_an_actionable_message(tmp_path):
    verify_state_dir(tmp_path, "paper")
    with pytest.raises(StateEnvMismatchError) as exc:
        assert_state_dir_env(tmp_path, "live")
    msg = str(exc.value)
    assert "REFUSING TO START" in msg
    # The message must name the fix, not just the fault.
    assert "STATE_DIR" in msg
    assert "trading state stamp" in msg


def test_mismatch_does_not_rewrite_the_stamp(tmp_path):
    """A refused process must leave the directory exactly as it found it —
    otherwise a retry loop silently converts the dir to the wrong env."""
    verify_state_dir(tmp_path, "paper")
    verify_state_dir(tmp_path, "live")
    assert read_stamp(tmp_path) == "paper"


def test_corrupt_stamp_reads_as_unstamped(tmp_path):
    (tmp_path / STAMP_FILENAME).write_text("{not json")
    assert read_stamp(tmp_path) is None
    assert verify_state_dir(tmp_path, "paper").ok


def test_adopt_false_leaves_the_directory_untouched(tmp_path):
    check = verify_state_dir(tmp_path, "paper", adopt=False)
    assert check.ok and not check.adopted
    assert not (tmp_path / STAMP_FILENAME).exists()


def test_adopting_a_directory_that_already_has_state_is_flagged(tmp_path):
    (tmp_path / "halt.json").write_text("{}")
    check = verify_state_dir(tmp_path, "live")
    assert check.ok
    assert "already held trading state" in check.detail


def test_write_stamp_is_atomic_and_leaves_no_temp_files(tmp_path):
    write_stamp(tmp_path, "live")
    write_stamp(tmp_path, "live")
    assert sorted(p.name for p in tmp_path.iterdir()) == [STAMP_FILENAME]
    payload = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert payload["trading_env"] == "live"
    assert "stamped_at" in payload


def test_restamping_is_how_an_operator_migrates_a_directory(tmp_path):
    verify_state_dir(tmp_path, "paper")
    write_stamp(tmp_path, "live")
    assert assert_state_dir_env(tmp_path, "live").ok
