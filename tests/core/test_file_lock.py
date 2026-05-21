"""Tests for the cross-process file lock helper.

These are smoke-level: we verify the context manager grants + releases,
and that a reentrant attempt from a separate process via subprocess
would block (verified by simulating with threads, which fcntl.flock also
serialises within the same process when keyed to different fds).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from trading.core.file_lock import file_lock


def test_lock_creates_sibling_lock_file(tmp_path: Path) -> None:
    target = tmp_path / "halt.json"
    with file_lock(target):
        # The .lock file exists as long as the context is active.
        assert (tmp_path / "halt.json.lock").exists()


def test_lock_yields_and_releases_cleanly(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    # Should be able to acquire twice in sequence.
    with file_lock(target):
        pass
    with file_lock(target):
        pass  # second acquire must not block


def test_concurrent_writers_serialise(tmp_path: Path) -> None:
    r"""Two threads racing to write the same file under the lock must
    serialise — the final file content is from whichever thread wrote
    last, NOT a torn mix of both payloads."""
    target = tmp_path / "halt.json"
    output_order: list[str] = []

    def writer(payload: str, delay_in_lock: float = 0.05) -> None:
        with file_lock(target):
            output_order.append(f"enter:{payload}")
            time.sleep(delay_in_lock)
            target.write_text(payload)
            output_order.append(f"exit:{payload}")

    t1 = threading.Thread(target=writer, args=("first",))
    t2 = threading.Thread(target=writer, args=("second",))
    t1.start()
    time.sleep(0.01)  # give t1 a head start
    t2.start()
    t1.join()
    t2.join()

    # The enter/exit events must NOT be interleaved if the lock works.
    assert output_order == [
        "enter:first",
        "exit:first",
        "enter:second",
        "exit:second",
    ]
    # The final file is whichever thread wrote last (second).
    assert target.read_text() == "second"
