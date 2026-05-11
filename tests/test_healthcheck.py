"""Tests for the Docker healthcheck script.

The script lives outside the package (under ``docker/``) so we import it
by path. Tests pin the exit-code contract Docker relies on.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

_HEALTHCHECK_PATH = (
    Path(__file__).resolve().parents[1] / "docker" / "healthcheck.py"
)


@pytest.fixture(scope="module")
def healthcheck():
    spec = importlib.util.spec_from_file_location("healthcheck", _HEALTHCHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_hb(path: Path, status: str = "ok") -> None:
    path.write_text(json.dumps({"ts": "2024-01-01T00:00:00+00:00", "status": status, "cycle": 1}))


def test_missing_file_exits_1(healthcheck, tmp_path: Path) -> None:
    assert healthcheck.main(str(tmp_path / "absent.json"), 30) == 1


def test_fresh_ok_heartbeat_exits_0(healthcheck, tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    _write_hb(p, status="ok")
    assert healthcheck.main(str(p), 30) == 0


def test_stale_heartbeat_exits_1(healthcheck, tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    _write_hb(p, status="ok")
    old = time.time() - 3600
    os.utime(p, (old, old))
    assert healthcheck.main(str(p), 30) == 1


def test_error_status_exits_1(healthcheck, tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    _write_hb(p, status="error")
    assert healthcheck.main(str(p), 30) == 1


def test_no_orders_status_is_ok(healthcheck, tmp_path: Path) -> None:
    """Strategy didn't want to trade — still a healthy runner."""
    p = tmp_path / "hb.json"
    _write_hb(p, status="no_orders")
    assert healthcheck.main(str(p), 30) == 0


def test_halted_status_is_ok(healthcheck, tmp_path: Path) -> None:
    """A halt is a deliberate state, not a process-level failure. Container
    stays running so the operator can decide what to do; alerts handle
    the human signal."""
    p = tmp_path / "hb.json"
    _write_hb(p, status="halted")
    assert healthcheck.main(str(p), 30) == 0


def test_unparseable_heartbeat_exits_1(healthcheck, tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    p.write_text("{not valid json")
    assert healthcheck.main(str(p), 30) == 1
