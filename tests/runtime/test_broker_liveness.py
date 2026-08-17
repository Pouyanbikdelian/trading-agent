"""Authenticated broker-liveness state is reliable and non-invasive."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.runtime.broker_liveness import (
    FILENAME,
    record_broker_liveness,
    record_broker_liveness_failure,
)

NOW = datetime(2026, 8, 17, 8, 0, tzinfo=timezone.utc)


class _HealthyBroker:
    def probe_liveness(self) -> datetime:
        return datetime(2026, 8, 17, 8, 0)  # Gateway commonly returns naive UTC.


class _UnavailableBroker:
    def probe_liveness(self) -> datetime:
        raise ConnectionError("Gateway is waiting for IBKR Mobile approval")


def test_successful_probe_is_normalized_and_persisted_atomically(tmp_path: Path) -> None:
    result = record_broker_liveness(_HealthyBroker(), tmp_path, now=NOW)

    assert result is not None and result["ready"] is True
    payload = json.loads((tmp_path / FILENAME).read_text())
    assert payload == {
        "checked_at": NOW.isoformat(),
        "ready": True,
        "probe": "reqCurrentTime",
        "server_time": NOW.isoformat(),
        "last_success_at": NOW.isoformat(),
    }
    assert not list(tmp_path.glob(f"{FILENAME}.*"))


def test_failed_probe_preserves_the_last_authenticated_response(tmp_path: Path) -> None:
    record_broker_liveness(_HealthyBroker(), tmp_path, now=NOW)

    result = record_broker_liveness(_UnavailableBroker(), tmp_path, now=NOW + timedelta(minutes=2))

    assert result is not None and result["ready"] is False
    assert result["last_success_at"] == NOW.isoformat()
    assert "ConnectionError" in str(result["detail"])
    assert json.loads((tmp_path / FILENAME).read_text()) == result


def test_explicit_connect_failure_preserves_prior_success_and_its_detail(tmp_path: Path) -> None:
    """Bootstrap failures happen before a probe can inspect an API socket."""
    record_broker_liveness(_HealthyBroker(), tmp_path, now=NOW)

    result = record_broker_liveness_failure(
        tmp_path,
        "ConnectionRefusedError: Gateway is awaiting mobile approval",
        probe="connect",
        now=NOW + timedelta(minutes=1),
    )

    assert result == {
        "checked_at": (NOW + timedelta(minutes=1)).isoformat(),
        "ready": False,
        "probe": "connect",
        "detail": "ConnectionRefusedError: Gateway is awaiting mobile approval",
        "last_success_at": NOW.isoformat(),
    }
    assert json.loads((tmp_path / FILENAME).read_text()) == result


def test_unsupported_broker_does_not_create_a_false_health_requirement(tmp_path: Path) -> None:
    assert record_broker_liveness(object(), tmp_path, now=NOW) is None
    assert not (tmp_path / FILENAME).exists()


def test_naive_observation_time_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        record_broker_liveness(_HealthyBroker(), tmp_path, now=NOW.replace(tzinfo=None))
