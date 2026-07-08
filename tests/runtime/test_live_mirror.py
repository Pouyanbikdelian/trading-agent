"""Live mirror — hermetic: fake broker, tmp state dir, no network."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading.core.types import AccountSnapshot
from trading.runner.state import RunnerStore
from trading.runtime import live_mirror


class _FakeBroker:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            ts=datetime.now(tz=timezone.utc),
            cash=5_000.0,
            equity=25_000.0,
            positions={},
            base_currency="CHF",
        )


class _DeadBroker(_FakeBroker):
    def connect(self) -> None:
        raise ConnectionRefusedError("gateway down")


def test_snapshot_once_persists(tmp_path: Path, monkeypatch) -> None:
    import trading.execution.ibkr as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IbkrBroker", _FakeBroker)
    ok = live_mirror.snapshot_once(tmp_path, "127.0.0.1", 4001, 27)
    assert ok is True
    snap = RunnerStore(tmp_path / "runner.db").latest_snapshot()
    assert snap is not None and snap.equity == 25_000.0
    assert snap.base_currency == "CHF"


def test_snapshot_once_survives_dead_gateway(tmp_path: Path, monkeypatch) -> None:
    """A down gateway must log-and-return, never raise — the loop retries."""
    import trading.execution.ibkr as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "IbkrBroker", _DeadBroker)
    ok = live_mirror.snapshot_once(tmp_path, "127.0.0.1", 4001, 27)
    assert ok is False
    assert not (tmp_path / "runner.db").exists()  # nothing half-written
