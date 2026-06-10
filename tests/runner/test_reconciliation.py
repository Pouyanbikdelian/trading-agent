"""Tests for the startup-reconciliation drift alert.

Why this exists: today (May 2026) we shipped a bug where the trader
silently used a synthetic empty account and stacked positions 3× target
across three cycles. The cycle is now fail-closed on that path, but a
sibling failure mode is harder to prevent automatically: container
restarts mid-cycle → local store is stale → broker holds positions the
runner doesn't know about. The startup reconciliation detects that
drift and alerts the operator loudly so they can /flatten or accept.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Position,
)
from trading.execution.base import Broker
from trading.runner import Runner, RunnerConfig
from trading.runner.alerts import TelegramAlerts
from trading.runner.cycle import Cycle
from trading.runner.state import RunnerStore


class _RecordingAlerts(TelegramAlerts):
    def __init__(self) -> None:
        super().__init__(token=None, chat_id=None, enabled=False)
        self.messages: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))

    def critical(self, msg: str) -> None:
        self.messages.append(("critical", msg))


class _FakeBroker:
    """Stub broker that returns whatever positions the test wires up."""

    name = "fake"

    def __init__(self, positions: list[Position]) -> None:
        self._positions = positions

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_positions(self) -> list[Position]:
        return list(self._positions)


def _make_runner(tmp_path: Path, broker: Broker, snap: AccountSnapshot | None) -> Runner:
    """Build a minimal Runner whose only purpose is to call _reconcile_startup."""
    store = RunnerStore(tmp_path / "runner.db")
    if snap is not None:
        store.save_snapshot(snap)

    cycle = Cycle.__new__(Cycle)  # bypass full __init__; we only need runner_store
    cycle.runner_store = store

    runner = Runner.__new__(Runner)
    runner.cycle = cycle
    runner.broker = broker
    runner.alerts = _RecordingAlerts()
    runner.config = RunnerConfig(universe="sp500", strategies=["top_k_momentum"])
    return runner


def _pos(symbol: str, qty: float) -> Position:
    return Position(
        instrument=Instrument(symbol=symbol, asset_class=AssetClass.EQUITY),
        quantity=qty,
        avg_price=100.0,
    )


def test_clean_book_emits_ok_message(tmp_path: Path) -> None:
    broker = _FakeBroker(positions=[])
    snap = AccountSnapshot(ts=datetime.now(tz=timezone.utc), cash=100_000, equity=100_000)
    runner = _make_runner(tmp_path, broker, snap)

    runner._reconcile_startup()

    msgs = [m for level, m in runner.alerts.messages if level in ("info", "critical")]
    assert any("startup reconciliation" in m for m in msgs)
    assert not any("DIFFER" in m for m in msgs)


def test_broker_has_positions_snapshot_has_none_alerts(tmp_path: Path) -> None:
    """The 3x-overbuy scenario: snapshot says zero, broker says 8 positions.
    This is the failure mode we just lived through — the alert must fire."""
    broker = _FakeBroker(positions=[_pos("CIEN", 51), _pos("MRNA", 623), _pos("LITE", 30)])
    snap = AccountSnapshot(ts=datetime.now(tz=timezone.utc), cash=100_000, equity=100_000)
    runner = _make_runner(tmp_path, broker, snap)

    runner._reconcile_startup()

    critical = [m for level, m in runner.alerts.messages if level == "critical"]
    assert len(critical) == 1
    msg = critical[0]
    assert "DIFFER" in msg
    assert "CIEN" in msg and "MRNA" in msg and "LITE" in msg
    # Each symbol's broker qty surfaces in the alert.
    assert "51" in msg and "623" in msg and "30" in msg


def test_snapshot_has_positions_broker_has_none_alerts(tmp_path: Path) -> None:
    """The mirror case: local store thinks we hold 1 position but broker
    is flat. Could happen if a fill record persisted before the actual
    fill came through, or if positions were closed outside the system."""
    broker = _FakeBroker(positions=[])
    pos = _pos("AAPL", 100)
    snap = AccountSnapshot(
        ts=datetime.now(tz=timezone.utc),
        cash=100_000,
        equity=100_000,
        positions={pos.instrument.key: pos},
    )
    runner = _make_runner(tmp_path, broker, snap)

    runner._reconcile_startup()

    critical = [m for level, m in runner.alerts.messages if level == "critical"]
    assert len(critical) == 1
    assert "AAPL" in critical[0]


def test_quantities_differ_alerts(tmp_path: Path) -> None:
    """Even matching symbols are flagged if the qty differs (partial fill,
    manual trade outside the system, etc.)."""
    broker = _FakeBroker(positions=[_pos("AAPL", 100)])
    pos = _pos("AAPL", 50)
    snap = AccountSnapshot(
        ts=datetime.now(tz=timezone.utc),
        cash=100_000,
        equity=100_000,
        positions={pos.instrument.key: pos},
    )
    runner = _make_runner(tmp_path, broker, snap)

    runner._reconcile_startup()

    critical = [m for level, m in runner.alerts.messages if level == "critical"]
    assert len(critical) == 1
    assert "AAPL" in critical[0]
    assert "100" in critical[0] and "50" in critical[0]


def test_broker_failure_is_logged_not_raised(tmp_path: Path) -> None:
    """If broker.get_positions fails (network, gateway wedge), we should
    skip the drift check — not crash the runner during startup."""

    class _ExplodingBroker:
        name = "exploding"

        def get_positions(self) -> list[Position]:
            raise RuntimeError("simulated gateway timeout")

    runner = _make_runner(tmp_path, _ExplodingBroker(), snap=None)  # type: ignore[arg-type]
    # Must not raise.
    runner._reconcile_startup()
    # No critical alert — we couldn't even check drift.
    assert not any(level == "critical" for level, _ in runner.alerts.messages)


def test_no_snapshot_yet_still_works(tmp_path: Path) -> None:
    """First-time startup with no snapshot file. Broker may have positions
    (e.g. operator imported a portfolio manually); we should still surface
    them so the operator knows what they're starting from."""
    broker = _FakeBroker(positions=[_pos("AAPL", 10)])
    runner = _make_runner(tmp_path, broker, snap=None)

    runner._reconcile_startup()
    critical = [m for level, m in runner.alerts.messages if level == "critical"]
    assert len(critical) == 1
    assert "AAPL" in critical[0]
