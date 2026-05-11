"""Live runner — schedules cycles, persists state, alerts on errors.

Public surface::

    from trading.runner import RunnerConfig, Runner, Cycle, CycleReport
    from trading.runner import RunnerStore, TelegramAlerts, NullAlerts
    from trading.runner import write_heartbeat, read_heartbeat, heartbeat_is_stale
"""

from __future__ import annotations

from trading.runner.alerts import NullAlerts, TelegramAlerts
from trading.runner.config import RunnerConfig
from trading.runner.cycle import Cycle, CycleReport
from trading.runner.heartbeat import (
    heartbeat_age_seconds,
    heartbeat_is_stale,
    read_heartbeat,
    write_heartbeat,
)
from trading.runner.runner import Runner
from trading.runner.state import RunnerStore

__all__ = [
    "Cycle",
    "CycleReport",
    "NullAlerts",
    "Runner",
    "RunnerConfig",
    "RunnerStore",
    "TelegramAlerts",
    "heartbeat_age_seconds",
    "heartbeat_is_stale",
    "read_heartbeat",
    "write_heartbeat",
]
