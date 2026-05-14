"""Broker Protocol — the surface every order-router must implement.

Two concrete brokers in this codebase:
  * ``Simulator`` (``trading.execution.simulator``) — in-memory, used by
    backtesting and dry runs.
  * ``IbkrBroker`` (``trading.execution.ibkr``) — ``ib-async`` adapter for
    paper/live trading.

The Protocol is intentionally synchronous. IBKR is async at the wire and
the adapter bridges internally; the runner (Phase 8) is the only async
caller. Keeping the surface synchronous keeps the simulator and the
risk-manager unit tests cheap to write.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from trading.core.types import AccountSnapshot, Bar, Fill, Order, Position


@runtime_checkable
class Broker(Protocol):
    """Anything that can route orders and report state."""

    name: str

    # --- lifecycle -----------------------------------------------------
    def connect(self) -> None:
        """Establish the underlying transport. No-op for in-memory brokers."""

    def disconnect(self) -> None:
        """Tear down the transport. Safe to call when already disconnected."""

    # --- orders --------------------------------------------------------
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker. Returns a copy with the
        broker-assigned status / ID. Raises ``BrokerError`` on rejection.

        Idempotent on ``client_order_id``: submitting the same id twice
        is a programmer error and brokers should reject it. The store
        layer enforces this for us by setting client_order_id as the PK.
        """

    def cancel_order(self, client_order_id: str) -> None:
        """Cancel an open order. No-op if the order is already terminal."""

    # --- state ---------------------------------------------------------
    def get_positions(self) -> list[Position]: ...

    def get_account(self) -> AccountSnapshot: ...

    def get_fills(self, *, since: datetime | None = None) -> list[Fill]:
        """All fills the broker has reported, optionally filtered by time.

        ``since`` is inclusive of its own bar (``>= since``)."""

    def tick(self, ts: datetime, bars: dict[str, Bar]) -> list[Fill]:
        """Advance the broker's internal clock by one bar (if it has one).

        IBKR doesn't need this — its fills arrive asynchronously from the
        gateway — so the adapter implements it as a no-op. The in-memory
        Simulator overrides it to fill queued market orders against the
        new bar. The runner calls ``tick()`` once per cycle after submitting
        orders so paper-trade fills materialize in the same cycle they were
        submitted in.

        Default: return ``[]`` (no fills produced)."""
        return []


class BrokerError(RuntimeError):
    """Raised when a broker rejects an action (bad symbol, no funds, etc.)."""


class NotConnectedError(BrokerError):
    """Operation attempted on a broker that hasn't been connected."""
