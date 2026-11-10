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

    def get_open_orders(self) -> list[Order]:
        """Orders working at the broker RIGHT NOW (submitted, unfilled),
        from ALL sources — cycle, guard exits, manual commands. The risk
        manager nets these into current positions when sizing deltas.
        Without this, after-hours order batches stack blindly: on
        2026-07-15 guard closes + two cycles all filled at the open and
        the paper book went SHORT two names. Default: [] (no working
        orders)."""
        return []

    def get_fx_rates(self) -> dict[str, float]:
        """Base-currency units per 1 unit of each foreign currency, e.g.
        ``{"USD": 0.8081}`` on a CHF-base account. Used by the risk
        manager to size positions in base-currency terms when the
        instrument trades in another currency (the CHF/USD sizing bug,
        2026-07-14). Default: empty — same-currency sizing assumed."""
        return {}

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


class OrderNotSentError(BrokerError):
    """Submission failed BEFORE anything was written to the broker.

    Why (2026-11-10 review). The November 8 fix rightly stopped treating a
    lost acknowledgement as a rejection: the order may be working, so its
    ledger row stays open and the desk halts. But it classified by error
    text, and the most common failure of all — the gateway is down or
    waiting for 2FA, so ``_ensure_connected`` raises before ``placeOrder``
    is ever called — reads "not connected". That order provably never left
    this process, yet the desk halted, told the operator IBKR "may have
    received" it, and left a PENDING row that nothing can settle and that
    cannot be retired for seven days: the September ghost rows, rebuilt.

    Raise this only where the adapter can prove no bytes reached the
    broker. Everything raised at or after the send stays "outcome unknown".
    """


class OrderNotSentDisconnectedError(OrderNotSentError, NotConnectedError):
    """Not sent because the session was down — still a ``NotConnectedError``
    to every caller that already handles one."""


#: Substrings that mark a dead session rather than a refused order.
DISCONNECT_MARKERS = (
    "not connected",
    "connection",
    "connect call failed",
    "timeout",
    "timed out",
    "broken pipe",
    "reset by peer",
    "no route to host",
    "econnrefused",
    "refused",
)


def looks_like_disconnect(exc: BaseException) -> bool:
    """Distinguish a dead session from a refused order."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in DISCONNECT_MARKERS)


def submission_outcome_unknown(exc: BaseException) -> bool:
    """Could the broker be working an order whose submission raised ``exc``?

    One rule for every submit path (cycle, playbook flatten, operator
    commands) — on 2026-11-08 only the cycle learned that a lost
    acknowledgement is not a rejection, and the other two kept writing
    REJECTED over orders that might be live.

    * ``OrderNotSentError`` — failed before the send: definitely not working.
    * an explicit ``BrokerError`` that does not read like a dead session —
      the broker answered and refused: definitely not working.
    * anything else (timeout, dropped socket, unexpected exception) —
      unknown. Keep the ledger row open so a late fill can still settle it.
    """
    if isinstance(exc, OrderNotSentError):
        return False
    return looks_like_disconnect(exc) or not isinstance(exc, BrokerError)
