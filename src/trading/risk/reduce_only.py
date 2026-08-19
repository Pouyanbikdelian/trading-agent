"""Verified reduce-only order filtering, shared by every halted execution path.

A halt must stop the system from *adding* risk. It must not stop the operator
from *removing* it — that was the failure on 2026-08-18, when a daily-loss
halt refused two trailing-stop exits and left the book fully exposed.

Two callers need the same proof, so the proof lives here rather than being
written twice:

* ``trading.runtime.command_processor`` — ``/close`` and ``/flatten`` while
  halted.
* ``trading.runner.cycle`` — the defensive rebalance a halted ``/cycle``
  proposes, which the operator still has to approve.

Every helper here is pure and fails closed. Anything it cannot prove reduces
exposure is rejected, never guessed at. The caller is expected to re-run
``filter_reduce_only`` against a *fresh* broker snapshot immediately before
submitting, because an approval window can be minutes long and the position
it was built from may no longer exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from trading.core.types import AssetClass, Instrument, Order, Position

# Position quantities from IBKR may be fractional.  Keep a tiny tolerance for
# arithmetic around a completely-covered position, but never silently turn a
# material over-sell into a zero-quantity no-op.
QUANTITY_EPSILON = 1e-9

# Broker contracts are not represented consistently across every IBKR API
# response.  In particular, the same listed fund can arrive as ETF/ARCA in a
# position snapshot and EQUITY/SMART in openTrades.  Exact Instrument model
# equality would then miss an already-working exit and allow a duplicate sell.
#
# Exchange is intentionally not part of this identity.  For a held account
# position, treating same-symbol stock/ETF contracts on different exchanges as
# one identity is fail-safe: it may refuse/under-close in an unusual routing
# setup, but it cannot submit an order that crosses the actual holding.
BrokerInstrumentIdentity = tuple[str, str, str]


def validated_quantity(value: Any, *, label: str) -> float:
    """Coerce a broker-supplied quantity, refusing anything non-finite."""
    try:
        quantity = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {label} quantity; no order sent") from e
    if not isfinite(quantity):
        raise ValueError(f"invalid {label} quantity; no order sent")
    return quantity


def open_order_side(order: Order) -> Any:
    """Normalise a broker order side, rejecting an unrecognisable one."""
    from trading.core.types import Side

    raw = getattr(order.side, "value", order.side)
    try:
        return Side(str(raw).lower())
    except ValueError as e:
        raise ValueError(
            f"cannot verify working order side for {order.instrument.symbol}; no order sent"
        ) from e


def normalized_broker_instrument_identity(
    instrument: Instrument,
) -> BrokerInstrumentIdentity:
    """Return the conservative identity used for reduce-only verification.

    IBKR's position and open-order APIs sometimes disagree on whether a
    listed fund is ``ETF`` or ``EQUITY`` and whether its exchange is ``SMART``
    or a primary venue.  For this safety check both describe the same stock
    exposure, so normalize them to one ``stock`` class and intentionally omit
    exchange.  Currency remains mandatory: a missing or malformed identifier
    is not something an emergency exit may guess about.
    """

    def _part(value: Any, *, label: str, upper: bool) -> str:
        try:
            normalized = str(value).strip()
        except Exception as e:
            raise ValueError(f"invalid broker {label}; no order sent") from e
        if not normalized:
            raise ValueError(f"invalid broker {label}; no order sent")
        return normalized.upper() if upper else normalized.lower()

    symbol = _part(getattr(instrument, "symbol", None), label="symbol", upper=True)
    currency = _part(getattr(instrument, "currency", None), label="currency", upper=True)
    raw_asset_class = getattr(
        getattr(instrument, "asset_class", None),
        "value",
        getattr(instrument, "asset_class", None),
    )
    asset_class = _part(raw_asset_class, label="asset class", upper=False)
    asset_group = (
        "stock" if asset_class in {AssetClass.EQUITY.value, AssetClass.ETF.value} else asset_class
    )
    return asset_group, symbol, currency


@dataclass(frozen=True)
class ReduceOnlyFilter:
    """The verified subset of a plan, plus a reason for everything removed.

    ``kept`` orders are safe to submit while halted: each one has a live
    position of the opposite sign and a quantity that cannot take that
    position past flat, even counting the broker's own working orders.
    ``clamped`` records the names whose size was cut down to the remaining
    headroom rather than dropped — a strictly smaller order is still a
    reduction, so refusing it outright would be needlessly unhelpful.
    """

    kept: list[Order]
    dropped: list[tuple[str, str]]
    clamped: list[tuple[str, float, float]]

    @property
    def has_orders(self) -> bool:
        return bool(self.kept)


def _exposure_sides(settled: float) -> tuple[Any, Any]:
    from trading.core.types import Side

    if settled > 0:
        return Side.SELL, Side.BUY
    return Side.BUY, Side.SELL


def filter_reduce_only(
    orders: list[Order],
    positions: list[Position],
    open_orders: list[Order],
) -> ReduceOnlyFilter:
    """Keep only the planned orders that provably lower exposure.

    An order survives when *all* of the following hold:

    1. a live position exists for the same conservative instrument identity;
    2. the order's side is that position's exit side (sell a long, buy back
       a short) — an order that would open, flip or add is dropped;
    3. no working broker order on that instrument points the other way;
    4. its quantity fits inside the remaining headroom after the broker's
       existing same-direction working orders, clamped down when it does not.

    A malformed position, order or side is a rejection, not an exception:
    a defensive rebalance should lose one name, not the whole basket. The
    exception is a malformed *planned* order, which is our own bug and is
    dropped with the reason recorded.
    """
    kept: list[Order] = []
    dropped: list[tuple[str, str]] = []
    clamped: list[tuple[str, float, float]] = []

    # Index the live book once. A duplicate identity means the account view
    # is ambiguous and nothing on that name may be proven safe.
    live: dict[BrokerInstrumentIdentity, float] = {}
    ambiguous: set[BrokerInstrumentIdentity] = set()
    for position in positions:
        try:
            identity = normalized_broker_instrument_identity(position.instrument)
            settled = validated_quantity(position.quantity, label="position")
        except ValueError:
            continue
        if abs(settled) <= QUANTITY_EPSILON:
            continue
        if identity in live:
            ambiguous.add(identity)
            continue
        live[identity] = settled

    # Working orders, bucketed by identity and direction.
    working: dict[BrokerInstrumentIdentity, dict[Any, float]] = {}
    unverifiable: set[BrokerInstrumentIdentity] = set()
    for order in open_orders:
        try:
            identity = normalized_broker_instrument_identity(order.instrument)
            quantity = validated_quantity(order.quantity, label="working order")
            side = open_order_side(order)
        except ValueError:
            # An unreadable working order could be anything, including an
            # exposure-increasing one. Poison only the names it could touch.
            try:
                unverifiable.add(normalized_broker_instrument_identity(order.instrument))
            except ValueError:
                # Not even the instrument is readable: refuse everything,
                # because this order might belong to any name in the plan.
                return ReduceOnlyFilter(
                    kept=[],
                    dropped=[("*", "a working broker order could not be read; no order sent")],
                    clamped=[],
                )
            continue
        if quantity <= QUANTITY_EPSILON:
            continue
        working.setdefault(identity, {})
        working[identity][side] = working[identity].get(side, 0.0) + quantity

    # Several planned orders should never share an instrument, but if they
    # ever did, the headroom must be shared rather than granted twice.
    consumed: dict[BrokerInstrumentIdentity, float] = {}

    for order in orders:
        symbol = str(getattr(order.instrument, "symbol", "?")).upper()
        try:
            identity = normalized_broker_instrument_identity(order.instrument)
            requested = validated_quantity(order.quantity, label="planned order")
            side = open_order_side(order)
        except ValueError as e:
            dropped.append((symbol, str(e)))
            continue

        if identity in ambiguous:
            dropped.append((symbol, "multiple live positions match this contract"))
            continue
        if identity in unverifiable:
            dropped.append((symbol, "a working broker order on this name could not be read"))
            continue
        held_quantity = live.get(identity)
        if held_quantity is None:
            dropped.append((symbol, "no live position — this order would open new exposure"))
            continue

        exit_side, increasing_side = _exposure_sides(held_quantity)
        if side == increasing_side:
            dropped.append((symbol, "would increase an existing position"))
            continue
        if side != exit_side:
            dropped.append((symbol, "cannot verify the order direction"))
            continue

        by_side = working.get(identity, {})
        if by_side.get(increasing_side, 0.0) > QUANTITY_EPSILON:
            dropped.append(
                (symbol, "a working order on this name would increase exposure; cancel it first")
            )
            continue
        working_exit = by_side.get(exit_side, 0.0)
        held = abs(held_quantity)
        if working_exit > held + QUANTITY_EPSILON:
            dropped.append((symbol, "working exits already exceed the live position"))
            continue

        headroom = max(0.0, held - working_exit - consumed.get(identity, 0.0))
        if headroom <= QUANTITY_EPSILON:
            dropped.append((symbol, "already fully covered by working exit orders"))
            continue
        if requested <= QUANTITY_EPSILON:
            dropped.append((symbol, "non-positive quantity"))
            continue

        if requested > headroom + QUANTITY_EPSILON:
            clamped.append((symbol, requested, headroom))
            order = order.model_copy(update={"quantity": headroom})
            requested = headroom

        consumed[identity] = consumed.get(identity, 0.0) + requested
        kept.append(order)

    return ReduceOnlyFilter(kept=kept, dropped=dropped, clamped=clamped)
