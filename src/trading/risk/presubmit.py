"""The long-only invariant, on the paths that do not go through sizing.

``RiskManager.signal_to_orders`` nets working orders into the position
before computing deltas, and clamps sells so they can never cross zero.
That fix landed 2026-07-15, after three unfilled batches stacked at the
open and took MU and SNDK short. It covers the cycle.

It does not cover the command pipeline. ``/sell all``, ``/close`` and
``/flatten`` size off ``broker.get_positions()`` — settled shares — and
call ``submit_order`` directly. Orders already working at the broker are
invisible to them. The same double-sell is therefore still reachable,
and the guards make it likely rather than theoretical: they run every 15
minutes through 20:59 UTC, so the last three slots of the day place
market orders that queue to the next open. An operator who sees
"🛑 Trailing stop VST — closing" and reaches for ``/close VST`` to be
sure has just sold the position twice.

The execution lock does not help here. It serialises *submissions*; it
says nothing about an order submitted an hour ago that is still working.

So the invariant gets enforced once more, at the last point before the
broker, against the same ground truth the cycle uses — IBKR's own
``openTrades``.

``/flatten`` is clamped too. It is the panic button and it stays
unconditional in every other respect, but a panic button that can open
a short position is not an exit.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from trading.core.logging import logger


class SellableResult(NamedTuple):
    """How much of ``symbol`` may still be sold, and why."""

    sellable: float
    settled: float
    working_sells: float
    orders_known: bool

    @property
    def blocked(self) -> bool:
        return self.sellable <= 0


def _norm(symbol: str) -> str:
    return str(symbol).upper()


def sellable_quantity(broker: Any, symbol: str) -> SellableResult:
    """Shares of ``symbol`` that can be sold without crossing zero.

    ``settled long − already-working sells``. Buys in flight are
    deliberately NOT counted as owned: an unfilled buy is not ours to
    sell, and counting it is how a sell fires ahead of its buy and ends
    short (the same reasoning as ``pending_sell`` in the risk manager).

    A broker that cannot report open orders degrades to settled-only
    with ``orders_known=False``. Callers decide what to do with that —
    this function does not guess.
    """
    sym = _norm(symbol)

    settled = 0.0
    for p in broker.get_positions() or []:
        if _norm(p.instrument.symbol) == sym:
            settled += float(p.quantity)

    working_sells = 0.0
    orders_known = True
    try:
        for o in broker.get_open_orders() or []:
            if _norm(o.instrument.symbol) != sym:
                continue
            side = getattr(o.side, "value", o.side)
            if str(side).upper() == "SELL":
                working_sells += float(o.quantity)
    except Exception as e:
        orders_known = False
        logger.bind(component="presubmit").warning(
            f"get_open_orders failed for {sym} ({type(e).__name__}) — "
            "clamping against settled shares only; a working sell would be invisible"
        )

    sellable = max(0.0, max(settled, 0.0) - working_sells)
    return SellableResult(sellable, settled, working_sells, orders_known)


def clamp_sell(broker: Any, symbol: str, requested: float) -> tuple[float, str | None]:
    """Clamp a sell to what can be sold without going short.

    Returns ``(quantity, note)``. ``note`` is None when nothing was
    changed; otherwise it is operator-facing text explaining the clamp,
    which matters more than the clamp itself — a sell that silently
    shrinks looks like a partial fill.

    Raises ``ValueError`` when nothing is sellable, so the caller reports
    a refused command rather than submitting a zero-quantity order.
    """
    r = sellable_quantity(broker, symbol)
    sym = _norm(symbol)

    if r.settled <= 0:
        raise ValueError(f"no long position in {sym} to sell")

    if r.blocked:
        raise ValueError(
            f"{sym}: a sell for {r.working_sells:g} share(s) is already working at the "
            f"broker and covers the whole {r.settled:g}-share position. Selling again "
            "would go short. Cancel the working order first (/orders, /cancel) if you "
            "want to re-issue it."
        )

    if requested <= r.sellable:
        return requested, None

    note = (
        f"{sym}: sell clamped {requested:g} → {r.sellable:g} "
        f"({r.settled:g} held − {r.working_sells:g} already working)"
    )
    if not r.orders_known:
        note = f"{sym}: sell clamped {requested:g} → {r.sellable:g} (held {r.settled:g}; "
        note += "working orders UNAVAILABLE — a queued sell would not have been seen)"
    logger.bind(component="presubmit").warning(note)
    return r.sellable, note
