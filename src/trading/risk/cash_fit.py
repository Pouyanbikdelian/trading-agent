"""Trim a basket to the cash that actually exists, instead of refusing it.

A CHF-base account buying US stocks holds two piles of cash, and only
the USD pile can pay for a USD share. Until now the no-margin check
answered an overdraft with all-or-nothing: one currency short by a
franc and the *entire* basket — sells included — was dropped, with a
message telling the operator to go and convert money by hand.

On 2026-08-21 that produced the obvious dead end. The sleeve had just
been raised, the PM asked for a USD 48.5k book, the account held USD
39.0k, and the cycle refused all eleven names over a USD 9,542 gap.
Nothing was wrong with the plan; it was simply 20% too big for the
wallet. A human handed the same instruction would have bought 80% of
each line and moved on.

That is all this module does:

  * Only BUY orders shrink. Sells raise cash and can never cause an
    overdraft, so they pass through untouched — which also means a
    basket that exists to *reduce* exposure is never blocked by a
    shortage of the currency it is about to produce.
  * Each currency is fitted independently against its own cash. A USD
    shortfall must not shrink an EUR order that was always affordable.
  * Every line scales by the same factor, so the portfolio manager's
    relative conviction survives. A 12% name stays three times a 4%
    name; the book is smaller, not re-opinionated.
  * Whole-share quantities are floored, never rounded. Rounding 4.6
    shares up to 5 is how a "fit" re-breaches the very limit it was
    supposed to respect, so the arithmetic here only ever frees cash.
    Lines that floor to zero drop out and are named.

The result is re-checked by the caller. This module is deliberately
not the thing that decides whether the basket is fundable — it only
makes it smaller, and the original check still has the last word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from trading.core.types import AssetClass, Order, Side

#: Marks the decisions this module produces so the cycle can lift them
#: onto the operator's card rather than burying them in the log.
REASON_PREFIX = "cash fit"

#: Quantities below this are nothing.
_EPS_QTY = 1e-9

#: Fraction of the affordable budget deliberately left unspent.
#:
#: The trim is computed when the basket is planned; the orders are MARKET
#: orders that may not reach the broker for another ten minutes if an
#: operator is reading an approval card. Sizing to exactly 100% of cash
#: means any upward tick in between overdraws the account — and an
#: overdraw is not a rounding error here, it is IBKR silently opening a
#: margin loan, which is the precise thing MAX_MARGIN_BORROWING_PCT=0
#: exists to prevent. One percent buys that slack back.
CASH_SAFETY_MARGIN = 0.01


@dataclass(frozen=True)
class CashFit:
    """What the trim did, in enough detail to explain it to a human."""

    orders: list[Order]
    #: currency -> the factor its buys were multiplied by (always < 1.0)
    scaled: dict[str, float] = field(default_factory=dict)
    #: (symbol, currency) for buys that floored to zero shares
    dropped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.scaled)

    def reasons(self) -> list[str]:
        """One human-readable line per currency that was trimmed."""
        out: list[str] = []
        for ccy in sorted(self.scaled):
            gone = sorted(sym for sym, c in self.dropped if c == ccy)
            line = f"{REASON_PREFIX}: {ccy} buys trimmed to {self.scaled[ccy]:.0%} of target"
            if gone:
                line += f" — dropped {', '.join(gone)} (below one share)"
            out.append(line)
        return out


def _is_whole_share(order: Order) -> bool:
    return order.instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF)


def fit_orders_to_cash(
    orders: list[Order],
    *,
    cash_by_currency: dict[str, float],
    base_currency: str,
    last_prices: dict[str, float],
    allowed_debit: float = 0.0,
    safety_margin: float = CASH_SAFETY_MARGIN,
) -> CashFit:
    """Shrink BUY orders until each currency's cash covers its own buys.

    ``allowed_debit`` is the permitted overdraft per currency, matching
    the convention of the no-margin check this feeds: 0.0 is strict
    cash-account behaviour.

    ``safety_margin`` is the fraction of the budget left deliberately
    unspent, so a price that moves between planning and fill does not
    turn a fitted basket into an overdraft.

    Orders with no usable price are left alone. They are invisible to
    the no-margin check too, so shrinking them would be guesswork that
    changes nothing downstream.
    """
    spendable = 1.0 - min(max(safety_margin, 0.0), 1.0)
    wanted: dict[str, float] = {}
    available: dict[str, float] = {}

    for order in orders:
        price = float(last_prices.get(order.instrument.key, 0.0) or 0.0)
        if price <= 0:
            continue
        ccy = order.instrument.currency or base_currency
        notional = float(order.quantity) * price
        if order.side == Side.BUY:
            wanted[ccy] = wanted.get(ccy, 0.0) + notional
        else:
            # A sell settles into the currency it was traded in, which is
            # exactly the currency its proceeds may then fund.
            available[ccy] = available.get(ccy, 0.0) + notional

    factors: dict[str, float] = {}
    for ccy, need in wanted.items():
        if need <= 0:
            continue
        budget = cash_by_currency.get(ccy, 0.0) + available.get(ccy, 0.0) + max(allowed_debit, 0.0)
        # One rule, used for both the question and the answer: the money
        # this basket may spend is the budget less the safety margin. A
        # basket that fits only by spending the very last franc is the
        # same risk as one that does not fit, so it is trimmed too.
        limit = max(budget, 0.0) * spendable
        if need <= limit:
            continue
        factors[ccy] = limit / need

    if not factors:
        return CashFit(orders=list(orders))

    kept: list[Order] = []
    dropped: list[tuple[str, str]] = []
    for order in orders:
        ccy = order.instrument.currency or base_currency
        factor = factors.get(ccy)
        price = float(last_prices.get(order.instrument.key, 0.0) or 0.0)
        if order.side != Side.BUY or factor is None or price <= 0:
            kept.append(order)
            continue

        quantity = float(order.quantity) * factor
        if _is_whole_share(order):
            # Floor, never round. See the module docstring: rounding up
            # is how a fit re-creates the overdraft it just removed.
            quantity = float(math.floor(quantity))
        if quantity < _EPS_QTY:
            dropped.append((order.instrument.symbol, ccy))
            continue
        kept.append(order.model_copy(update={"quantity": quantity}))

    return CashFit(orders=kept, scaled=factors, dropped=dropped)
