"""The account as the desk sees it: everything except what you pinned.

``/hold NVDA`` used to mean only "do not trade NVDA". The position still
counted toward the equity every order was sized against, and toward the
equity both kill switches measured. Two consequences, both of which bit:

  * the desk's position sizes moved when the operator's personal book
    moved, for no reason connected to the strategy;
  * on 2026-08-18 a CHF 3,677 drawdown, almost all of it in pinned
    personal names the desk was forbidden to touch, halted the desk and
    then refused the trailing-stop exits that would have protected the
    part it *was* responsible for.

A hold now means invisible. The held positions are removed from the
snapshot and their market value is taken out of equity, so the risk
manager sizes against — and halts on — the book it actually runs. The
cash stays: all of it is the desk's to deploy.

The inverse matters just as much. ``/unhold NVDA`` hands the position
back, and from that moment the portfolio manager is shown it as its own
and has to have a view: keep it, trim it, or exit. Nothing is liquidated
because a name is merely absent from a target book — the PM has to
actually decide, and the operator still approves the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading.core.types import AccountSnapshot, Position
from trading.core.valuation import position_value_base


@dataclass(frozen=True)
class ManagedView:
    """The desk's account, plus what had to be taken out to build it."""

    account: AccountSnapshot
    excluded: dict[str, float] = field(default_factory=dict)
    #: Held symbols whose value could not be converted to base currency.
    #: Their number was used as-is, which understates a USD position in a
    #: CHF account by roughly a fifth — the caller should say so.
    unconverted: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.excluded)

    @property
    def excluded_value(self) -> float:
        return sum(self.excluded.values())

    def note(self, currency: str) -> str:
        names = ", ".join(f"`{s}`" for s in sorted(self.excluded))
        line = (
            f"🔒 Pinned and invisible to the desk: {names} "
            f"— {currency} {self.excluded_value:,.0f} excluded from sizing and from "
            f"the kill switches. Desk equity {self.account.equity:,.0f}."
        )
        if self.unconverted:
            line += (
                f"\n⚠️ No FX rate for {', '.join(sorted(self.unconverted))} — "
                "their value was subtracted unconverted and the desk's equity "
                "is overstated. Check `broker.get_fx_rates()`."
            )
        return line


def _symbol_of(position: Position) -> str:
    return position.instrument.symbol.upper()


def managed_view(
    account: AccountSnapshot,
    held_symbols: set[str],
    *,
    fx_rates: dict[str, float] | None = None,
    last_prices: dict[str, float] | None = None,
) -> ManagedView:
    """Return the account with held positions removed and paid for.

    Never raises and never returns a nonsensical account: if the
    subtraction would take equity to zero or below — a pinned book worth
    more than the account, which should be impossible but would be
    catastrophic to size against — the original snapshot is returned
    untouched and nothing is marked excluded.
    """
    if not held_symbols or account.scope != "account":
        return ManagedView(account=account)

    wanted = {s.upper() for s in held_symbols}
    excluded: dict[str, float] = {}
    unconverted: list[str] = []
    keep: dict[str, Position] = {}

    for key, position in account.positions.items():
        symbol = _symbol_of(position)
        if symbol not in wanted:
            keep[key] = position
            continue
        value, clean = position_value_base(
            position,
            base_currency=account.base_currency,
            fx_rates=fx_rates,
            last_prices=last_prices,
        )
        excluded[symbol] = excluded.get(symbol, 0.0) + value
        if not clean:
            unconverted.append(symbol)

    if not excluded:
        return ManagedView(account=account)

    remaining = account.equity - sum(excluded.values())
    if remaining <= 0:
        # Refusing to build a view is the honest outcome. A zero or
        # negative sizing base would either produce no orders at all or,
        # worse, absurd ones; and it can only arise from bad data.
        return ManagedView(account=account)

    return ManagedView(
        account=account.model_copy(
            update={
                "equity": remaining,
                "positions": keep,
                "scope": "managed",
                "excluded_value": sum(excluded.values()),
                "excluded_symbols": tuple(sorted(excluded)),
            }
        ),
        excluded=excluded,
        unconverted=tuple(sorted(set(unconverted))),
    )
