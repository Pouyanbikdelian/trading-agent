"""What a position is worth, in a currency you can actually add up.

IBKR reports ``averageCost`` and ``unrealizedPNL`` in the CONTRACT's
currency — dollars for a US share — while ``NetLiquidation`` and
``TotalCashValue`` come back in the account's base currency. Nothing in
this codebase converted between them, so every place that summed
position values and compared them to equity was adding dollars to
francs.

It showed on 2026-08-21: ``/positions`` reported six holdings totalling
"CHF 35,052" against equity of CHF 84,312 and cash of CHF 56,225 — which
is CHF 7k more than the account contains. The positions were USD, the
header said CHF, and every weight in that table was 25% too large.

Cosmetic while nothing depended on it. Not cosmetic once a held position
has to be subtracted from equity: a 25% error there is a 25% error in
the sizing base for every order the desk sends.
"""

from __future__ import annotations

from trading.core.types import Position


def position_value_native(position: Position) -> float:
    """Market value in the instrument's OWN currency.

    ``quantity * avg_price`` is cost; adding unrealised P&L walks it to
    the current mark. This is the arithmetic the reporting code has
    always used — it was only ever the currency that was wrong.
    """
    return float(position.quantity) * float(position.avg_price) + float(position.unrealized_pnl)


def position_value_base(
    position: Position,
    *,
    base_currency: str,
    fx_rates: dict[str, float] | None = None,
    last_prices: dict[str, float] | None = None,
) -> tuple[float, bool]:
    """``(value_in_base_currency, converted_cleanly)``.

    ``last_prices`` is preferred when it carries the instrument, because
    a live mark beats cost-plus-unrealised — the two agree in principle
    and the mark is fresher.

    The second element is False when the value had to be passed through
    at 1.0 for want of an FX rate. Callers about to size real orders off
    this number are expected to look at it and say so, rather than
    quietly treating dollars as francs the way the old code did.
    """
    currency = position.instrument.currency or base_currency

    native: float | None = None
    if last_prices is not None:
        price = float(last_prices.get(position.instrument.key, 0.0) or 0.0)
        if price > 0:
            native = float(position.quantity) * price
    if native is None:
        native = position_value_native(position)

    if currency == base_currency:
        return native, True

    rate = float((fx_rates or {}).get(currency, 0.0) or 0.0)
    if rate <= 0:
        return native, False
    return native * rate, True
