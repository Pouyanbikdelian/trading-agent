"""Human-readable, currency-correct views of a pending cycle plan.

The runner owns the plan and the bot only reads it.  Keeping the small
presentation layer here gives both processes one interpretation of the exact
same JSON payload: an approval screen must never say that turnover is new
deployment, or make an exit look like a fresh short sale.
"""

from __future__ import annotations

from typing import Any


def _number(value: float) -> str:
    """Render shares without inventing precision the order does not have."""
    return f"{value:g}"


def _money(value: float, currency: str) -> str:
    return f"{currency} {value:,.0f}"


def _side(order: Any) -> str:
    return str(getattr(getattr(order, "side", ""), "value", getattr(order, "side", ""))).upper()


def build_plan(
    orders: list[Any],
    account: Any,
    last_prices: dict[str, float],
    *,
    fx_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a portable pending-plan payload from risk-approved orders.

    Every account-level value is converted into the account's base currency,
    just as ``RiskManager.signal_to_orders`` does.  The previous Telegram
    preview added USD order values and labelled them CHF on a CHF account,
    while also calling buy-plus-sell turnover "deployment".  This view keeps
    the two concepts explicit before an operator has to make a decision.
    """
    base_currency = str(getattr(account, "base_currency", "USD") or "USD").upper()
    equity = float(getattr(account, "equity", 0.0) or 0.0)
    positions = getattr(account, "positions", {}) or {}
    current_by_symbol = {
        str(position.instrument.symbol).upper(): float(position.quantity)
        for position in positions.values()
    }
    traded = {str(order.instrument.symbol).upper() for order in orders}
    rows: list[dict[str, Any]] = []
    gross = 0.0
    net = 0.0
    fx_fallback_symbols: list[str] = []

    for order in orders:
        instrument = order.instrument
        symbol = str(instrument.symbol).upper()
        side = _side(order)
        quantity = float(order.quantity)
        price = float(last_prices.get(instrument.key, 0.0) or 0.0)
        quote_currency = str(
            getattr(instrument, "currency", base_currency) or base_currency
        ).upper()
        rate = 1.0
        converted = True
        if quote_currency != base_currency:
            rate = float((fx_rates or {}).get(quote_currency, 0.0) or 0.0)
            if rate <= 0:
                # This mirrors the documented risk-manager fallback.  It is
                # visible in the payload so an operator is not misled into
                # believing the amount was currency-converted.
                rate = 1.0
                converted = False
                fx_fallback_symbols.append(symbol)
        notional = abs(quantity * price * rate)
        gross += notional
        if side == "BUY":
            net += notional
        else:
            net -= notional
        current = current_by_symbol.get(symbol, 0.0)
        if side == "BUY":
            context = "new position" if abs(current) < 1e-9 else f"adds to {_number(current)} held"
        else:
            context = (
                f"closes {_number(current)} held"
                if current > 0 and quantity >= current - 1e-9
                else f"reduces {_number(current)} held"
            )
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "current_quantity": current,
                "context": context,
                "quote_currency": quote_currency,
                "price": price,
                "notional_base": round(notional, 2),
                "converted_to_base": converted,
            }
        )

    return {
        "base_currency": base_currency,
        "equity": round(equity, 2),
        "order_count": len(rows),
        "gross_turnover": round(gross, 2),
        "turnover_pct": round(gross / equity * 100.0, 2) if equity > 0 else 0.0,
        "net_cash_change": round(net, 2),
        "orders": rows,
        "unchanged_symbols": sorted(symbol for symbol in current_by_symbol if symbol not in traded),
        "fx_fallback_symbols": sorted(set(fx_fallback_symbols)),
    }


def format_trade_review(
    pending: dict[str, Any],
    *,
    final_confirmation: bool = False,
    include_controls: bool = False,
    heading: str | None = None,
) -> str:
    """Render the exact plan in decision order, not implementation order."""
    raw_plan = pending.get("plan")
    plan: dict[str, Any] = raw_plan if isinstance(raw_plan, dict) else pending
    rows = plan.get("orders") if isinstance(plan.get("orders"), list) else []
    if not rows:
        return "_No readable orders are attached to this pending cycle._"

    ccy = str(plan.get("base_currency") or pending.get("currency") or "USD")
    order_count = int(plan.get("order_count") or pending.get("n_orders") or len(rows))
    gross = float(plan.get("gross_turnover") or pending.get("gross") or 0.0)
    turnover_pct = float(plan.get("turnover_pct") or pending.get("deploy_pct") or 0.0)
    net = float(plan.get("net_cash_change") or 0.0)
    cycle_id = str(pending.get("id") or "")[:8]
    title = heading or (
        "🔎 *Revised trade review* — final confirmation required"
        if final_confirmation
        else "🧾 *Trade review* — approval required"
    )
    lines = [
        f"{title} — cycle `{cycle_id}`",
        f"{order_count} changes · {_money(gross, ccy)} turnover ({turnover_pct:.1f}% of equity)",
    ]
    if net > 0:
        lines.append(f"Net cash used: {_money(net, ccy)}")
    elif net < 0:
        lines.append(f"Net cash released: {_money(abs(net), ccy)}")
    else:
        lines.append("Net cash change: roughly zero")
    lines.append("_Turnover is total buys + sells, not additional exposure._")

    buys = [row for row in rows if str(row.get("side", "")).upper() == "BUY"]
    sells = [row for row in rows if str(row.get("side", "")).upper() == "SELL"]
    if buys:
        lines += ["", "*Buy / increase*"]
        for row in buys:
            lines.append(
                f"  `{row['symbol']}` buy {_number(float(row['quantity']))} · "
                f"{_money(float(row.get('notional_base', 0.0)), ccy)} "
                f"_({row.get('context', 'position change')})_"
            )
    if sells:
        lines += ["", "*Sell / reduce*"]
        for row in sells:
            lines.append(
                f"  `{row['symbol']}` sell {_number(float(row['quantity']))} · "
                f"{_money(float(row.get('notional_base', 0.0)), ccy)} "
                f"_({row.get('context', 'position change')})_"
            )

    unchanged = [str(symbol).upper() for symbol in plan.get("unchanged_symbols", [])]
    if unchanged:
        lines.append(f"\n_Unchanged: {', '.join(unchanged[:8])}_")
    fallback = [str(symbol).upper() for symbol in plan.get("fx_fallback_symbols", [])]
    if fallback:
        lines.append(
            f"⚠️ FX rate unavailable for {', '.join(fallback)}; shown at a 1:1 fallback used by risk."
        )
    if include_controls:
        lines += [
            "",
            "✅ Approve as shown · ❌ Reject",
            "Edit: `/approve only AMD DD` · `/approve all except XLV`",
            "`/proposal` repeats this plan · `/candidates` shows alternate ranked picks.",
        ]
    return "\n".join(lines)


def format_candidates(pending: dict[str, Any], *, limit: int = 20) -> str:
    """Show the optional replacement basket only when the operator asks."""
    candidates = pending.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "_No alternate candidate ladder is available for this cycle._"
    lines = ["🏆 *Candidate ladder* — alternatives, not the proposed trade"]
    for rank, candidate in enumerate(candidates[:limit], start=1):
        if not isinstance(candidate, dict):
            continue
        symbol = str(candidate.get("symbol") or "?").upper()
        score = float(candidate.get("score") or 0.0)
        mark = " · in current plan" if candidate.get("in_basket") else ""
        lines.append(f"  `{rank:>2}` `{symbol:<6}` {score:+.1%}{mark}")
    if pending.get("can_pick"):
        lines += ["", "`/pick 1 3 5` replaces the proposed basket; it does not edit it."]
    return "\n".join(lines)
