"""Pre-cycle broker readiness — ask for the human BEFORE it matters.

IBKR requires two-factor authentication for every user; there is no
API-only exemption. The gateway re-authenticates after IBKR's daily
auto-restart and its mandatory weekly shutdown, and each of those can
raise an IBKR Mobile prompt. IBC retries a missed prompt rather than
dying, so a missed tap is recoverable — but only if somebody taps.

Every other consequence of a dead gateway is non-fatal: a monitor
degrades, a snapshot is skipped, the dashboard shows stale numbers. The
ONE moment it matters is the cycle, because a cycle that cannot reach
the broker submits nothing, and a silently skipped rebalance looks
exactly like a rebalance that decided to do nothing.

So this runs an hour ahead of the cycle and, if the broker is not
usable, sends one Telegram message while there is still time to act.
That converts "notice on Monday that Friday never traded" into "tap
your phone at 20:05".

Deliberately a REAL broker call, not a TCP probe. The gateway's port
accepts connections the whole time it is sitting at a login screen —
`docker compose ps` says healthy and nothing works. Only asking for the
account proves the session is authenticated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger


def check_broker_ready(broker: Any) -> dict[str, Any]:
    """Is the broker usable right now? Never raises.

    ``{"ready": bool, "detail": str, "equity": float | None}``.
    """
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    try:
        snap = broker.get_account()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"[:200]
        logger.bind(component="broker_ready").warning(f"broker not ready: {detail}")
        return {"ready": False, "detail": detail, "equity": None, "checked_at": now}

    equity = getattr(snap, "equity", None)
    if equity is None:
        # A snapshot with no equity means the session answered but has no
        # account data — the shape a half-authenticated gateway returns.
        return {
            "ready": False,
            "detail": "broker returned a snapshot with no equity — session may be half-open",
            "equity": None,
            "checked_at": now,
        }
    return {"ready": True, "detail": "ok", "equity": float(equity), "checked_at": now}


def format_not_ready_alert(result: dict[str, Any], *, minutes_to_cycle: int) -> str:
    """The message that has to make someone pick up their phone.

    Leads with the deadline, because the only thing that matters is how
    long there is left to fix it.
    """
    return (
        f"🔴 *Broker not ready — cycle in {minutes_to_cycle} min*\n"
        f"`{result.get('detail', 'unknown')}`\n\n"
        "The gateway is not answering with account data. Most likely it is "
        "waiting on an IBKR Mobile 2FA prompt after a restart.\n\n"
        "*Do now:*\n"
        "1. Open IBKR Mobile and approve any pending login.\n"
        "2. If there is no prompt, force a fresh login:\n"
        "   `docker compose restart ib-gateway` — then watch for the prompt.\n"
        "3. Re-check with `/health`.\n\n"
        "_If this is not fixed, the cycle will submit nothing and say so._"
    )


#: The currency US equities settle in. The account's BASE currency is CHF;
#: these are not the same thing, and the gap is the whole problem below.
QUOTE_CCY = "USD"

#: Headroom on the funding estimate. Prices move between this check and
#: the cycle, and a basket sized at exactly the cash available leaves no
#: room for the last order. Cheap insurance against a near-miss.
FUNDING_BUFFER = 1.05


def check_trade_currency_funding(
    broker: Any,
    *,
    gross_exposure_pct: float,
    quote_ccy: str = QUOTE_CCY,
) -> dict[str, Any]:
    """Is there enough USD to fund the basket the cycle is about to size?

    The account is CHF-based; US equities settle in USD. Buying them from
    a CHF balance creates a USD debit — i.e. borrowing. With
    ``max_margin_borrowing_pct = 0.0`` (the default, and the right setting
    for a cash account) the risk manager REJECTS any order that pushes a
    currency's cash below zero.

    The failure mode that makes this worth a scheduled check: the cycle
    runs, proposes a full basket, the risk manager refuses all of it, and
    the result is a completed cycle that bought nothing. In a position
    report that is indistinguishable from a strategy that saw no
    opportunities. Paper never showed it, because paper has accumulated
    USD over months of trading.

    Checked an hour ahead so there is still time to convert. Never raises.
    """
    out: dict[str, Any] = {"ok": True, "reason": "ok", "quote_ccy": quote_ccy}
    try:
        snap = broker.get_account()
        equity = float(getattr(snap, "equity", 0.0) or 0.0)
    except Exception as e:
        return {**out, "ok": True, "reason": f"skipped: account unavailable ({type(e).__name__})"}

    if equity <= 0:
        return {**out, "reason": "skipped: no equity"}

    balances: dict[str, float] = {}
    try:
        balances = {k: float(v) for k, v in (broker.get_balances() or {}).items()}
    except Exception as e:
        # Degrade to "cannot tell" rather than to a false alarm. A weekly
        # cry-wolf alert is worse than no alert.
        return {**out, "reason": f"skipped: balances unavailable ({type(e).__name__})"}

    if quote_ccy not in balances:
        return {**out, "reason": f"skipped: broker reports no {quote_ccy} balance line"}

    rates: dict[str, float] = {}
    try:
        rates = {k: float(v) for k, v in (broker.get_fx_rates() or {}).items()}
    except Exception:
        rates = {}
    # Rates are quoted as "one unit of X in base currency". Absent a rate
    # we assume parity, which OVERSTATES nothing and understates the USD
    # need only if USD is worth less than base — acceptable for a warning.
    rate = rates.get(quote_ccy) or 1.0

    need_base = equity * max(0.0, gross_exposure_pct) * FUNDING_BUFFER
    need_quote = need_base / rate if rate else need_base
    have_quote = balances.get(quote_ccy, 0.0)

    out.update(
        {
            "equity": equity,
            "required": need_quote,
            "available": have_quote,
            "shortfall": max(0.0, need_quote - have_quote),
            "rate": rate,
            "balances": balances,
        }
    )
    if have_quote + 1e-9 < need_quote:
        out["ok"] = False
        out["reason"] = (
            f"{quote_ccy} cash {have_quote:,.0f} < {need_quote:,.0f} needed "
            f"for {gross_exposure_pct:.1%} gross"
        )
        logger.bind(component="broker_ready").warning(f"funding shortfall: {out['reason']}")
    return out


def format_funding_alert(result: dict[str, Any], *, minutes_to_cycle: int) -> str:
    """Names the number to convert, because 'insufficient USD' without an
    amount still leaves the operator doing arithmetic under time pressure."""
    ccy = result.get("quote_ccy", QUOTE_CCY)
    short = result.get("shortfall", 0.0)
    have = result.get("available", 0.0)
    need = result.get("required", 0.0)
    return (
        f"🟠 *Not enough {ccy} to fund the basket — cycle in {minutes_to_cycle} min*\n"
        f"have `{have:,.0f}` · need `{need:,.0f}` · short `{short:,.0f}` {ccy}\n\n"
        f"The account is CHF-based and US equities settle in {ccy}. With margin "
        "borrowing set to 0, the risk manager will REJECT the whole basket "
        "rather than borrow — the cycle will complete having bought nothing, "
        "which looks exactly like a strategy that saw nothing worth buying.\n\n"
        f"*Do now:* convert about `{short * 1.02:,.0f}` CHF to {ccy} in Client "
        "Portal (or raise MAX_MARGIN_BORROWING_PCT if you intend to borrow)."
    )


#: Where an operator's acknowledgement of the account's pre-existing
#: positions is recorded. Keyed by symbol set, so a position opened after
#: the acknowledgement re-raises the question rather than inheriting it.
UNHELD_ACK_FILENAME = "unheld_ack.json"


def check_unheld_positions(broker: Any, *, state_dir: Path | str) -> dict[str, Any]:
    """Which positions in this account is nothing protecting?

    Two independent subsystems can sell a position the operator opened by
    hand, and on 2026-08-07 both were live and neither was understood:

    * the **rebalance** sells anything whose target weight is zero, and
      every symbol the strategy did not pick has a target weight of zero;
    * the **position guards** (``GUARDS_ENABLED``) trail a stop on every
      position in the account regardless of who opened it. On paper they
      only ever saw system positions, so nothing had made them visible.
      They sold 63 shares of VST seven minutes into the first live
      session.

    ``/hold`` is the only thing that blocks BOTH paths — disabling the
    guards still leaves the rebalance. So the question worth asking
    before arming is not "are the guards on" but "is there anything in
    this account that neither ``holds.json`` nor an acknowledgement
    covers".

    Never raises; an unreachable broker degrades to ``ok=True`` with a
    reason, because a check that blocks startup on a network blip is a
    check that gets removed.
    """
    from trading.runner.holds import load_holds

    state_dir = Path(state_dir)
    out: dict[str, Any] = {"ok": True, "unheld": [], "held": [], "reason": "ok"}
    try:
        positions = broker.get_positions() or []
    except Exception as e:
        return {**out, "reason": f"skipped: positions unavailable ({type(e).__name__})"}

    symbols = {
        str(p.instrument.symbol).upper()
        for p in positions
        if abs(float(getattr(p, "quantity", 0.0) or 0.0)) > 0
    }
    held = load_holds(state_dir)
    acked = _load_unheld_ack(state_dir)
    unheld = sorted(symbols - held - acked)

    out.update({"held": sorted(held & symbols), "acked": sorted(acked & symbols)})
    if unheld:
        out["ok"] = False
        out["unheld"] = unheld
        out["reason"] = f"{len(unheld)} position(s) neither held nor acknowledged: " + ", ".join(
            unheld
        )
        logger.bind(component="broker_ready").warning(out["reason"])
    return out


def _load_unheld_ack(state_dir: Path) -> set[str]:
    path = Path(state_dir) / UNHELD_ACK_FILENAME
    if not path.exists():
        return set()
    try:
        import json

        payload = json.loads(path.read_text())
        return {str(s).upper() for s in payload.get("symbols", [])}
    except Exception:
        return set()


def save_unheld_ack(state_dir: Path | str, symbols: set[str]) -> Path:
    """Record that the operator has seen these positions and accepts that
    the system may trade them. Replaces, never merges — an acknowledgement
    that silently accumulated symbols would defeat the purpose."""
    import json
    import os
    import tempfile

    path = Path(state_dir) / UNHELD_ACK_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(
            {
                "symbols": sorted(s.upper() for s in symbols),
                "acked_at": datetime.now(tz=timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    os.replace(tmp, path)
    return path


def format_unheld_alert(result: dict[str, Any]) -> str:
    """The message that would have saved VST."""
    names = ", ".join(result.get("unheld", []))
    return (
        "🔴 *Unprotected positions in the live account*\n"
        f"`{names}`\n\n"
        "Nothing is protecting these. Two separate subsystems can sell them:\n"
        "• the *rebalance* — any symbol the strategy did not pick has a target "
        "weight of zero, and a zero target is a sell;\n"
        "• the *position guards* — ATR trailing stops apply to every position "
        "in the account, not only the ones the system opened.\n\n"
        "`GUARDS_ENABLED=false` only closes the second path. `/hold SYMBOL` "
        "closes both — it is the only thing that does.\n\n"
        "*Do now:* `/hold` each one, or run `trading preflight ack` to accept "
        "that the system may trade them."
    )


def format_ready_note(result: dict[str, Any], *, minutes_to_cycle: int) -> str:
    """Quiet confirmation. Sent only when it had previously failed —
    a green message every week is a message nobody reads."""
    eq = result.get("equity")
    return f"🟢 Broker ready — cycle in {minutes_to_cycle} min" + (
        f" · equity {eq:,.0f}" if isinstance(eq, (int, float)) else ""
    )
