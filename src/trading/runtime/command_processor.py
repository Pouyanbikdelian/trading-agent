r"""Processes commands queued by the Telegram bot.

The runner schedules ``process_pending`` every few seconds. It picks
up pending command files, executes them against the broker, and
broadcasts a result alert to Telegram. The bot writes commands and
forgets — all execution + user-facing notifications flow from the
runner.

This keeps all broker interactions on a single thread (the runner's
worker pool) so we never race the cycle. Latency cost: 0-5 seconds
between bot ``/buy`` and execution, which is invisible at the cadence
we operate at.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger
from trading.core.types import AssetClass, Instrument, Order, OrderType, Side, TimeInForce
from trading.execution.base import Broker
from trading.risk.manager import RiskManager
from trading.runner.alerts import TelegramAlerts
from trading.runtime.commands import (
    Command,
    CommandType,
    mark_executed,
    mark_running,
    pending_commands,
)

# Commands that submit orders to the broker. These MUST be gated by the
# risk manager's halt state — otherwise an operator who has typed /halt
# can still trade via /buy, /sell, /close, /flatten, /fx. Audit (May 2026)
# flagged this as a critical bypass. Non-order commands (cancel, refresh,
# reconnect) are still allowed during halt — they're recovery actions.
_ORDER_SUBMITTING_COMMANDS = {
    CommandType.BUY,
    CommandType.SELL,
    CommandType.CLOSE,
    CommandType.FLATTEN,
    CommandType.FX_CONVERT,
}


def _short_id(uuid_str: str) -> str:
    return uuid_str[:8]


def process_pending(
    broker: Broker,
    state_dir: Path,
    alerts: TelegramAlerts,
    *,
    risk_manager: RiskManager | None = None,
) -> int:
    r"""Pick up all pending commands, execute them, and alert the operator.

    Returns the number of commands processed (useful for tests + logs).
    Never raises — each command's failure is captured and reported.

    The ``risk_manager`` argument is optional for backward compatibility
    with existing tests, but the runner always passes it in production so
    the halt gate is enforced on manual orders. When omitted, the halt
    gate is skipped (callers should explicitly pass ``None`` only in
    tests that don't exercise the halt-bypass path).
    """
    cmds = pending_commands(state_dir)
    if not cmds:
        return 0
    logger.bind(component="command_processor").info(f"processing {len(cmds)} pending command(s)")
    for cmd in cmds:
        try:
            mark_running(cmd, state_dir)
        except FileNotFoundError:
            # Another watcher beat us to it — skip.
            continue
        _execute_one(cmd, broker, state_dir, alerts, risk_manager=risk_manager)
    return len(cmds)


def _execute_one(
    cmd: Command,
    broker: Broker,
    state_dir: Path,
    alerts: TelegramAlerts,
    *,
    risk_manager: RiskManager | None = None,
) -> None:
    handler = _HANDLERS.get(cmd.type)
    if handler is None:
        msg = f"unknown command type `{cmd.type.value}`"
        mark_executed(cmd, state_dir, status="error", result=msg)
        alerts.error(f"❌ command `{_short_id(cmd.id)}` rejected: {msg}")
        return

    # Halt gate: refuse to submit orders while the risk manager is halted.
    # Operator must /resume before manual trading resumes.
    #
    # Reload halt.json before the check — the Telegram bot writes that
    # file from a separate process, so the risk manager's in-memory
    # state can be stale. Without this reload, a /resume from Telegram
    # wouldn't unblock manual orders until the next cycle ran (the same
    # bug that hit evaluate_intraday, fixed in commit a914db2).
    if risk_manager is not None:
        try:
            risk_manager._reload_halt_state()
        except Exception:
            logger.bind(component="command_processor").exception(
                "halt state reload failed; falling back to in-memory state"
            )
    if (
        risk_manager is not None
        and risk_manager.is_halted()
        and cmd.type in _ORDER_SUBMITTING_COMMANDS
    ):
        reason = getattr(risk_manager._state, "reason", "") or "halted"
        msg = f"refused — risk manager halted: {reason}. /resume first."
        logger.bind(component="command_processor").warning(
            f"halt gate blocked {cmd.type.value} command {cmd.id}"
        )
        mark_executed(cmd, state_dir, status="error", result=msg)
        alerts.error(f"🛑 `{_short_id(cmd.id)}` {cmd.type.value} {msg}")
        return

    try:
        result = handler(cmd, broker)
        mark_executed(cmd, state_dir, status="ok", result=result)
        alerts.info(_format_success(cmd, result))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.bind(component="command_processor").exception(f"command {cmd.type.value} failed")
        mark_executed(cmd, state_dir, status="error", result=err)
        alerts.error(_format_failure(cmd, err))


# ---------------------------------------------------------------------------
# Handlers — one per CommandType
# ---------------------------------------------------------------------------


def _h_buy(cmd: Command, broker: Broker) -> dict[str, Any]:
    sym = str(cmd.args["symbol"]).upper()
    qty = float(cmd.args["qty"])
    limit = cmd.args.get("limit")
    instrument = Instrument(symbol=sym, asset_class=AssetClass.EQUITY)
    order = Order(
        client_order_id=f"manual-{_short_id(cmd.id)}",
        instrument=instrument,
        side=Side.BUY,
        quantity=qty,
        order_type=OrderType.LIMIT if limit else OrderType.MARKET,
        limit_price=float(limit) if limit else None,
        tif=TimeInForce.DAY,
        created_at=datetime.now(tz=timezone.utc),
    )
    broker.submit_order(order)
    return {
        "symbol": sym,
        "qty": qty,
        "side": "BUY",
        "type": order.order_type.value,
        "limit": limit,
        "client_order_id": order.client_order_id,
    }


def _h_sell(cmd: Command, broker: Broker) -> dict[str, Any]:
    sym = str(cmd.args["symbol"]).upper()
    qty_arg = cmd.args.get("qty", "all")
    limit = cmd.args.get("limit")

    # "all" → close the full position
    if str(qty_arg).lower() == "all":
        positions = broker.get_positions()
        pos = next((p for p in positions if p.instrument.symbol == sym), None)
        if pos is None or pos.quantity == 0:
            raise ValueError(f"no open position in {sym}")
        qty = abs(pos.quantity)
    else:
        qty = float(qty_arg)

    instrument = Instrument(symbol=sym, asset_class=AssetClass.EQUITY)
    order = Order(
        client_order_id=f"manual-{_short_id(cmd.id)}",
        instrument=instrument,
        side=Side.SELL,
        quantity=qty,
        order_type=OrderType.LIMIT if limit else OrderType.MARKET,
        limit_price=float(limit) if limit else None,
        tif=TimeInForce.DAY,
        created_at=datetime.now(tz=timezone.utc),
    )
    broker.submit_order(order)
    return {
        "symbol": sym,
        "qty": qty,
        "side": "SELL",
        "type": order.order_type.value,
        "limit": limit,
        "client_order_id": order.client_order_id,
    }


def _h_close(cmd: Command, broker: Broker) -> dict[str, Any]:
    # Equivalent to sell "all"
    return _h_sell(
        Command(
            id=cmd.id,
            type=CommandType.SELL,
            args={**cmd.args, "qty": "all"},
            requested_by=cmd.requested_by,
            requested_at=cmd.requested_at,
        ),
        broker,
    )


def _h_flatten(_cmd: Command, broker: Broker) -> dict[str, Any]:
    closed: list[str] = []
    skipped: list[str] = []
    for pos in broker.get_positions():
        if pos.quantity == 0:
            continue
        side = Side.SELL if pos.quantity > 0 else Side.BUY
        order = Order(
            client_order_id=f"flatten-{pos.instrument.symbol}-{_short_id('x' * 8)}",
            instrument=pos.instrument,
            side=side,
            quantity=abs(pos.quantity),
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            created_at=datetime.now(tz=timezone.utc),
        )
        try:
            broker.submit_order(order)
            closed.append(pos.instrument.symbol)
        except Exception as e:
            skipped.append(f"{pos.instrument.symbol} ({type(e).__name__})")
    return {"closed": closed, "skipped": skipped, "n_closed": len(closed)}


def _h_cancel_order(cmd: Command, broker: Broker) -> dict[str, Any]:
    coid = str(cmd.args["client_order_id"])
    broker.cancel_order(coid)
    return {"client_order_id": coid, "cancelled": True}


def _h_fx_convert(cmd: Command, broker: Broker) -> dict[str, Any]:
    from_ccy = str(cmd.args["from_ccy"]).upper()
    to_ccy = str(cmd.args["to_ccy"]).upper()
    amount = float(cmd.args["amount"])
    if not hasattr(broker, "convert_currency"):
        raise RuntimeError("this broker doesn't support FX conversion")
    return broker.convert_currency(  # type: ignore[attr-defined]
        from_ccy=from_ccy, to_ccy=to_ccy, from_amount=amount
    )


def _h_refresh_data(_cmd: Command, _broker: Broker) -> dict[str, Any]:
    # No-op placeholder — the runner picks up REFRESH commands separately
    # and re-runs the data fetch via its own pipeline.
    return {"note": "refresh queued; runner will pick it up on next cycle"}


def _h_reconnect_broker(_cmd: Command, broker: Broker) -> dict[str, Any]:
    """Reconnect to the broker. Detects the "port refused" case where
    the gateway TCP listener is fully down (vs. just wedged at the
    subscription level) and triggers a docker restart, since the
    timeout-driven self-heal in _bounded only fires on wedged calls.
    Falls back gracefully on non-IBKR brokers."""

    with contextlib.suppress(Exception):
        broker.disconnect()
    try:
        broker.connect()
        return {"reconnected": True, "via": "direct"}
    except (ConnectionRefusedError, OSError, TimeoutError) as e:
        # Only the IBKR adapter has the gateway-restart helper. If the
        # broker is a Simulator or something else, surface the original
        # error rather than guessing.
        restarter = getattr(broker, "_docker_restart_via_socket", None)
        gateway_container = getattr(broker, "_GATEWAY_CONTAINER_NAME", None)
        if restarter is None or gateway_container is None:
            raise
        # Make sure the error is actually a port-down style failure —
        # not, say, an auth refusal that a restart wouldn't fix.
        msg = str(e).lower()
        port_down = (
            isinstance(e, ConnectionRefusedError)
            or "refused" in msg
            or "no route" in msg
            or "no such" in msg
            or "errno 111" in msg
        )
        if not port_down:
            raise
        logger.bind(component="command_processor").warning(
            f"connect refused (gateway port down); restarting {gateway_container}"
        )
        restarter(gateway_container, timeout=30.0)
        return {
            "reconnected": False,
            "via": "gateway_restart",
            "note": (
                f"gateway port was down; {gateway_container} is restarting and "
                "will be ready in ~90s. Try /cycle once gateway is back."
            ),
        }


_HANDLERS = {
    CommandType.BUY: _h_buy,
    CommandType.SELL: _h_sell,
    CommandType.CLOSE: _h_close,
    CommandType.FLATTEN: _h_flatten,
    CommandType.CANCEL_ORDER: _h_cancel_order,
    CommandType.FX_CONVERT: _h_fx_convert,
    CommandType.REFRESH_DATA: _h_refresh_data,
    CommandType.RECONNECT_BROKER: _h_reconnect_broker,
}


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------


def _format_success(cmd: Command, result: dict[str, Any] | str | None) -> str:
    cmd_id = _short_id(cmd.id)
    if cmd.type == CommandType.BUY:
        return (
            f"✅ `{cmd_id}` BUY {result['symbol']} {result['qty']:g} "  # type: ignore[index]
            f"({result['type']}) submitted"  # type: ignore[index]
        )
    if cmd.type == CommandType.SELL:
        return (
            f"✅ `{cmd_id}` SELL {result['symbol']} {result['qty']:g} "  # type: ignore[index]
            f"({result['type']}) submitted"  # type: ignore[index]
        )
    if cmd.type == CommandType.CLOSE:
        return f"✅ `{cmd_id}` Close {result['symbol']} ({result['qty']:g} shares) submitted"  # type: ignore[index]
    if cmd.type == CommandType.FLATTEN:
        n = result["n_closed"]  # type: ignore[index]
        names = ", ".join(result["closed"]) or "—"  # type: ignore[index]
        skipped = result["skipped"]  # type: ignore[index]
        msg = f"✅ `{cmd_id}` Flatten: {n} positions closing\n  {names}"
        if skipped:
            msg += f"\n  ⚠️ skipped: {', '.join(skipped)}"
        return msg
    if cmd.type == CommandType.CANCEL_ORDER:
        return f"✅ `{cmd_id}` Cancel order `{result['client_order_id']}` submitted"  # type: ignore[index]
    if cmd.type == CommandType.FX_CONVERT:
        return (
            f"✅ `{cmd_id}` FX: spending {result['from_amount']:g} {result['from_ccy']} "  # type: ignore[index]
            f"→ {result['to_ccy']} (market) submitted"  # type: ignore[index]
        )
    if cmd.type == CommandType.RECONNECT_BROKER:
        via = result.get("via") if isinstance(result, dict) else None  # type: ignore[union-attr]
        if via == "gateway_restart":
            note = result.get("note", "")  # type: ignore[union-attr]
            return f"🔄 `{cmd_id}` Gateway port was down — restart issued.\n{note}"
        return f"✅ `{cmd_id}` Broker reconnected"
    return f"✅ `{cmd_id}` {cmd.type.value} executed"


def _format_failure(cmd: Command, err: str) -> str:
    return f"❌ `{_short_id(cmd.id)}` {cmd.type.value} failed: {err[:300]}"
