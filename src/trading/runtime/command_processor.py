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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from trading.core.exec_lock import ExecutionBusyError, execution_lock
from trading.core.logging import logger
from trading.core.types import (
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
    TimeInForce,
)
from trading.execution.base import Broker
from trading.risk.manager import RiskManager
from trading.risk.reduce_only import (
    QUANTITY_EPSILON,
    BrokerInstrumentIdentity,
    normalized_broker_instrument_identity,
    open_order_side,
    validated_quantity,
)
from trading.runner.alerts import TelegramAlerts
from trading.runtime.commands import (
    Command,
    CommandType,
    mark_executed,
    mark_running,
    pending_commands,
)

# Commands that submit orders to the broker. These are all serialised and
# audited, but a halt has a deliberately narrower policy: it blocks anything
# that could add or rotate exposure.  Only the two explicit *exit* commands
# may continue, and only through the fresh, reduce-only path below.
#
# This distinction matters during a loss-limit halt. Refusing a BUY/SELL/FX
# command is the point of the kill switch; refusing the trailing stop or a
# human's /close leaves the account exposed precisely when they need to get
# out.  Audit (2026-08-18) found that the old all-or-nothing gate did both.
_ORDER_SUBMITTING_COMMANDS = {
    CommandType.BUY,
    CommandType.SELL,
    CommandType.CLOSE,
    CommandType.FLATTEN,
    CommandType.FX_CONVERT,
}

# These command types are eligible for the halted reduce-only path.  This is
# intentionally type-based, not an "all" alias on SELL: an operator's
# discretionary /sell remains blocked while halted, even if its quantity
# happens to equal the current holding.
_HALTED_REDUCE_ONLY_COMMANDS = {CommandType.CLOSE, CommandType.FLATTEN}

# The quantity tolerance, contract-identity normalisation and side/quantity
# validators live in ``trading.risk.reduce_only`` so this module and the
# cycle's defensive rebalance prove "this order reduces exposure" the same
# way. Two independent definitions of that proof is one too many.
_QUANTITY_EPSILON = QUANTITY_EPSILON
_BrokerInstrumentIdentity = BrokerInstrumentIdentity
_open_order_side = open_order_side
_validated_quantity = validated_quantity
_normalized_broker_instrument_identity = normalized_broker_instrument_identity

# Order-submitting commands expire. A /flatten typed while the runner was
# down must NOT fire when the runner comes back hours or days later — the
# operator's intent was "flatten NOW", not "flatten whenever you wake up".
# (Stale May commands were found sitting in state/commands/pending/ in a
# July audit; without this gate a restart would have executed them.)
# Non-order commands (cancel, refresh, reconnect) don't expire — they're
# recovery actions, same exemption as the halt gate.
COMMAND_MAX_AGE_SECONDS = 15 * 60


def _short_id(uuid_str: str) -> str:
    return uuid_str[:8]


def _age_seconds(cmd: Command) -> float | None:
    """Age of the command, or None if requested_at is missing/unparseable."""
    try:
        ts = datetime.fromisoformat(cmd.requested_at)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        # Naive timestamps shouldn't happen (submit() writes aware UTC),
        # but if one appears, assume UTC rather than guessing local time.
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(tz=timezone.utc) - ts).total_seconds()


def _reload_and_is_halted(risk_manager: RiskManager | None) -> bool:
    """Read the cross-process halt state and return its current value.

    Telegram writes ``halt.json`` from a different container.  The initial
    read lets ordinary blocked commands fail quickly; the caller repeats it
    *inside* ``execution_lock`` before any order is built, closing the race
    where a cycle halts while a command is waiting for the lock.
    """
    if risk_manager is None:
        return False
    try:
        risk_manager._reload_halt_state()
    except Exception:
        logger.bind(component="command_processor").exception(
            "halt state reload failed; falling back to in-memory state"
        )
    return risk_manager.is_halted()


def _halt_reason(risk_manager: RiskManager) -> str:
    return getattr(risk_manager._state, "reason", "") or "halted"


def _reject_halted_command(
    cmd: Command,
    state_dir: Path,
    alerts: TelegramAlerts,
    risk_manager: RiskManager,
) -> None:
    """Record and explain a halt refusal without ever touching the broker."""
    reason = _halt_reason(risk_manager)
    msg = (
        f"refused — risk manager halted: {reason}. "
        "Only verified /close or /flatten exits are available."
    )
    logger.bind(component="command_processor").warning(
        f"halt gate blocked {cmd.type.value} command {cmd.id}"
    )
    mark_executed(cmd, state_dir, status="error", result=msg)
    alerts.error(f"🛑 `{_short_id(cmd.id)}` {cmd.type.value} {msg}")


def _fresh_positions_for_halted_exit(broker: Broker) -> list[Position]:
    """Return a position snapshot suitable for a live reduce-only exit.

    The IBKR adapter exposes ``get_positions_strict`` specifically for a
    request/response snapshot rather than its subscription cache.  Prefer it
    when available.  Other Broker implementations retain the Protocol's
    ``get_positions`` surface, but any read failure is fatal: an exit cannot
    be proven reduce-only from a stale or absent position view.
    """
    strict = getattr(broker, "get_positions_strict", None)
    try:
        raw = strict() if callable(strict) else broker.get_positions()
    except Exception as e:
        raise RuntimeError(
            "cannot verify fresh broker positions for halted reduce-only exit; no order sent"
        ) from e
    if raw is None:
        raise RuntimeError(
            "broker returned no position snapshot for halted reduce-only exit; no order sent"
        )
    return list(raw)


def _fresh_open_orders_for_halted_exit(broker: Broker) -> list[Order]:
    """Return working orders, failing closed when they cannot be checked.

    A position alone is insufficient: a queued opposite-side order can make
    a nominal close flip the account after the current order fills.  The
    normal manual path degrades when this read is unavailable; the halted
    exit path must not, because this is its sole safety proof.
    """
    try:
        raw = broker.get_open_orders()
    except Exception as e:
        raise RuntimeError(
            "cannot verify broker working orders for halted reduce-only exit; no order sent"
        ) from e
    if raw is None:
        raise RuntimeError(
            "broker returned no working-order snapshot for halted reduce-only exit; no order sent"
        )
    return list(raw)


def _halted_reduce_only_plan(
    position: Position,
    open_orders: list[Order],
) -> tuple[Side, float, float]:
    """Determine the one order that can take ``position`` exactly to zero.

    This is intentionally stricter than the normal manual sell clamp.  A
    halted exit is allowed only after the live position and *all* same-
    instrument working orders agree that the order will reduce, never flip or
    increase, exposure.  An existing order in the exposure-increasing
    direction makes that proof impossible, so the command fails closed.

    Returns ``(side, quantity_to_submit, already_working_exit_quantity)``.
    A zero quantity means existing verified exit orders already cover the
    position and nothing new should be submitted.
    """
    settled = _validated_quantity(position.quantity, label="position")
    if abs(settled) <= _QUANTITY_EPSILON:
        raise ValueError(f"no open position in {position.instrument.symbol}")

    exit_side = Side.SELL if settled > 0 else Side.BUY
    increasing_side = Side.BUY if settled > 0 else Side.SELL
    working_exit = 0.0
    position_identity = _normalized_broker_instrument_identity(position.instrument)

    for order in open_orders:
        if _normalized_broker_instrument_identity(order.instrument) != position_identity:
            continue
        quantity = _validated_quantity(order.quantity, label="working order")
        if quantity <= _QUANTITY_EPSILON:
            continue
        side = _open_order_side(order)
        if side == increasing_side:
            raise ValueError(
                f"{position.instrument.symbol}: working {side.value.upper()} order would "
                "increase exposure; cancel it before a halted close"
            )
        if side != exit_side:
            # Defensive for future enum expansion: never guess that an
            # unfamiliar order direction is harmless.
            raise ValueError(
                f"{position.instrument.symbol}: cannot verify working order direction; "
                "no order sent"
            )
        working_exit += quantity

    held = abs(settled)
    if working_exit > held + _QUANTITY_EPSILON:
        raise ValueError(
            f"{position.instrument.symbol}: working {exit_side.value.upper()} quantity "
            f"{working_exit:g} exceeds live position {held:g}; refusing an exit that could flip"
        )
    remaining = max(0.0, held - working_exit)
    if remaining <= _QUANTITY_EPSILON:
        remaining = 0.0
    return exit_side, remaining, working_exit


def _halted_close_order(
    cmd: Command,
    broker: Broker,
) -> dict[str, Any]:
    """Submit a fresh, full, reduce-only close while a halt is active."""
    sym = str(cmd.args["symbol"]).upper()
    positions = _fresh_positions_for_halted_exit(broker)
    matches = [
        p
        for p in positions
        if str(p.instrument.symbol).upper() == sym
        and abs(_validated_quantity(p.quantity, label="position")) > _QUANTITY_EPSILON
    ]
    if not matches:
        raise ValueError(f"no open position in {sym}")
    if len(matches) != 1:
        instruments = ", ".join(p.instrument.key for p in matches)
        raise ValueError(
            f"multiple live positions match {sym} ({instruments}); use /flatten or resolve "
            "the broker positions first"
        )

    position = matches[0]
    side, quantity, already_working = _halted_reduce_only_plan(
        position,
        _fresh_open_orders_for_halted_exit(broker),
    )
    if quantity == 0.0:
        return {
            "symbol": position.instrument.symbol,
            "qty": 0.0,
            "side": side.value.upper(),
            "type": OrderType.MARKET.value,
            "already_working": already_working,
            "halted_reduce_only": True,
            "submitted": False,
        }

    order = Order(
        client_order_id=f"halted-close-{_short_id(cmd.id)}",
        instrument=position.instrument,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        tif=TimeInForce.DAY,
        created_at=datetime.now(tz=timezone.utc),
    )
    broker.submit_order(order)
    return {
        "symbol": position.instrument.symbol,
        "qty": quantity,
        "side": side.value.upper(),
        "type": order.order_type.value,
        "client_order_id": order.client_order_id,
        "working_exit_qty": already_working,
        "halted_reduce_only": True,
        "submitted": True,
    }


def _halted_flatten_orders(cmd: Command, broker: Broker) -> dict[str, Any]:
    """Flatten only positions whose live broker state proves it is safe.

    The entire verification pass happens before the first submit.  In
    particular, an orphaned working order on a flat symbol could reopen the
    book after a supposedly successful flatten, so it rejects the command
    rather than reporting a misleading partial success.
    """
    positions = [
        p
        for p in _fresh_positions_for_halted_exit(broker)
        if abs(_validated_quantity(p.quantity, label="position")) > _QUANTITY_EPSILON
    ]
    open_orders = _fresh_open_orders_for_halted_exit(broker)

    seen_identities: set[_BrokerInstrumentIdentity] = set()
    for position in positions:
        identity = _normalized_broker_instrument_identity(position.instrument)
        if identity in seen_identities:
            raise ValueError(
                f"duplicate or ambiguous live position for {position.instrument.key}; "
                "no halted flatten sent"
            )
        seen_identities.add(identity)

    # Every working order must belong to a currently-held instrument.  An
    # order on a flat name would create new exposure after this command, and
    # we cannot call a flatten verified while it remains live.
    for order in open_orders:
        quantity = _validated_quantity(order.quantity, label="working order")
        if quantity <= _QUANTITY_EPSILON:
            continue
        _open_order_side(order)
        if _normalized_broker_instrument_identity(order.instrument) not in seen_identities:
            raise ValueError(
                f"working order for {order.instrument.symbol} has no live position and could "
                "create exposure; cancel it before a halted flatten"
            )

    plans = [(position, *_halted_reduce_only_plan(position, open_orders)) for position in positions]

    closed: list[str] = []
    skipped: list[str] = []
    submitted: list[Order] = []
    for idx, (position, side, quantity, already_working) in enumerate(plans):
        if quantity == 0.0:
            skipped.append(
                f"{position.instrument.symbol} (already covered by {already_working:g} "
                f"working {side.value.upper()})"
            )
            continue
        order = Order(
            client_order_id=f"halted-flatten-{_short_id(cmd.id)}-{idx}",
            instrument=position.instrument,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            created_at=datetime.now(tz=timezone.utc),
        )
        broker.submit_order(order)
        submitted.append(order)
        closed.append(position.instrument.symbol)

    return {
        "closed": closed,
        "skipped": skipped,
        "n_closed": len(closed),
        "client_order_ids": [order.client_order_id for order in submitted],
        "halted_reduce_only": True,
    }


def _execute_halted_reduce_only(cmd: Command, broker: Broker) -> dict[str, Any]:
    """Route the two explicitly permitted halted commands to safe handlers."""
    if cmd.type == CommandType.CLOSE:
        return _halted_close_order(cmd, broker)
    if cmd.type == CommandType.FLATTEN:
        return _halted_flatten_orders(cmd, broker)
    raise AssertionError(f"{cmd.type.value} is not a halted reduce-only command")


class _RecordingBroker:
    """Broker proxy that writes every submitted order to the ledger.

    Command handlers called ``broker.submit_order`` directly and NOTHING
    wrote to orders.db, so an operator ``/buy``, ``/sell``, ``/close`` or
    ``/flatten`` was invisible to ``/orders``, to ``/pending``, to fill
    reconciliation, to realized P&L, and to ``agents.exits.recent_exits``.
    Observed 2026-08-11: `/sell UNH` submitted at 14:14:43 and `/orders`
    a minute later listed only the ten cycle buys.

    The cycle already saved its orders (``cycle.py``), so the gap was
    exactly the operator's own trades — the ones least likely to be
    reconstructable from memory later.

    Wrapping the broker rather than patching each handler is deliberate:
    six handlers submit orders today and the seventh would have been
    missed. ``__getattr__`` delegates everything else untouched.

    Saved BEFORE submitting, matching ``Cycle.force_flatten_orders``: an
    order that was sent and then lost to a crash is worse than a PENDING
    row with no fill, which reconciliation simply never closes.
    """

    def __init__(
        self,
        inner: Any,
        store: Any,
        *,
        risk_manager: RiskManager | None = None,
        allow_when_halted: bool = False,
    ) -> None:
        self._inner = inner
        self._store = store
        self._risk_manager = risk_manager
        self._allow_when_halted = allow_when_halted

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def submit_order(self, order: Any) -> Any:
        if self._risk_manager is None:
            return self._record_and_submit(order)
        with self._risk_manager.submission_gate(
            allow_when_halted=self._allow_when_halted
        ) as permitted:
            if not permitted:
                raise RuntimeError("halt engaged before broker submission; no order sent")
            return self._record_and_submit(order)

    def convert_currency(self, *args: Any, **kwargs: Any) -> Any:
        """Gate an FX conversion like any other exposure-increasing action."""
        converter = getattr(self._inner, "convert_currency", None)
        if not callable(converter):
            raise RuntimeError("this broker doesn't support FX conversion")
        if self._risk_manager is None:
            return converter(*args, **kwargs)
        with self._risk_manager.submission_gate(
            allow_when_halted=self._allow_when_halted
        ) as permitted:
            if not permitted:
                raise RuntimeError("halt engaged before FX conversion; no conversion sent")
            return converter(*args, **kwargs)

    def _record_and_submit(self, order: Any) -> Any:
        if self._store is not None:
            try:
                self._store.save_order(order)
            except Exception as e:  # never block a trade on bookkeeping
                logger.bind(component="command_processor").exception(
                    f"could not record {getattr(order, 'client_order_id', '?')} to the ledger: {e}"
                )
        return self._inner.submit_order(order)


def _recording(
    broker: Broker,
    state_dir: Path,
    *,
    risk_manager: RiskManager | None = None,
    allow_when_halted: bool = False,
) -> Any:
    try:
        from trading.execution.store import OrderStore

        return _RecordingBroker(
            broker,
            OrderStore(Path(state_dir) / "orders.db"),
            risk_manager=risk_manager,
            allow_when_halted=allow_when_halted,
        )
    except Exception as e:
        logger.bind(component="command_processor").exception(f"ledger unavailable: {e}")
        # A bookkeeping outage is not permission to bypass the hard halt
        # gate. Keep the safety proxy even when the ledger itself cannot be
        # constructed; it simply records no row for this command.
        if risk_manager is not None:
            return _RecordingBroker(
                broker,
                None,
                risk_manager=risk_manager,
                allow_when_halted=allow_when_halted,
            )
        return broker


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

    # TTL gate: order-submitting commands must be fresh. Fail closed — a
    # missing/unparseable timestamp on an order command is refused too,
    # because we cannot prove it isn't stale.
    if cmd.type in _ORDER_SUBMITTING_COMMANDS:
        age = _age_seconds(cmd)
        if age is None or age > COMMAND_MAX_AGE_SECONDS:
            shown = "unknown age" if age is None else f"{age / 60:.0f} min old"
            msg = (
                f"expired — requested {shown}, limit is "
                f"{COMMAND_MAX_AGE_SECONDS // 60} min. Re-issue if still wanted."
            )
            logger.bind(component="command_processor").warning(
                f"TTL gate blocked {cmd.type.value} command {cmd.id} ({shown})"
            )
            mark_executed(cmd, state_dir, status="error", result=msg)
            alerts.error(f"⌛ `{_short_id(cmd.id)}` {cmd.type.value} {msg}")
            return

    # Fast halt gate.  CLOSE and FLATTEN are deliberately exempt here, but
    # they do not get a normal order path: after acquiring the execution lock
    # we re-read the halt file and verify a fresh broker position + all open
    # orders before allowing a strictly reduce-only exit.  Every other order
    # type remains blocked while halted.
    if (
        risk_manager is not None
        and _reload_and_is_halted(risk_manager)
        and cmd.type in _ORDER_SUBMITTING_COMMANDS
        and cmd.type not in _HALTED_REDUCE_ONLY_COMMANDS
    ):
        _reject_halted_command(cmd, state_dir, alerts, risk_manager)
        return

    # Execution gate: one mutex across every path that can reach the
    # broker — the scheduled cycle, the on-demand cycle, the approval
    # flow and these manual commands. They previously shared nothing but
    # per-job locks, which stop a job racing ITSELF and do nothing about
    # two different jobs racing each other; the 2026-07-14 order-stacking
    # incident came out of exactly that gap.
    #
    # Only order-submitting commands take it. /reconnect and /refresh do
    # not touch positions and must stay available while a cycle runs —
    # in particular /reconnect is how you recover a wedged broker.
    #
    # A refusal here is reported to the operator with the current holder,
    # because "try again" is only useful if you know what to wait for.
    needs_lock = cmd.type in _ORDER_SUBMITTING_COMMANDS
    try:
        if needs_lock:
            with execution_lock(state_dir, holder=f"command:{cmd.type.value}"):
                # Check once more *inside* the lock.  A cycle can halt the
                # book while this command is queued behind another submit;
                # a pre-lock check alone would let a stale BUY race through.
                halted_now = _reload_and_is_halted(risk_manager)
                if (
                    halted_now
                    and risk_manager is not None
                    and cmd.type not in _HALTED_REDUCE_ONLY_COMMANDS
                ):
                    _reject_halted_command(cmd, state_dir, alerts, risk_manager)
                    return

                # Order-submitting commands go through the recording proxy
                # so the operator's own trades reach the ledger like the
                # cycle's do.  The halted exit path uses the same proxy,
                # execution lock and command audit as ordinary commands.
                used = _recording(
                    broker,
                    state_dir,
                    risk_manager=risk_manager,
                    allow_when_halted=halted_now,
                )
                if halted_now:
                    result = _execute_halted_reduce_only(cmd, used)
                else:
                    result = handler(cmd, used)
        else:
            result = handler(cmd, broker)
        mark_executed(cmd, state_dir, status="ok", result=result)
        alerts.info(_format_success(cmd, result))
    except ExecutionBusyError as e:
        msg = f"busy — {e.holder} is trading right now. Re-issue in a moment."
        logger.bind(component="command_processor").warning(
            f"execution lock blocked {cmd.type.value} command {cmd.id}: {e.holder}"
        )
        mark_executed(cmd, state_dir, status="error", result=msg)
        alerts.error(f"⏳ `{_short_id(cmd.id)}` {cmd.type.value} {msg}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.bind(component="command_processor").exception(f"command {cmd.type.value} failed")
        mark_executed(cmd, state_dir, status="error", result=err)
        alerts.error(_format_failure(cmd, err))


# ---------------------------------------------------------------------------
# Handlers — one per CommandType
# ---------------------------------------------------------------------------


def _h_gateway_stop(_cmd: Command, _broker: Broker) -> dict[str, Any]:
    """Stop the gateway container, freeing the operator's IBKR session.

    Halts first. IBC holds the account's only session, so while this
    container is up the operator cannot log into TWS or mobile — and the
    moment it is down the desk has no broker. Trading is stopped BEFORE
    the session is released so the runner does not spend the manual
    session failing order submissions against a dead socket.
    """
    from trading.core.config import settings
    from trading.risk.halt_file import set_halted
    from trading.runtime.docker_ctl import GATEWAY_CONTAINER, docker_action

    set_halted(settings.state_dir, halted=True, reason="gateway stopped for manual trading")
    result = docker_action(GATEWAY_CONTAINER, "stop")
    return {
        "container": GATEWAY_CONTAINER,
        "action": "stop",
        "result": result,
        "note": "trading HALTED; your IBKR session is free — /gateway start when done",
    }


def _h_gateway_start(_cmd: Command, _broker: Broker) -> dict[str, Any]:
    """Start the gateway again. Does NOT resume trading.

    Login takes ~90s including 2FA, and the halt stays set on purpose:
    the operator has just traded by hand, so the desk's view of the book
    is stale until the next snapshot. Resuming is a separate, deliberate
    /resume.
    """
    from trading.runtime.docker_ctl import GATEWAY_CONTAINER, docker_action

    result = docker_action(GATEWAY_CONTAINER, "start")
    return {
        "container": GATEWAY_CONTAINER,
        "action": "start",
        "result": result,
        "note": "logging in (~90s, 2FA). Still HALTED — send /resume when you are ready",
    }


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

    # Long-only invariant. The cycle nets working orders before sizing
    # (2026-07-15); this path never did, so a sell already queued at the
    # broker was invisible here. Guards fire on a 15-minute cadence
    # through 20:59 UTC, past the close, so their exits sit unfilled
    # overnight — and an operator who sees "🛑 Trailing stop VST" and
    # reaches for /close sells the position twice.
    from trading.risk.presubmit import clamp_sell

    qty, clamp_note = clamp_sell(broker, sym, qty)

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
    out = {
        "symbol": sym,
        "qty": qty,
        "side": "SELL",
        "type": order.order_type.value,
        "limit": limit,
        "client_order_id": order.client_order_id,
    }
    if clamp_note:
        # Surfaced to the operator: a sell that silently shrinks is
        # indistinguishable from a partial fill.
        out["clamped"] = clamp_note
    return out


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
    """Panic button — closes everything, ignoring /hold by design.

    Still netted against working orders. Flatten stays unconditional in
    every other respect, but a panic button that can leave you SHORT the
    names you were trying to exit is not an exit: if a sell for the full
    position is already queued, a second one crosses zero. Netting only
    ever makes flatten sell LESS, never less complete.
    """
    from trading.risk.presubmit import sellable_quantity

    closed: list[str] = []
    skipped: list[str] = []
    for pos in broker.get_positions():
        if pos.quantity == 0:
            continue
        side = Side.SELL if pos.quantity > 0 else Side.BUY
        qty = abs(pos.quantity)
        if pos.quantity > 0:
            r = sellable_quantity(broker, pos.instrument.symbol)
            if r.blocked:
                skipped.append(f"{pos.instrument.symbol} (already fully covered by a working sell)")
                continue
            qty = min(qty, r.sellable)
        order = Order(
            client_order_id=f"flatten-{pos.instrument.symbol}-{_short_id('x' * 8)}",
            instrument=pos.instrument,
            side=side,
            quantity=qty,
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
    converter = getattr(broker, "convert_currency", None)
    if not callable(converter):
        raise RuntimeError("this broker doesn't support FX conversion")
    typed_converter = cast(Callable[..., dict[str, Any]], converter)
    return typed_converter(from_ccy=from_ccy, to_ccy=to_ccy, from_amount=amount)


#: Dropped by /refresh, picked up by the runner. Same pattern as
#: ``trigger_now.flag``, for the same reason: the command processor has
#: no handle on the runner's scheduler.
REFRESH_FLAG = "refresh_requested.flag"


def _h_refresh_data(_cmd: Command, _broker: Broker) -> dict[str, Any]:
    """Ask the runner to top up the price cache now.

    This used to be a no-op that returned "refresh queued; runner will
    pick it up on next cycle". Nothing picked it up. The parquet cache is
    refreshed by an independent scheduled job at 21:40 UTC, which runs
    whether or not anyone asked — so ``/refresh`` reported success and
    changed nothing, and an operator refreshing before a manual ``/cycle``
    got stale data and a green tick.

    Now it drops a flag the runner watches. Deliberately does NOT do the
    fetch here: this runs inside the 5-second command-processor job with
    ``max_instances=1``, and a full universe pass takes ~2 minutes — which
    would block ``/halt`` and ``/flatten`` behind a data refresh.
    """
    from trading.core.config import settings as _s

    path = Path(_s.state_dir) / REFRESH_FLAG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now(tz=timezone.utc).isoformat())
    return {"note": "price-cache refresh requested; the runner starts it within ~5s"}


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
    CommandType.GATEWAY_STOP: _h_gateway_stop,
    CommandType.GATEWAY_START: _h_gateway_start,
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
        if isinstance(result, dict) and result.get("submitted") is False:
            return (
                f"✅ `{cmd_id}` Close {result['symbol']}: already covered by "
                f"{result['already_working']:g} working {result['side']} order(s); no duplicate sent"
            )
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
        reconnect_result = result if isinstance(result, dict) else {}
        via = reconnect_result.get("via")
        if via == "gateway_restart":
            note = reconnect_result.get("note", "")
            return f"🔄 `{cmd_id}` Gateway port was down — restart issued.\n{note}"
        return f"✅ `{cmd_id}` Broker reconnected"
    return f"✅ `{cmd_id}` {cmd.type.value} executed"


def _format_failure(cmd: Command, err: str) -> str:
    return f"❌ `{_short_id(cmd.id)}` {cmd.type.value} failed: {err[:300]}"
