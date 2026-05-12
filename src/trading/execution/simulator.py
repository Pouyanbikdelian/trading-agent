"""In-memory broker — fills orders against historical bars.

Design
------
The simulator is *clock-driven*: the caller calls ``step(ts, bars)`` once per
bar, supplying a ``{symbol: Bar}`` snapshot for the new bar. Orders that
were submitted *during* (or before) bar ``t-1`` are eligible to fill at
bar ``t``'s open price plus a slippage offset. This matches the
backtester's no-lookahead convention: weights at close of bar ``t-1``
become positions held during bar ``t``.

Order types in v1
-----------------
Only ``MARKET`` orders are supported. Limit / stop / MOC orders are
recognized but not filled — they remain ``PENDING`` and emit a warning.
Adding them is mechanical (compare to bar high/low) but each adds
edge cases (gap-throughs, partial fills) we'd rather defer until we
have a real strategy that uses them.

Cash + positions
----------------
* Buys subtract from cash, add to position.
* Sells the inverse. Shorts are allowed (no margin model — that belongs
  in the risk manager).
* Cash includes commission and slippage costs as they happen.
* Equity = cash + sum(position_qty * last_close) across positions.

Why not just use the ``backtest.engine``? Because the simulator is the
runner's broker — it accepts orders, owns the position state, and gets
queried by the risk manager. The backtester goes from weights to PnL in
one shot and has no concept of order-by-order lifecycle.
"""

from __future__ import annotations

from datetime import datetime

from trading.core.logging import logger
from trading.core.types import (
    AccountSnapshot,
    Bar,
    Fill,
    Instrument,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from trading.execution.base import Broker, BrokerError


class Simulator(Broker):
    """Clock-driven in-memory broker.

    Construct, then call ``connect()``, then drive with ``step()`` calls.
    Orders sit in a pending queue until the next ``step``.
    """

    name = "simulator"

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        *,
        slippage_bps: float = 2.0,
        commission_bps: float = 1.0,
        min_commission: float = 0.0,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.slippage_bps = float(slippage_bps)
        self.commission_bps = float(commission_bps)
        self.min_commission = float(min_commission)
        self._connected = False
        self._pending: list[Order] = []
        self._orders: dict[str, tuple[Order, OrderStatus]] = {}
        self._fills: list[tuple[Fill, str]] = []  # (fill, client_order_id)
        self._positions: dict[str, Position] = {}
        self._cash: float = self.initial_cash
        self._last_close: dict[str, float] = {}
        self._ts: datetime | None = None

    # ---------------------------------------------------------- lifecycle

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise BrokerError("simulator is not connected — call connect() first")

    # -------------------------------------------------------------- orders

    def submit_order(self, order: Order) -> Order:
        self._ensure_connected()
        if order.client_order_id in self._orders:
            raise BrokerError(f"duplicate client_order_id={order.client_order_id!r}")
        if order.order_type != OrderType.MARKET:
            logger.bind(broker=self.name).warning(
                f"simulator v1 only fills MARKET orders; "
                f"{order.client_order_id} ({order.order_type.value}) will stay PENDING"
            )
        self._orders[order.client_order_id] = (order, OrderStatus.SUBMITTED)
        self._pending.append(order)
        return order

    def cancel_order(self, client_order_id: str) -> None:
        self._ensure_connected()
        rec = self._orders.get(client_order_id)
        if rec is None:
            raise BrokerError(f"unknown client_order_id={client_order_id!r}")
        order, status = rec
        if status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return
        self._orders[client_order_id] = (order, OrderStatus.CANCELLED)
        self._pending = [o for o in self._pending if o.client_order_id != client_order_id]

    # ---------------------------------------------------------- step / clock

    def step(self, ts: datetime, bars: dict[str, Bar]) -> list[Fill]:
        """Advance the clock to ``ts`` with new bars. Returns fills produced
        on this bar (so the caller can react to them immediately)."""
        self._ensure_connected()
        if ts.tzinfo is None:
            raise ValueError("step ts must be timezone-aware")
        self._ts = ts
        new_fills: list[Fill] = []

        # Update last-close marks so equity / unrealized PnL are current.
        for sym, bar in bars.items():
            self._last_close[sym] = bar.close

        # Walk the pending queue once. We mutate ``self._pending`` while
        # iterating, so use a snapshot copy.
        carry_over: list[Order] = []
        for order in self._pending:
            bar = bars.get(order.instrument.symbol)
            if bar is None or order.order_type != OrderType.MARKET:
                carry_over.append(order)
                continue
            fill = self._fill_market_at_open(order, bar)
            new_fills.append(fill)
        self._pending = carry_over

        # Recompute positions' unrealized PnL against the latest close.
        self._mark_to_market()
        self._fills.extend((f, f.order_id) for f in new_fills)
        return new_fills

    # ----------------------------------------------------------- fill engine

    def _fill_market_at_open(self, order: Order, bar: Bar) -> Fill:
        """Fill a market order at the bar's open price, with bps slippage
        applied against the trade direction (worse fill for the taker)."""
        slip = self.slippage_bps / 1e4
        side_mult = 1.0 if order.side == Side.BUY else -1.0
        fill_price = bar.open * (1.0 + side_mult * slip)
        notional = fill_price * order.quantity
        commission = max(self.min_commission, abs(notional) * self.commission_bps / 1e4)

        # Cash flow: buy reduces cash; sell increases it. Commission always reduces.
        self._cash -= side_mult * notional + commission
        self._update_position(order.instrument, side_mult * order.quantity, fill_price)

        self._orders[order.client_order_id] = (order, OrderStatus.FILLED)
        return Fill(
            order_id=order.client_order_id,
            ts=bar.ts,
            quantity=order.quantity * side_mult,
            price=fill_price,
            commission=commission,
            venue=self.name,
        )

    def _update_position(self, instrument: Instrument, signed_qty: float, price: float) -> None:
        """Apply a fill to the running position, computing realized PnL on
        the closing portion of opposite-signed trades."""
        key = instrument.key
        existing = self._positions.get(key)
        if existing is None:
            self._positions[key] = Position(
                instrument=instrument,
                quantity=signed_qty,
                avg_price=price,
            )
            return

        new_qty = existing.quantity + signed_qty
        # Same-direction add: VWAP the entry. Opposite-direction: realize PnL
        # on the portion being closed; the remainder either flattens or flips.
        if existing.quantity * signed_qty >= 0:
            total_notional = existing.quantity * existing.avg_price + signed_qty * price
            avg = total_notional / new_qty if new_qty != 0 else 0.0
            realized = existing.realized_pnl
        else:
            closing = min(abs(existing.quantity), abs(signed_qty)) * (
                1.0 if existing.quantity > 0 else -1.0
            )
            # Realized PnL = closing_qty * (exit_price - avg_price) — signed properly.
            realized = existing.realized_pnl + closing * (price - existing.avg_price)
            # If we flipped past flat, the residual side's avg_price is this fill's.
            avg = price if abs(signed_qty) > abs(existing.quantity) else existing.avg_price

        self._positions[key] = Position(
            instrument=instrument,
            quantity=new_qty,
            avg_price=avg,
            realized_pnl=realized,
            unrealized_pnl=0.0,  # filled in by _mark_to_market
        )

    def _mark_to_market(self) -> None:
        for key, pos in list(self._positions.items()):
            last = self._last_close.get(pos.instrument.symbol)
            unreal = (last - pos.avg_price) * pos.quantity if last is not None else 0.0
            self._positions[key] = pos.model_copy(update={"unrealized_pnl": unreal})

    # ------------------------------------------------------------- state

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> AccountSnapshot:
        if self._ts is None:
            raise BrokerError("step() at least once before querying the account")
        market_value = sum(
            self._last_close.get(p.instrument.symbol, p.avg_price) * p.quantity
            for p in self._positions.values()
        )
        return AccountSnapshot(
            ts=self._ts,
            cash=self._cash,
            equity=self._cash + market_value,
            positions={p.instrument.key: p for p in self._positions.values()},
        )

    def get_fills(self, *, since: datetime | None = None) -> list[Fill]:
        fills = [f for f, _ in self._fills]
        if since is not None:
            fills = [f for f in fills if f.ts >= since]
        return fills

    def get_order_status(self, client_order_id: str) -> OrderStatus:
        rec = self._orders.get(client_order_id)
        if rec is None:
            raise BrokerError(f"unknown client_order_id={client_order_id!r}")
        return rec[1]
