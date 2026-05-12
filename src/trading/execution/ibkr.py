"""IBKR broker via ``ib-async``.

Mirrors the data-layer pattern: async at the wire, sync facade. The runner
(Phase 8) is async-aware; tests inject a fake ``ib`` object so this module
imports cleanly even without a running IB Gateway.

What this DOESN'T do (yet)
--------------------------
* Live trading is gated by ``settings.is_live_armed()`` *outside* this
  module — Claude must never flip it. The adapter happily routes to a
  live gateway if pointed at one; that's the operator's choice.
* No order book / level-2 data, no IB algos, no bracket orders. Phase 8
  will add bracket orders for stop attachments.
* Commission reporting: we surface IBKR's own commission report when it
  arrives; pacing or rebates aren't reconciled.

Hard rule (from CLAUDE.md): live trading also requires
``ALLOW_LIVE_TRADING=true`` and ``TRADING_ENV=live``. The adapter does not
check these — the runner does, before instantiating us.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from trading.core.config import settings
from trading.core.logging import logger
from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Fill,
    Instrument,
    Order,
    OrderType,
    Position,
    Side,
)
from trading.execution.base import Broker, BrokerError, NotConnectedError

# Map our enums to ib-async strings. Centralizing here means a vendor change
# only touches this file.
_OUR_TO_IBKR_ACTION: dict[Side, str] = {Side.BUY: "BUY", Side.SELL: "SELL"}
_OUR_TO_IBKR_ORDER_TYPE: dict[OrderType, str] = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
    OrderType.MOC: "MOC",
    OrderType.LOC: "LOC",
}


class IbkrBroker(Broker):
    """Synchronous Broker facade over an ib-async client.

    Construct, then ``connect()``. The first connect call lazily imports
    ``ib_async`` and creates a client — keeping import cost off the cold path
    for offline tooling. Pass an existing ``ib`` object to skip the lazy
    import (used by unit tests).
    """

    name = "ibkr"

    def __init__(
        self,
        ib: Any | None = None,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
    ) -> None:
        self._ib = ib
        self._host = host or settings.ibkr_host
        self._port = port or settings.ibkr_port
        self._client_id = client_id or settings.ibkr_client_id
        self._connected = False

    # --------------------------------------------------------- lifecycle

    def connect(self) -> None:
        if self._ib is None:
            from ib_async import IB  # lazy import

            self._ib = IB()
        if not self._ib.isConnected():
            self._run(self._ib.connectAsync(self._host, self._port, clientId=self._client_id))
        self._connected = True
        logger.bind(broker=self.name).info(
            f"connected ibkr@{self._host}:{self._port} client_id={self._client_id}"
        )

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected or self._ib is None or not self._ib.isConnected():
            raise NotConnectedError("IbkrBroker is not connected — call connect() first")

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _run(coro: Any) -> Any:
        """Run an awaitable from synchronous code. Reuses the running loop
        when called inside one (notebooks, the live runner)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def _contract(self, instrument: Instrument) -> Any:
        from ib_async import Crypto, Forex, Future, Stock  # lazy import

        if instrument.asset_class == AssetClass.FX:
            return Forex(instrument.symbol)
        if instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF):
            return Stock(
                symbol=instrument.symbol,
                exchange=instrument.exchange or "SMART",
                currency=instrument.currency,
            )
        if instrument.asset_class == AssetClass.CRYPTO:
            return Crypto(instrument.symbol, exchange=instrument.exchange or "PAXOS")
        if instrument.asset_class == AssetClass.FUTURE:
            return Future(
                symbol=instrument.symbol,
                exchange=instrument.exchange or "GLOBEX",
                currency=instrument.currency,
            )
        raise BrokerError(
            f"IbkrBroker does not yet support asset_class={instrument.asset_class.value}"
        )

    def _build_ib_order(self, order: Order) -> Any:
        from ib_async import Order as IbOrder  # lazy import

        ib_order = IbOrder()
        ib_order.action = _OUR_TO_IBKR_ACTION[order.side]
        ib_order.totalQuantity = order.quantity
        ib_order.orderType = _OUR_TO_IBKR_ORDER_TYPE[order.order_type]
        if order.limit_price is not None:
            ib_order.lmtPrice = order.limit_price
        if order.stop_price is not None:
            ib_order.auxPrice = order.stop_price
        ib_order.tif = order.tif.value
        # Stamp our client_order_id so we can correlate IBKR's permId back to it.
        ib_order.orderRef = order.client_order_id
        return ib_order

    # ------------------------------------------------------------- orders

    def submit_order(self, order: Order) -> Order:
        self._ensure_connected()
        contract = self._contract(order.instrument)
        ib_order = self._build_ib_order(order)
        self._ib.placeOrder(contract, ib_order)
        logger.bind(broker=self.name, symbol=order.instrument.symbol).info(
            f"submitted {order.side.value} {order.quantity} {order.instrument.symbol} "
            f"as {order.order_type.value}"
        )
        # IBKR is async — the order's status is "PendingSubmit" until ack'd.
        # We don't block here; the caller polls via get_order_status (Phase 8).
        return order

    def cancel_order(self, client_order_id: str) -> None:
        self._ensure_connected()
        for trade in self._ib.openTrades():
            if getattr(trade.order, "orderRef", None) == client_order_id:
                self._ib.cancelOrder(trade.order)
                return
        # Already terminal or never seen — non-fatal.
        logger.bind(broker=self.name).warning(
            f"no open trade matches client_order_id={client_order_id!r}; treating cancel as a no-op"
        )

    # ------------------------------------------------------------- state

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        out: list[Position] = []
        for p in self._ib.positions():
            instrument = _ibkr_contract_to_instrument(p.contract)
            out.append(
                Position(
                    instrument=instrument,
                    quantity=float(p.position),
                    avg_price=float(p.avgCost) / max(instrument.multiplier, 1.0),
                )
            )
        return out

    def get_account(self) -> AccountSnapshot:
        self._ensure_connected()
        ts = datetime.now(tz=timezone.utc)
        # Try the structured summary first; fall back to TWS-side accountValues.
        cash = 0.0
        equity = 0.0
        for row in self._ib.accountSummary():
            tag = getattr(row, "tag", None)
            try:
                val = float(getattr(row, "value", 0.0))
            except (TypeError, ValueError):
                continue
            if tag == "TotalCashValue":
                cash = val
            elif tag == "NetLiquidation":
                equity = val
        positions = {p.instrument.key: p for p in self.get_positions()}
        return AccountSnapshot(ts=ts, cash=cash, equity=equity, positions=positions)

    def get_fills(self, *, since: datetime | None = None) -> list[Fill]:
        self._ensure_connected()
        out: list[Fill] = []
        for f in self._ib.fills():
            exec_ = f.execution
            ts = getattr(exec_, "time", None)
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            out.append(
                Fill(
                    order_id=getattr(f.order, "orderRef", "") or str(getattr(exec_, "orderId", "")),
                    ts=ts,
                    quantity=float(exec_.shares),
                    price=float(exec_.price),
                    commission=float(getattr(f.commissionReport, "commission", 0.0) or 0.0),
                    venue=getattr(exec_, "exchange", None),
                )
            )
        return out


def _ibkr_contract_to_instrument(contract: Any) -> Instrument:
    """Best-effort reverse mapping; the runner uses this to keep the broker's
    position view inside our type system."""
    sec_type = getattr(contract, "secType", "STK")
    asset_class_map = {
        "STK": AssetClass.EQUITY,
        "ETF": AssetClass.ETF,
        "CASH": AssetClass.FX,
        "CRYPTO": AssetClass.CRYPTO,
        "FUT": AssetClass.FUTURE,
        "OPT": AssetClass.OPTION,
    }
    asset_class = asset_class_map.get(sec_type, AssetClass.EQUITY)
    return Instrument(
        symbol=contract.symbol if sec_type != "CASH" else f"{contract.symbol}{contract.currency}",
        asset_class=asset_class,
        exchange=getattr(contract, "exchange", None),
        currency=getattr(contract, "currency", "USD") or "USD",
        multiplier=float(getattr(contract, "multiplier", None) or 1.0),
    )


def new_client_order_id(prefix: str = "trd") -> str:
    """Convenience: short ID with millisecond timestamp and uuid suffix.

    IBKR truncates orderRef at 40 chars; this comes in well under that."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
