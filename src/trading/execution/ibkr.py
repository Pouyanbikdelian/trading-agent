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


class BrokerTimeoutError(BrokerError):
    """Raised when an IBKR API call exceeds its timeout budget.

    The cycle catches this and logs a clear "broker hung" error + Telegram
    alert, instead of waiting forever for the gateway to respond.
    """


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

    # IBKR's well-known port convention:
    #   4001 = live IB Gateway,   4002 = paper IB Gateway
    #   7496 = live TWS,          7497 = paper TWS
    # If the operator is connected to a LIVE port we re-check is_live_armed
    # on every submit_order. Paper ports skip the check (paper money is
    # the playground; the gate exists to protect real capital).
    _LIVE_PORTS: frozenset[int] = frozenset({4001, 7496})

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

    def _is_live_port(self) -> bool:
        return self._port in self._LIVE_PORTS

    # --------------------------------------------------------- lifecycle

    def connect(self) -> None:
        if self._ib is None:
            from ib_async import IB, util  # lazy import

            # ib-async is single-event-loop. If we let ``asyncio.run`` in
            # ``_run`` create a transient loop just for ``connectAsync``,
            # that loop closes after the call and every subsequent ib-async
            # call (placeOrder, accountSummary, …) raises ``Event loop is
            # closed``. ``util.startLoop()`` spins up a *persistent*
            # background loop in a daemon thread so the IB instance and
            # all its async machinery stay alive for the process lifetime.
            # If we're already inside an async context (e.g. inside the
            # runner's APScheduler thread), skip startLoop — ib-async will
            # use the running loop instead.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                util.startLoop()
            self._ib = IB()
        if not self._ib.isConnected():
            # Route connectAsync through ib-async's util.run, which uses
            # the persistent background loop set up above. We can't use
            # asyncio.run() — that creates+closes a temporary loop and
            # leaves the IB instance with stale loop references.
            from ib_async import util as _util

            # Wrap connectAsync in asyncio.wait_for so we get a hard
            # ceiling (60s) on the initial handshake — useful when the
            # gateway is still in its login dance.
            _util.run(
                asyncio.wait_for(
                    self._ib.connectAsync(self._host, self._port, clientId=self._client_id),
                    timeout=60.0,
                )
            )
        self._connected = True
        logger.bind(broker=self.name).info(
            f"connected ibkr@{self._host}:{self._port} client_id={self._client_id}"
        )

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._ib is None or not self._connected or not self._ib.isConnected():
            # Self-heal: if a prior failed auto-reconnect (e.g. _bounded
            # timeout path) left us disconnected, try once more before
            # giving up. The next cycle's get_account / get_positions
            # would otherwise fail forever until manual /reconnect.
            try:
                self.connect()
            except Exception as e:
                raise NotConnectedError(
                    f"IbkrBroker is not connected and auto-reconnect failed: {e!r}"
                ) from e

    # ------------------------------------------------------------ helpers

    # Hard per-call timeout for IBKR Gateway requests. Gateway can be
    # connected at the TCP layer while its broker session is dead, in which
    # case API calls hang forever — exactly what we lived through earlier.
    # 30 seconds is generous for healthy gateways, draconian for broken ones.
    DEFAULT_API_TIMEOUT_S: float = 30.0

    @staticmethod
    def _run(coro: Any) -> Any:
        """Run an awaitable from synchronous code. Reuses the running loop
        when called inside one (notebooks, the live runner)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def _bounded(self, what: str, fn: Any, *, timeout: float | None = None) -> Any:
        """Run a synchronous ib-async call under a thread-based timeout.

        ib-async's sync API blocks on its internal event loop. If that loop
        is wedged (broker session dead, gateway not responsive), the call
        hangs indefinitely. We dispatch into a worker thread and reap with
        a hard timeout — on expiry, attempt ONE auto-reconnect + retry. If
        that also fails, raise ``BrokerTimeoutError`` so the cycle aborts
        cleanly instead of silently waiting hours.

        Self-heal rationale: IBKR Gateway routinely drops its server-side
        session (Error 1100 in the IBKR API). The gateway recovers in
        seconds, but our existing client connection thinks it's still
        live and every API call hangs until the local 30s timeout. A
        single reconnect-and-retry transparently rides through these
        transient session drops without operator intervention.

        IMPORTANT: do not use ThreadPoolExecutor as a context manager here.
        The default ``__exit__`` calls ``shutdown(wait=True)`` which blocks
        waiting for the still-running wedged task to complete — that defeats
        the entire timeout. We explicitly ``shutdown(wait=False)`` so the
        leaked thread can die in the background (the GC will eventually
        clean it up; the cost is one dead Python thread per IBKR wedge,
        which is acceptable until the operator restarts the gateway).
        """
        try:
            return self._call_with_timeout(what, fn, timeout)
        except BrokerTimeoutError as first_err:
            # Try one reconnect + retry before surrendering. Most "wedged
            # gateway" symptoms come from a dead session that recovers on a
            # fresh connection. We protect against reconnect storms by
            # only attempting once per call.
            logger.bind(broker=self.name).warning(
                f"{what} timed out; attempting one auto-reconnect + retry"
            )
            try:
                self._reconnect_session()
            except Exception as reconnect_err:
                logger.bind(broker=self.name).exception(
                    f"auto-reconnect failed during {what}: {reconnect_err!r}"
                )
                raise first_err from None
            try:
                return self._call_with_timeout(what, fn, timeout)
            except BrokerTimeoutError:
                # Second timeout after fresh connect — gateway is genuinely
                # wedged. Re-raise so the cycle aborts and the operator is
                # alerted; auto-halt counter will tick.
                logger.bind(broker=self.name).warning(
                    f"{what} still timing out after reconnect — surrender"
                )
                raise

    def _call_with_timeout(self, what: str, fn: Any, timeout: float | None) -> Any:
        """Inner: one ThreadPoolExecutor-bounded call, no retry."""
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FutTimeout

        timeout = timeout if timeout is not None else self.DEFAULT_API_TIMEOUT_S
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ibkr-{what}")
        try:
            future = ex.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FutTimeout as e:
                raise BrokerTimeoutError(
                    f"IBKR {what} timed out after {timeout:.0f}s — gateway likely "
                    "has a dead broker session (try restarting ib-gateway container)"
                ) from e
        finally:
            ex.shutdown(wait=False)

    def _reconnect_session(self) -> None:
        """Best-effort reconnect after a session drop.

        Disconnect (suppress errors — we'll connect fresh either way),
        then call connect() which goes through the normal handshake path
        with its own 60s timeout.
        """
        import contextlib

        with contextlib.suppress(Exception):
            self.disconnect()
        self.connect()

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
        # Defense-in-depth: even though the CLI checks ``is_live_armed`` once
        # at runner start, a long-running process can have its environment
        # drift (operator edits .env, container is rebuilt with new vars).
        # We re-check on EVERY order so the live gate can't be bypassed by
        # an in-process flag flip. The CLI-side check still stands as the
        # first line of defense; this is the last.
        if self._is_live_port() and not settings.is_live_armed():
            raise BrokerError(
                "live trading not armed — refusing to submit order. Set "
                "TRADING_ENV=live and ALLOW_LIVE_TRADING=true in .env to enable."
            )
        contract = self._contract(order.instrument)
        ib_order = self._build_ib_order(order)
        # placeOrder is fire-and-forget; the trade-status update arrives
        # async. But the call itself can wedge if the gateway's order book
        # isn't accepting submissions — bound it.
        trade = self._bounded("placeOrder", lambda: self._ib.placeOrder(contract, ib_order))
        logger.bind(broker=self.name, symbol=order.instrument.symbol).info(
            f"submitted {order.side.value} {order.quantity} {order.instrument.symbol} "
            f"as {order.order_type.value}"
        )
        # Best-effort: check whether the gateway rejected the order
        # synchronously (most commonly: insufficient buying power, bad
        # contract, market closed in non-LRPO mode). If trade.log has any
        # 'Reject' entries within the first second, surface them. We
        # don't WAIT for the trade to settle — that's still async.
        try:
            import time

            time.sleep(0.5)
            rejected = self._extract_reject_reason(trade)
            if rejected:
                raise BrokerError(
                    f"IBKR rejected {order.side.value} {order.quantity} "
                    f"{order.instrument.symbol}: {rejected}"
                )
        except BrokerError:
            raise
        except Exception:
            # Don't let log parsing fail order submission.
            pass
        return order

    @staticmethod
    def _extract_reject_reason(trade: Any) -> str | None:
        """Pull out the first explicit reject/error reason from an
        ib-async Trade object. Returns None if there's no rejection."""
        log = getattr(trade, "log", None) or []
        for entry in log:
            status = (getattr(entry, "status", "") or "").lower()
            msg = getattr(entry, "message", None) or getattr(entry, "errorMessage", None)
            if not msg:
                continue
            if "reject" in status or "cancelled" in status:
                return str(msg)[:300]
        order_status = getattr(getattr(trade, "orderStatus", None), "status", "").lower()
        if order_status in ("apicancelled", "cancelled", "inactive", "rejected"):
            why = getattr(trade.orderStatus, "whyHeld", "") or ""
            return f"status={order_status} whyHeld={why}"[:300]
        return None

    def cancel_order(self, client_order_id: str) -> None:
        self._ensure_connected()
        open_trades = self._bounded("openTrades", self._ib.openTrades)
        for trade in open_trades:
            if getattr(trade.order, "orderRef", None) == client_order_id:
                self._bounded("cancelOrder", lambda t=trade: self._ib.cancelOrder(t.order))
                return
        # Already terminal or never seen — non-fatal.
        logger.bind(broker=self.name).warning(
            f"no open trade matches client_order_id={client_order_id!r}; treating cancel as a no-op"
        )

    # ------------------------------------------------------------- state

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        raw = self._bounded("positions", self._ib.positions)
        out: list[Position] = []
        for p in raw:
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
        summary = self._bounded("accountSummary", self._ib.accountSummary)
        for row in summary:
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

    # --------------------------------------------------------------- FX

    def get_balances(self) -> dict[str, float]:
        r"""Return per-currency cash balances from the account.

        Used by the Telegram ``/balances`` command and as a pre-flight
        check before placing FX-converted orders. We aggregate
        ``CashBalance`` rows from ``accountValues``; the same currency
        may appear multiple times across accounts so we sum.
        """
        self._ensure_connected()
        raw = self._bounded("accountValues", self._ib.accountValues)
        out: dict[str, float] = {}
        for row in raw:
            tag = getattr(row, "tag", None)
            ccy = getattr(row, "currency", None) or "BASE"
            if tag != "CashBalance":
                continue
            try:
                amount = float(getattr(row, "value", 0.0))
            except (TypeError, ValueError):
                continue
            out[ccy] = out.get(ccy, 0.0) + amount
        return out

    def get_fx_rate(self, base_ccy: str, quote_ccy: str) -> float:
        r"""Spot rate ``base_ccy``/``quote_ccy`` — i.e. how many units of
        ``quote_ccy`` one unit of ``base_ccy`` buys right now.

        Uses yfinance for the reference quote (free, slightly delayed).
        Not used for trade execution — that uses IBKR's market price.
        This is purely for showing rates to the operator before they
        confirm a conversion.
        """
        import yfinance as yf

        symbol = f"{base_ccy}{quote_ccy}=X"
        df = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
        if df.empty:
            raise BrokerError(f"no FX quote available for {symbol}")
        close = df["Close"]
        # yfinance can return either Series or 1-col DataFrame here.
        if hasattr(close, "iloc"):
            try:
                val = float(close.iloc[-1])
            except TypeError:
                val = float(close.iloc[-1, 0])
            return val
        return float(close[-1])

    def convert_currency(self, *, from_ccy: str, to_ccy: str, from_amount: float) -> dict[str, Any]:
        r"""Spend ``from_amount`` of ``from_ccy`` at market to receive ``to_ccy``.

        Submits a market order on IDEALPRO. We use IBKR's ``cashQty`` so the
        operator can specify exactly how much of the source currency to
        spend, rather than computing a target quantity from a stale rate.

        Returns a dict with submission details; the actual fill comes
        back asynchronously and is captured by the next account snapshot.
        """
        from ib_async import Forex
        from ib_async import Order as IbOrder

        if from_ccy == to_ccy:
            raise BrokerError(f"from_ccy and to_ccy are identical ({from_ccy})")
        if from_amount <= 0:
            raise BrokerError(f"from_amount must be > 0, got {from_amount}")

        # IBKR Forex contract convention: USD comes before CHF/JPY/CAD;
        # EUR before USD/JPY/GBP/CHF; GBP before USD/JPY/CHF. We pick the
        # pair whose base/quote sides we recognise, then decide direction.
        pair_base, pair_quote = _fx_pair_for(from_ccy, to_ccy)
        contract = Forex(pair_base + pair_quote)

        ib_order = IbOrder()
        ib_order.orderType = "MKT"
        ib_order.tif = "DAY"
        if from_ccy == pair_quote:
            # Spending the quote side → buying the base.
            ib_order.action = "BUY"
            ib_order.cashQty = float(from_amount)
        else:
            # Spending the base side → selling it.
            ib_order.action = "SELL"
            ib_order.totalQuantity = float(from_amount)

        trade = self._bounded(
            f"placeOrder-fx-{pair_base}{pair_quote}",
            lambda: self._ib.placeOrder(contract, ib_order),
        )
        logger.bind(broker=self.name).info(
            f"FX order: {ib_order.action} {pair_base}{pair_quote} "
            f"(spending {from_amount} {from_ccy} → {to_ccy})"
        )

        # Wait briefly for the broker to acknowledge — IBKR FX rejections
        # are common (below IdealPro minimum, currency leverage, etc.) and
        # arrive within a few seconds *after* placeOrder returns. Without
        # this poll we report "submitted" to Telegram even when the order
        # was immediately rejected, leaving the operator misled. We poll
        # for up to ~5s; if no terminal status by then we return as-before
        # (the trade is still in flight and the next cycle's get_fills
        # will surface it).
        rejected = self._poll_for_fx_rejection(trade, timeout_s=5.0)
        if rejected:
            raise BrokerError(
                f"IBKR rejected FX {ib_order.action} {pair_base}{pair_quote} "
                f"(spending {from_amount} {from_ccy} → {to_ccy}): {rejected}"
            )

        return {
            "pair": pair_base + pair_quote,
            "action": ib_order.action,
            "from_ccy": from_ccy,
            "to_ccy": to_ccy,
            "from_amount": float(from_amount),
        }

    def _poll_for_fx_rejection(self, trade: Any, *, timeout_s: float) -> str | None:
        """Poll a Trade object briefly for a terminal rejection.

        Reuses ``_extract_reject_reason``'s parsing, but loops so we catch
        async rejections that arrive after placeOrder returns. Returns the
        reject reason if found within ``timeout_s``, else None.
        """
        import time

        deadline = time.monotonic() + timeout_s
        last_reason: str | None = None
        while time.monotonic() < deadline:
            try:
                last_reason = self._extract_reject_reason(trade)
            except Exception:
                last_reason = None
            if last_reason:
                return last_reason
            time.sleep(0.25)
        return None

    def get_fills(self, *, since: datetime | None = None) -> list[Fill]:
        self._ensure_connected()
        raw = self._bounded("fills", self._ib.fills)
        out: list[Fill] = []
        for f in raw:
            exec_ = f.execution
            ts = getattr(exec_, "time", None)
            if ts is None:
                continue
            if since is not None and ts < since:
                continue
            # ib-async's Fill exposes contract/execution/commissionReport but
            # NOT a top-level .order — that was an earlier API. The orderRef
            # (our client_order_id) lives on the Execution itself; fall back
            # to the IBKR orderId if it's missing.
            out.append(
                Fill(
                    order_id=getattr(exec_, "orderRef", "") or str(getattr(exec_, "orderId", "")),
                    ts=ts,
                    quantity=float(exec_.shares),
                    price=float(exec_.price),
                    commission=float(getattr(f.commissionReport, "commission", 0.0) or 0.0),
                    venue=getattr(exec_, "exchange", None),
                )
            )
        return out


def _fx_pair_for(a: str, b: str) -> tuple[str, str]:
    r"""Return the canonical ``(base, quote)`` IBKR Forex pair for two
    currencies, in either order.

    IBKR's convention: the more-conventional base currency comes first.
    USD is base in USDCHF / USDJPY / USDCAD; EUR is base in EURUSD;
    GBP is base in GBPUSD, etc. We don't try to cover every cross —
    just the handful that matter for a typical retail SP500 trader.
    """
    # Ordering: the currency that appears FIRST in this list is the base.
    priority = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
    a, b = a.upper(), b.upper()
    if a not in priority or b not in priority:
        raise BrokerError(f"unsupported FX pair: {a}/{b}")
    if priority.index(a) < priority.index(b):
        return a, b
    return b, a


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
