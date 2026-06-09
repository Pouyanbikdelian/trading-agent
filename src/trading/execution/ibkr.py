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
        # Background event loop owned by ib-async's transport. Populated
        # lazily in connect() — only when we instantiate a real IB(), not
        # when tests inject a stub. See _ensure_ib_loop_thread.
        self._ib_loop: asyncio.AbstractEventLoop | None = None
        self._ib_loop_thread: Any | None = None

    def _is_live_port(self) -> bool:
        return self._port in self._LIVE_PORTS

    # --------------------------------------------------------- lifecycle

    def connect(self) -> None:
        if self._ib is None:
            from ib_async import IB  # lazy import

            # ib-async's transport is bound to whichever event loop awaits
            # connectAsync. Every subsequent read/write must happen on
            # that same loop — or the socket data goes nowhere and calls
            # like accountSummary hang forever (prod incident 2026-05-22).
            #
            # We solve this with a dedicated daemon thread that runs the
            # loop continuously, and dispatch all async work via
            # asyncio.run_coroutine_threadsafe. Sync callers from any
            # thread (APScheduler workers, ThreadPoolExecutor pool, etc.)
            # safely reach the transport that way.
            self._ensure_ib_loop_thread()
            self._ib = IB()
        if not self._ib.isConnected():
            # Hard 60s ceiling on the handshake — useful when the gateway
            # is still in its login dance.
            self._await_async(
                self._ib.connectAsync(self._host, self._port, clientId=self._client_id),
                timeout=60.0,
            )
        self._connected = True
        logger.bind(broker=self.name).info(
            f"connected ibkr@{self._host}:{self._port} client_id={self._client_id}"
        )

    def _ensure_ib_loop_thread(self) -> None:
        """Start the daemon thread that owns ib-async's event loop.

        Idempotent: only the first call creates the thread. Subsequent
        connects (re-handshakes after gateway bounce) reuse the same loop.
        """
        if self._ib_loop is not None:
            return
        import threading

        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_runner, daemon=True, name="ibkr-loop")
        thread.start()
        if not ready.wait(timeout=5.0):
            raise BrokerError("ibkr-loop thread did not start within 5s")
        self._ib_loop = loop
        self._ib_loop_thread = thread

    def _await_async(self, coro: Any, *, timeout: float) -> Any:
        """Await ``coro`` with a hard timeout, on the ib-loop thread if
        one exists.

        When the IB instance was constructed via ``connect()`` (production
        path), ``self._ib_loop`` is set and we dispatch via
        ``run_coroutine_threadsafe`` so the transport's owning loop runs
        the work. When ``ib`` is a test stub injected via the constructor,
        no loop thread exists — we run the coroutine on a transient loop
        on the current thread; stubs don't have a real transport so this
        is safe.
        """
        import concurrent.futures

        wrapped = asyncio.wait_for(coro, timeout=timeout)
        if self._ib_loop is not None:
            fut = asyncio.run_coroutine_threadsafe(wrapped, self._ib_loop)
            try:
                # Small buffer over the inner timeout so the cancellation
                # has a chance to propagate before we raise here.
                return fut.result(timeout=timeout + 5.0)
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                fut.cancel()
                raise
        # Test fallback: stubs run on whatever loop we hand them.
        return asyncio.new_event_loop().run_until_complete(wrapped)

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
                self._trigger_gateway_restart(reason=f"reconnect failed during {what}")
                raise first_err from None
            try:
                return self._call_with_timeout(what, fn, timeout)
            except BrokerTimeoutError:
                # Second timeout after fresh CLIENT-side connect — the
                # gateway's IBKR session is genuinely dead even though TCP
                # is alive. Only a container restart fixes this. Trigger
                # the restart, then re-raise so this cycle aborts cleanly;
                # the next cycle will land on a fresh gateway.
                logger.bind(broker=self.name).warning(
                    f"{what} still timing out after reconnect — triggering gateway restart"
                )
                self._trigger_gateway_restart(reason=f"{what} timed out twice")
                raise

    # Gateway-container name for the docker API call. Matches docker-compose.yml.
    _GATEWAY_CONTAINER_NAME: str = "ibkr-gateway"
    # Default location of the docker daemon socket inside the trader container.
    # docker-compose mounts /var/run/docker.sock here.
    _DOCKER_SOCKET_PATH: str = "/var/run/docker.sock"

    def _trigger_gateway_restart(self, *, reason: str) -> None:
        """Restart the gateway container by talking to the docker daemon
        directly via the unix socket (Python stdlib — no docker CLI needed).

        Requires the docker socket to be mounted into the trader container
        (see docker-compose.yml). Disable by setting env
        ``ENABLE_GATEWAY_AUTO_RESTART=false`` in .env.

        Why we do this from the trader: the gateway's TCP port can stay
        open while its IBKR API session is dead. Our compose-level TCP
        healthcheck doesn't see that — only the trader does, because only
        the trader makes real API calls. So the trader is the right
        owner of the "session is dead, restart the container" decision.

        Safe to call repeatedly: each call is rate-limited to one restart
        per ``_RESTART_COOLDOWN_S`` so a bad cycle can't trigger restart
        storms. On failure (socket missing, container not found) we log
        and continue — the cycle is already aborting.
        """
        import http.client
        import os
        import socket
        import time

        if os.environ.get("ENABLE_GATEWAY_AUTO_RESTART", "true").lower() in {"false", "0", "no"}:
            logger.bind(broker=self.name).info(
                "gateway auto-restart disabled (ENABLE_GATEWAY_AUTO_RESTART=false)"
            )
            return

        now = time.monotonic()
        last = getattr(self, "_last_restart_ts", 0.0)
        if now - last < self._RESTART_COOLDOWN_S:
            elapsed = now - last
            logger.bind(broker=self.name).info(
                f"gateway restart suppressed (last one was {elapsed:.0f}s ago, "
                f"cooldown {self._RESTART_COOLDOWN_S:.0f}s)"
            )
            return

        if not os.path.exists(self._DOCKER_SOCKET_PATH):
            logger.bind(broker=self.name).warning(
                f"docker.sock not mounted at {self._DOCKER_SOCKET_PATH}; "
                "cannot self-restart gateway. Add the bind-mount to docker-compose.yml "
                "or run `docker compose restart ib-gateway` manually from the host."
            )
            return

        logger.bind(broker=self.name).warning(
            f"triggering docker restart of {self._GATEWAY_CONTAINER_NAME} — reason: {reason}"
        )
        try:
            self._docker_restart_via_socket(self._GATEWAY_CONTAINER_NAME)
        except Exception:
            logger.bind(broker=self.name).exception(
                f"docker restart of {self._GATEWAY_CONTAINER_NAME} failed"
            )
            return

        self._last_restart_ts = now
        logger.bind(broker=self.name).info(
            f"gateway restart issued; container will be back in ~90s. "
            "Next cycle should land on a fresh session."
        )

    def _docker_restart_via_socket(self, container_name: str, timeout: float = 30.0) -> None:
        """POST /containers/<name>/restart against the docker unix socket.

        Stdlib-only (http.client + socket). Cleaner than shelling out to
        the docker CLI — no extra binary in the image. Raises on any
        non-2xx response or transport error; caller logs.
        """
        import http.client
        import socket as _socket

        class _UDSConnection(http.client.HTTPConnection):
            """HTTPConnection that talks over a unix domain socket instead of TCP."""

            def __init__(self, sock_path: str, timeout: float) -> None:
                super().__init__("localhost", timeout=timeout)
                self._sock_path = sock_path

            def connect(self) -> None:  # type: ignore[override]
                s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                s.connect(self._sock_path)
                self.sock = s

        conn = _UDSConnection(self._DOCKER_SOCKET_PATH, timeout=timeout)
        try:
            # t=10 → docker SIGTERMs the container, waits up to 10s for graceful
            # exit, then SIGKILLs. Gateway responds to TERM cleanly so 10s is fine.
            conn.request("POST", f"/containers/{container_name}/restart?t=10")
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")[:300]
            if resp.status >= 300:
                raise RuntimeError(
                    f"docker API returned {resp.status} {resp.reason}: {body}"
                )
        finally:
            conn.close()

    # Rate-limit gateway restarts. Repeated triggers within this window are
    # suppressed — protects against restart storms when many things are
    # failing at once (e.g. multiple cycles queued during a gateway outage).
    _RESTART_COOLDOWN_S: float = 180.0

    def _call_with_timeout(self, what: str, fn: Any, timeout: float | None) -> Any:
        """Inner: one bounded call, no retry.

        Two paths:

        * **Async-aware path** — if ``fn`` is an ib-async sync wrapper
          like ``self._ib.accountSummary`` (which has a real coroutine
          sibling ``self._ib.accountSummaryAsync``), we MUST run the
          coroutine on the loop that owns the transport. Calling the
          sync wrapper from a worker thread creates a fresh per-thread
          event loop, sends the request on that loop, and waits forever
          for a response that arrives on a different loop — the 30s
          "timeout" we saw in prod was this, not a real broker hang.

        * **Direct path** — for cached reads (``positions``, ``fills``,
          ``openTrades``, ``accountValues``) and fire-and-forget sends
          (``placeOrder``, ``cancelOrder``), no awaiting happens
          internally. We run them on a ThreadPoolExecutor purely for
          the hard timeout escape hatch.
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FutTimeout

        timeout = timeout if timeout is not None else self.DEFAULT_API_TIMEOUT_S

        async_fn = self._async_variant(fn)
        if async_fn is not None and self._ib_loop is not None:
            try:
                return self._await_async(async_fn(), timeout=timeout)
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
                raise BrokerTimeoutError(
                    f"IBKR {what} timed out after {timeout:.0f}s — gateway likely "
                    "has a dead broker session (try restarting ib-gateway container)"
                ) from e

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

    def _async_variant(self, fn: Any) -> Any | None:
        """If ``fn`` is an ib-async sync method with an ``*Async``
        sibling, return that sibling. Otherwise return None.

        We use this so callers can keep passing the familiar sync
        bindings (``self._ib.accountSummary``) while the broker quietly
        routes them through the proper async-on-the-right-loop path.
        """
        name = getattr(fn, "__name__", None)
        bound_to = getattr(fn, "__self__", None)
        if not name or bound_to is None or name.endswith("Async"):
            return None
        if bound_to is not self._ib:
            return None
        async_method = getattr(self._ib, name + "Async", None)
        if async_method is None or not asyncio.iscoroutinefunction(async_method):
            return None
        return async_method

    def _reconnect_session(self) -> None:
        """Force a fresh handshake after a session drop.

        After a gateway bounce ib-async's ``isConnected()`` can briefly
        report True while the TCP teardown propagates. If we route
        through ``connect()`` it short-circuits on that stale True and
        no real ``connectAsync`` runs — the next API call then dies
        with ``ConnectionError: Not connected``. We instead disconnect
        explicitly and call ``connectAsync`` directly so the handshake
        runs unconditionally — and on the ib-loop thread so the new
        transport is bound to the same loop our other calls use.
        """
        import contextlib

        with contextlib.suppress(Exception):
            self.disconnect()
        self._connected = False

        if self._ib is None:
            # Cold start (no prior IB instance) — defer to the normal
            # connect() path which lazy-imports and instantiates IB.
            self.connect()
            return

        self._await_async(
            self._ib.connectAsync(self._host, self._port, clientId=self._client_id),
            timeout=60.0,
        )
        self._connected = True
        logger.bind(broker=self.name).info(
            f"reconnected ibkr@{self._host}:{self._port} client_id={self._client_id}"
        )

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
        # Collect every plausible base-currency hint we see and pick at the
        # end. Different tags carry the currency differently: NetLiquidation
        # is normally per-account (base ccy), but during a gateway wedge it
        # can come through with currency=None. AvailableFunds and
        # TotalCashValue are also base-currency-only in IBKR's schema.
        # Voting across multiple tags is much more robust than relying on
        # one row whose currency attribute might be missing.
        currency_votes: dict[str, int] = {}
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
            # Base-currency candidates: only tags IBKR reports against the
            # account's base. CashBalance / AccruedCash come per-currency
            # and would skew the vote toward whichever currency happens to
            # appear first.
            if tag in ("NetLiquidation", "AvailableFunds", "TotalCashValue"):
                ccy = getattr(row, "currency", None)
                if ccy and ccy != "BASE":
                    currency_votes[str(ccy)] = currency_votes.get(str(ccy), 0) + 1
        base_currency = (
            max(currency_votes.items(), key=lambda kv: kv[1])[0]
            if currency_votes
            else "USD"
        )
        positions = {p.instrument.key: p for p in self.get_positions()}
        # Per-currency cash MUST ride along on the snapshot: the risk
        # manager's no-margin check reads account.cash_by_currency and,
        # when it's empty, falls back to {base_currency: cash}. On a
        # CHF-base account buying USD stocks that fallback sees zero USD
        # forever, so every cycle is rejected at full basket notional —
        # and FX conversions can't fix it because the check never sees
        # the USD side (June 2026 incident: 3+ weeks of refused cycles).
        # Best-effort: a failed accountValues read leaves the dict empty,
        # which keeps the old (conservative) fallback behaviour.
        cash_by_currency: dict[str, float] = {}
        try:
            cash_by_currency = self.get_balances()
        except Exception as e:
            logger.bind(broker=self.name).warning(
                f"get_balances failed in get_account ({type(e).__name__}: {e!r}); "
                "cash_by_currency left empty — no-margin check will use base-ccy fallback"
            )
        return AccountSnapshot(
            ts=ts,
            cash=cash,
            equity=equity,
            positions=positions,
            base_currency=base_currency,
            cash_by_currency=cash_by_currency,
        )

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
