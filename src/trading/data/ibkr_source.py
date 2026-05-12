"""IBKR adapter via ``ib-async`` — FX (IDEALPRO) and intraday equities.

Requires a running IB Gateway / TWS reachable at ``settings.ibkr_host:port``.
Connection is established lazily; tests use a mock ``ib`` object so this
module imports cleanly even without a gateway.

Why ib-async (not ib_insync, not the raw API)
---------------------------------------------
ib-async is the maintained fork; ib_insync's author stepped away. The async
boundary is unavoidable for IBKR because the underlying TWS protocol is
push-based with request IDs that need correlation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from trading.core.config import settings
from trading.core.types import AssetClass, Instrument
from trading.data.base import BAR_COLUMNS, Frequency, empty_bars_frame, validate_bars_frame

if TYPE_CHECKING:  # pragma: no cover
    pass


# Canonical pandas alias -> IBKR bar size string. IBKR is fussy about
# exact spelling here.
_FREQ_TO_IBKR_BAR: dict[Frequency, str] = {
    "1min": "1 min",
    "5min": "5 mins",
    "15min": "15 mins",
    "30min": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1D": "1 day",
    "1W": "1 week",
}


def _ibkr_duration(start: datetime, end: datetime) -> str:
    """IBKR wants a ``durationStr`` like ``'30 D'`` or ``'2 Y'`` instead of a
    start date. Pick the smallest unit that covers the requested span."""
    delta = end - start
    days = max(1, delta.days + 1)
    if days <= 30:
        return f"{days} D"
    if days <= 365:
        return f"{(days + 6) // 7} W"
    return f"{(days + 364) // 365} Y"


class IbkrSource:
    """``DataSource`` adapter for IBKR via ib-async.

    The adapter is synchronous-on-the-outside: it runs the underlying async
    call on an event loop so callers don't need to await. Strategies and the
    backtester are synchronous; only the live runner is async.
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

    async def _ensure_connected(self) -> Any:
        if self._ib is None:
            from ib_async import IB  # lazy import — heavy

            self._ib = IB()
        if not self._ib.isConnected():
            await self._ib.connectAsync(self._host, self._port, clientId=self._client_id)
        return self._ib

    def _contract(self, instrument: Instrument) -> Any:
        from ib_async import Forex, Stock  # lazy import

        if instrument.asset_class == AssetClass.FX:
            # IBKR FX symbols come pre-joined ("EURUSD"); ib-async wants
            # the pair, optional venue defaults to IDEALPRO.
            return Forex(instrument.symbol)
        if instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF):
            return Stock(
                symbol=instrument.symbol,
                exchange=instrument.exchange or "SMART",
                currency=instrument.currency,
            )
        raise ValueError(
            f"IBKR adapter does not yet support asset_class={instrument.asset_class.value}"
        )

    async def _get_bars_async(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
    ) -> pd.DataFrame:
        ib = await self._ensure_connected()
        bar_size = _FREQ_TO_IBKR_BAR[freq]
        duration = _ibkr_duration(start, end)
        what_to_show = "MIDPOINT" if instrument.asset_class == AssetClass.FX else "TRADES"

        bars = await ib.reqHistoricalDataAsync(
            self._contract(instrument),
            endDateTime=end.astimezone(timezone.utc).strftime("%Y%m%d-%H:%M:%S"),
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=2,  # UTC seconds since epoch
        )
        if not bars:
            return empty_bars_frame()

        df = pd.DataFrame(
            [
                {
                    "ts": pd.Timestamp(b.date, tz="UTC")
                    if not isinstance(b.date, pd.Timestamp)
                    else (b.date.tz_convert("UTC") if b.date.tzinfo else b.date.tz_localize("UTC")),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": float(b.volume) if b.volume is not None else 0.0,
                    "adj_close": pd.NA,
                }
                for b in bars
            ]
        )
        df = df.set_index("ts").sort_index()
        df = df.loc[:, list(BAR_COLUMNS)]
        # IBKR can return bars slightly outside the requested window when
        # we asked for a duration > requested span. Trim.
        df = df[(df.index >= start) & (df.index <= end)]
        return validate_bars_frame(df)

    def get_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        freq: Frequency,
    ) -> pd.DataFrame:
        if freq not in _FREQ_TO_IBKR_BAR:
            raise ValueError(f"IBKR does not support freq={freq!r}")
        # Pace requests minimally — IBKR throttles ~6 historical reqs / sec.
        # We don't pace globally here; the caller's per-symbol loop is the
        # natural throttle. A real backfill should add explicit pacing.
        coro = self._get_bars_async(instrument, start, end, freq)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — synchronous caller. Run in a fresh loop.
            return asyncio.run(coro)
        # Inside an existing loop (notebook, runner): schedule onto it.
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()


# Module-level convenience for symmetry with the other adapters.
_FREQ_TO_IBKR = _FREQ_TO_IBKR_BAR
_ = timedelta  # silence "imported but unused" if a future refactor drops it
