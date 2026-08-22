"""Core domain types used everywhere in the system.

Design rules
------------
* All times are timezone-aware ``datetime`` (UTC at the boundary, converted to
  exchange-local time only for display).
* All money / price types are ``Decimal``-free for performance reasons; we use
  ``float`` and accept the ~1e-15 representation error. The risk manager
  re-quantizes to broker tick sizes before submitting.
* Models are immutable (`frozen=True`) to make signals/orders safe to pass
  across async boundaries without aliasing bugs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrEnum(str, Enum):
    """str-mixin Enum, equivalent to stdlib ``enum.StrEnum`` (3.11+).

    Defined locally so the codebase runs on Python 3.10 too.
    """

    def __str__(self) -> str:  # match 3.11 StrEnum behavior
        return str(self.value)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    FX = "fx"
    CRYPTO = "crypto"
    FUTURE = "future"
    OPTION = "option"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    MOC = "moc"  # market-on-close
    LOC = "loc"  # limit-on-close


class TimeInForce(StrEnum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(StrEnum):
    PENDING = "pending"  # created locally, not sent
    SUBMITTED = "submitted"  # acknowledged by broker
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


class Instrument(BaseModel):
    """Canonical instrument identifier.

    For equities: ``symbol`` is the ticker on its primary exchange (e.g. ``AAPL``).
    For FX: ``symbol`` is the pair joined without separator (e.g. ``EURUSD``).
    For crypto: ``symbol`` follows ccxt convention (e.g. ``BTC/USDT``) and
    ``exchange`` is required.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    asset_class: AssetClass
    exchange: str | None = None
    currency: str = "USD"
    multiplier: float = 1.0  # contract multiplier for futures/options
    min_tick: float = 0.01  # smallest price increment

    @property
    def key(self) -> str:
        """Stable cache key, e.g. ``equity:AAPL`` or ``crypto:binance:BTC/USDT``."""
        parts = [self.asset_class.value]
        if self.exchange:
            parts.append(self.exchange)
        parts.append(self.symbol)
        return ":".join(parts)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class Bar(BaseModel):
    """OHLCV bar. ``ts`` is the bar OPEN time, timezone-aware UTC."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    # Adjusted close kept separate so we never silently use adjusted prices for
    # execution-style logic (only for return calculations).
    adj_close: float | None = None

    @field_validator("ts")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Bar.ts must be timezone-aware")
        return v


# ---------------------------------------------------------------------------
# Signals, orders, fills, positions
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """A target portfolio expressed as a weight per instrument.

    Strategies output Signals; the portfolio combiner aggregates them; the
    risk manager translates the final target weights into order deltas.
    Using *target weights* (rather than buy/sell orders directly) is what
    lets us combine many strategies cleanly.
    """

    model_config = ConfigDict(frozen=True)

    ts: datetime
    strategy: str  # producer name, for attribution
    target_weights: dict[str, float]  # instrument.key -> weight in [-1, 1]
    confidence: float = 1.0  # 0..1 — combiner can use this
    metadata: dict[str, str] = Field(default_factory=dict)


class Order(BaseModel):
    """An instruction to the broker. Created by the risk manager, never by
    strategies directly."""

    model_config = ConfigDict(frozen=True)

    client_order_id: str
    instrument: Instrument
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    created_at: datetime


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True)

    order_id: str
    ts: datetime
    quantity: float
    price: float
    commission: float = 0.0
    venue: str | None = None
    #: The broker's own execution id (IBKR ``execution.execId``). The only
    #: thing that distinguishes one execution from another: reconciliation
    #: deliberately re-reads a window of fills every cycle, so without a
    #: stable identity the same execution is stored again on every pass.
    #: None for brokers that do not supply one — the store then falls back
    #: to a composite key.
    exec_id: str | None = None


class Position(BaseModel):
    """Net position in a single instrument."""

    model_config = ConfigDict(frozen=True)

    instrument: Instrument
    quantity: float  # signed; negative = short
    avg_price: float  # average entry price (signed-quantity weighted)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


class AccountSnapshot(BaseModel):
    """A point-in-time view of the broker account. The runner persists one
    of these per cycle; the risk manager reads the latest to size new
    orders against current equity."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    cash: float
    equity: float  # cash + market value of positions
    positions: dict[str, Position] = Field(default_factory=dict)  # keyed by instrument.key
    # ISO-4217 code of the account's base currency. IBKR brokerage accounts
    # have one base currency in which cash, equity and P&L are denominated;
    # holdings in other currencies are converted into it on the broker side.
    # We surface this so the bot can display "CHF 997k" instead of a
    # misleading hardcoded "$" prefix when the account isn't USD.
    base_currency: str = "USD"
    # Per-currency cash balances (e.g. ``{"CHF": 140_000, "USD": 308_000}``).
    # Populated by brokers that distinguish multi-currency cash;
    # Simulator-backed snapshots leave this empty. The Telegram bot reads
    # this to show per-currency breakdowns without a second broker call.
    cash_by_currency: dict[str, float] = Field(default_factory=dict)
    # Which book this snapshot describes. "account" is everything the
    # broker holds. "managed" is that minus the operator's pinned
    # positions, with their value already taken out of `equity` — the
    # view the risk manager sizes and halts against, so that a personal
    # holding the desk may not trade also cannot trip its kill switches.
    # Both are honest snapshots; confusing them is how a 33% "drawdown"
    # gets reported on an account that did not move, which is why the
    # kill-switch baselines record the scope they were stamped in.
    scope: Literal["account", "managed"] = "account"
    # Base-currency units per 1 unit of each foreign currency, as the
    # broker reported them when this snapshot was taken. Persisted with
    # the snapshot because everything that reads one back later — the
    # Telegram views, the daily report — needs to value a USD position in
    # CHF and has no broker connection of its own to ask.
    fx_rates: dict[str, float] = Field(default_factory=dict)
    # Base-currency value removed to build a managed view, and the
    # symbols it came from. Empty on an "account" snapshot.
    excluded_value: float = 0.0
    excluded_symbols: tuple[str, ...] = ()

    @field_validator("ts")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("AccountSnapshot.ts must be timezone-aware")
        return v


# ---------------------------------------------------------------------------
# Risk events
# ---------------------------------------------------------------------------

RiskAction = Literal["allow", "scale", "reject", "halt"]


class RiskDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: RiskAction
    reason: str
    scale_factor: float = 1.0  # used when action == "scale"
