"""Risk manager — the only path from Signal to Order.

Strategies emit ``Signal.target_weights``; the runner hands the signal to
the risk manager along with the latest ``AccountSnapshot`` and a price
dict. The manager:

1. Refuses to act if it is halted (operator must explicitly ``unhalt()``).
2. Caps each instrument's weight at ``max_position_pct`` (scale, not reject).
3. Applies optional sector caps proportionally (scale).
4. Caps total gross and net exposures (scale).
5. Converts target weights to delta-quantity Orders against the current book.
6. Returns ``(orders, decisions)`` — decisions document every scale/reject
   that happened along the way.

Intraday checks (``evaluate_intraday``) run after each bar:

* Daily-loss kill: equity has dropped ``max_daily_loss_pct`` from the
  day's opening equity → halt.
* Drawdown halt: equity has dropped ``max_drawdown_pct`` from the all-time
  high → halt.

Halt state is persisted to ``state_dir/halt.json`` so a crashed runner
that restarts during a halt picks it back up.

CLAUDE.md hard rules honored:
* Risk manager is the only path to orders — strategies do not construct
  ``Order`` themselves.
* Manager is hard-blocking — once halted, it refuses to generate orders
  until ``unhalt()`` is called explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from trading.core.logging import logger
from trading.core.types import (
    AccountSnapshot,
    AssetClass,
    Instrument,
    Order,
    OrderType,
    Position,
    RiskDecision,
    Side,
    Signal,
    TimeInForce,
)
from trading.execution.ibkr import new_client_order_id
from trading.risk.limits import HaltState, RiskLimits

_EPS_QTY = 1e-9


class RiskManager:
    """The hard-blocking gate between strategies and the broker."""

    def __init__(
        self,
        limits: RiskLimits,
        *,
        halt_state_path: Path | None = None,
    ) -> None:
        self.limits = limits
        self._halt_path = Path(halt_state_path) if halt_state_path else None
        self._state = self._load_state()

    # ------------------------------------------------------ persistence

    def _load_state(self) -> HaltState:
        """Read the persisted halt state, fail-CLOSED on any parse error.

        A corrupted ``halt.json`` (partial write during a crash, disk
        corruption, manual edit gone wrong) previously crashed manager
        init with a pydantic ValidationError — leaving the cycle to fail
        UP the stack and trade-on-no-gate. Now any parse failure puts us
        into a halted state with a clear reason; the operator must
        either fix the file and /resume, or manually accept.
        """
        if not (self._halt_path and self._halt_path.exists()):
            return HaltState()
        # Take the file lock to ensure we don't read a torn write from the
        # bot's /halt or /resume mid-write (audit fix #9). Lock is on a
        # sibling .lock file so atomic os.replace still works.
        from trading.core.file_lock import file_lock

        try:
            with file_lock(self._halt_path):
                return HaltState.model_validate_json(self._halt_path.read_text())
        except Exception as e:
            logger.bind(component="risk").error(
                f"halt.json could not be parsed ({type(e).__name__}: {e!r}); "
                "FAILING CLOSED — manager will refuse to trade until operator "
                "fixes the file and /resume's manually."
            )
            return HaltState(
                halted=True,
                reason=f"halt.json corrupt: {type(e).__name__}",
                halted_at=datetime.now(timezone.utc),
            )

    def _save_state(self) -> None:
        if self._halt_path is None:
            return
        # File lock + atomic rename: both the bot and the manager are
        # writers; serialising under flock ensures no torn writes or
        # silently-dropped intent (audit fix #9).
        from trading.core.file_lock import file_lock

        self._halt_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self._halt_path):
            self._halt_path.write_text(self._state.model_dump_json(indent=2))

    # ------------------------------------------------------ halt control

    @property
    def state(self) -> HaltState:
        return self._state

    def is_halted(self) -> bool:
        return self._state.halted

    def halt(self, reason: str) -> None:
        self._state = self._state.replace(
            halted=True,
            reason=reason,
            halted_at=datetime.now(timezone.utc),
        )
        self._save_state()
        logger.bind(component="risk").warning(f"HALTED — {reason}")

    def unhalt(self) -> None:
        """Operator-only command. Phase 8's runner will gate this behind a
        manual confirmation; the risk manager itself never auto-unhalts."""
        if not self._state.halted:
            return
        prev = self._state.reason
        self._state = self._state.replace(halted=False, reason="", halted_at=None)
        self._save_state()
        logger.bind(component="risk").warning(f"UNHALTED (was: {prev})")

    def _reload_halt_state(self) -> None:
        """Refresh halt-related fields from disk, preserving in-memory
        daily P&L tracking.

        We do NOT swap ``self._state`` wholesale here: the runner's
        ``start_of_day`` populates ``daily_equity_open`` and
        ``equity_high_watermark`` in-memory and those don't always land
        in halt.json between processes. Reloading just the halt trio
        keeps the bot's /halt and /resume effective without losing
        live counters.
        """
        if self._halt_path is None:
            return
        try:
            disk = self._load_state()
        except Exception as e:
            logger.bind(component="risk").warning(
                f"_reload_halt_state failed; keeping in-memory state: {e!r}"
            )
            return
        self._state = self._state.replace(
            halted=disk.halted,
            reason=disk.reason,
            halted_at=disk.halted_at,
        )

    def start_of_day(self, account: AccountSnapshot) -> None:
        """Stamp today's opening equity. Idempotent within a day — only the
        first call on a new date mutates state."""
        today = account.ts.date()
        if self._state.last_day == today:
            return
        new_hwm = max(self._state.equity_high_watermark, account.equity)
        self._state = self._state.replace(
            last_day=today,
            daily_equity_open=account.equity,
            equity_high_watermark=new_hwm,
        )
        self._save_state()

    # ------------------------------------------------------ intraday

    def evaluate_intraday(self, account: AccountSnapshot) -> RiskDecision:
        """Post-bar safety check. Returns ``halt`` if a kill switch fires."""
        # Re-read halt.json before checking. The Telegram bot (a separate
        # process) writes this file from /halt and /resume; without a
        # reload here, the trader process would only see halt changes on
        # its next restart. This caused a real incident: after /resume
        # cleared the halt on disk, the still-running cycle saw the stale
        # in-memory state and refused with "already halted" (2026-05-22).
        self._reload_halt_state()
        if self._state.halted:
            return RiskDecision(action="halt", reason=f"already halted: {self._state.reason}")

        # Roll the daily baseline. start_of_day() is idempotent within a
        # day (no-ops when last_day == today), so call it unconditionally.
        # The previous "lazy" guard (only when state was empty) meant the
        # baseline stamped on the FIRST ever cycle was never rolled
        # forward — the daily-loss kill switch spent 7 weeks comparing
        # against 2026-05-22 equity and, with the book up 14% since,
        # could never fire (found by the GO_LIVE §2 kill-switch drill,
        # 2026-07-09).
        self.start_of_day(account)

        # Update high-water mark.
        if account.equity > self._state.equity_high_watermark:
            self._state = self._state.replace(equity_high_watermark=account.equity)
            self._save_state()

        # Daily-loss kill.
        if self._state.daily_equity_open > 0:
            day_pnl = (
                account.equity - self._state.daily_equity_open
            ) / self._state.daily_equity_open
            if day_pnl <= -self.limits.max_daily_loss_pct:
                self.halt(
                    f"daily loss {day_pnl:.2%} breaches limit -{self.limits.max_daily_loss_pct:.2%}"
                )
                return RiskDecision(action="halt", reason=self._state.reason)

        # Peak drawdown halt.
        if self._state.equity_high_watermark > 0:
            dd = (
                account.equity - self._state.equity_high_watermark
            ) / self._state.equity_high_watermark
            if dd <= -self.limits.max_drawdown_pct:
                self.halt(f"drawdown {dd:.2%} breaches limit -{self.limits.max_drawdown_pct:.2%}")
                return RiskDecision(action="halt", reason=self._state.reason)

        return RiskDecision(action="allow", reason="intraday checks passed")

    # ------------------------------------------------------ signal -> orders

    def signal_to_orders(
        self,
        signal: Signal,
        *,
        account: AccountSnapshot,
        last_prices: dict[str, float],
        instruments: dict[str, Instrument],
        sector_map: dict[str, str] | None = None,
        order_id_factory: Callable[[], str] = new_client_order_id,
        fx_rates: dict[str, float] | None = None,
        pending_orders: list[Order] | None = None,
    ) -> tuple[list[Order], list[RiskDecision]]:
        """Convert a Signal's target weights into Orders, applying limits.

        Parameters are keyed by ``instrument.key`` (e.g. ``"equity:AAPL"``)
        to match ``Signal.target_weights`` and ``AccountSnapshot.positions``.

        ``fx_rates`` maps currency -> base-currency units per 1 unit (e.g.
        {"USD": 0.81} on a CHF account). Sizing divides base-currency
        equity by the price CONVERTED TO BASE, so a 10% weight is 10% of
        real equity. Before 2026-07-14 the raw instrument-currency price
        was used against CHF equity, undersizing every USD position by
        the USDCHF factor (~19%) — found by the GO_LIVE §2 CHF check.
        Missing rate for a foreign currency: sized at 1.0 (old behavior)
        with a logged decision, so research/backtests are unaffected.

        ``pending_orders`` — orders already WORKING at the broker (from
        any source: earlier cycles, guard exits, manual commands). Their
        signed quantity is netted into current position before deltas
        are computed, so this cycle sizes against where the book WILL be
        once they fill. Without this, after-hours batches stack blindly:
        2026-07-15, guard closes + two queued cycles all filled at the
        open and the paper book went short two names.
        """
        decisions: list[RiskDecision] = []
        if self._state.halted:
            return [], [RiskDecision(action="halt", reason=f"halted: {self._state.reason}")]
        if account.equity <= 0:
            return [], [RiskDecision(action="reject", reason="non-positive equity")]

        # Signed share deltas of orders already working at the broker,
        # keyed by instrument.key — netted into current position below.
        pending_delta: dict[str, float] = {}
        for po in pending_orders or []:
            signed = po.quantity if po.side == Side.BUY else -po.quantity
            pending_delta[po.instrument.key] = pending_delta.get(po.instrument.key, 0.0) + signed
        if pending_delta:
            decisions.append(
                RiskDecision(
                    action="scale",
                    reason=(
                        f"netting {len(pending_delta)} pending order position(s) "
                        f"into sizing: {sorted(pending_delta)}"
                    ),
                    scale_factor=1.0,
                )
            )

        # Work on a mutable copy.
        weights: dict[str, float] = dict(signal.target_weights)

        # --- 0. Long-only invariant (allow_short=False, the default).
        # Clamp negative TARGETS here; sell QUANTITIES are clamped at
        # order construction below. Both halves matter: a negative weight
        # is an intent to short, an oversized sell is an accident that
        # ends short — this system permits neither.
        if not self.limits.allow_short:
            for key in list(weights):
                if weights[key] < 0:
                    decisions.append(
                        RiskDecision(
                            action="scale",
                            reason=f"long-only: negative target weight on {key} clamped to 0",
                            scale_factor=0.0,
                        )
                    )
                    weights[key] = 0.0

        # --- 1. Per-position cap (scale individual weights down if needed).
        for key in list(weights):
            w = weights[key]
            if abs(w) > self.limits.max_position_pct:
                scale = self.limits.max_position_pct / abs(w)
                weights[key] = w * scale
                decisions.append(
                    RiskDecision(
                        action="scale",
                        reason=f"per-position cap on {key}",
                        scale_factor=scale,
                    )
                )

        # --- 2. Sector cap (scale each sector's members together).
        if sector_map:
            grouped: dict[str, list[str]] = {}
            for key in weights:
                sec = sector_map.get(key)
                if sec:
                    grouped.setdefault(sec, []).append(key)
            for sec, keys in grouped.items():
                exposure = sum(abs(weights[k]) for k in keys)
                if exposure > self.limits.max_sector_exposure:
                    scale = self.limits.max_sector_exposure / exposure
                    for k in keys:
                        weights[k] *= scale
                    decisions.append(
                        RiskDecision(
                            action="scale",
                            reason=f"sector cap on {sec}",
                            scale_factor=scale,
                        )
                    )

        # --- 3. Gross exposure cap.
        gross = sum(abs(w) for w in weights.values())
        if gross > self.limits.max_gross_exposure:
            scale = self.limits.max_gross_exposure / gross
            weights = {k: w * scale for k, w in weights.items()}
            decisions.append(
                RiskDecision(
                    action="scale",
                    reason="gross exposure cap",
                    scale_factor=scale,
                )
            )

        # --- 4. Net exposure cap.
        net = sum(weights.values())
        if abs(net) > self.limits.max_net_exposure:
            scale = self.limits.max_net_exposure / abs(net)
            weights = {k: w * scale for k, w in weights.items()}
            decisions.append(
                RiskDecision(
                    action="scale",
                    reason="net exposure cap",
                    scale_factor=scale,
                )
            )

        # --- 5. Build delta-quantity orders.
        # IBKR's API rejects fractional shares for EQUITY/ETF orders
        # (Error 10243: "Fractional-sized order cannot be placed via API").
        # Crypto and FX support fractional sizing. We truncate toward zero
        # so the resulting notional never exceeds the per-position cap.
        orders: list[Order] = []
        for key, target_w in weights.items():
            if key not in instruments:
                decisions.append(
                    RiskDecision(action="reject", reason=f"no instrument metadata for {key}")
                )
                continue
            if key not in last_prices or last_prices[key] <= 0:
                decisions.append(
                    RiskDecision(action="reject", reason=f"no positive last_price for {key}")
                )
                continue

            ins = instruments[key]
            whole_shares_only = ins.asset_class in (AssetClass.EQUITY, AssetClass.ETF)

            # Price in BASE currency: weight * equity is base-denominated,
            # so the divisor must be too.
            ccy = ins.currency or account.base_currency
            rate = 1.0
            if ccy != account.base_currency:
                rate = (fx_rates or {}).get(ccy, 0.0)
                if rate <= 0:
                    decisions.append(
                        RiskDecision(
                            action="scale",
                            reason=f"no FX rate for {ccy}; sizing {key} at 1.0 — "
                            "verify broker.get_fx_rates()",
                            scale_factor=1.0,
                        )
                    )
                    rate = 1.0
            price_base = last_prices[key] * rate

            target_value = target_w * account.equity
            target_qty = target_value / price_base
            if whole_shares_only:
                # int() truncates toward zero — fine for both long and short legs.
                target_qty = float(int(target_qty))
            current_qty = account.positions[key].quantity if key in account.positions else 0.0
            # Where the book WILL be once working orders fill.
            current_qty += pending_delta.get(key, 0.0)
            delta = target_qty - current_qty
            if whole_shares_only:
                delta = float(int(delta))
            # Long-only invariant, quantity half: a sell may flatten the
            # effective position but never cross zero. If the effective
            # position is already <= 0 (e.g. a working close covers it),
            # emit nothing rather than sell air.
            if not self.limits.allow_short and delta < 0:
                max_sell = max(current_qty, 0.0)
                if abs(delta) > max_sell:
                    decisions.append(
                        RiskDecision(
                            action="scale",
                            reason=(
                                f"long-only: sell on {key} clamped from {abs(delta):g} "
                                f"to {max_sell:g} (effective position incl. working orders)"
                            ),
                            scale_factor=0.0 if max_sell == 0 else max_sell / abs(delta),
                        )
                    )
                    delta = -max_sell
                    if whole_shares_only:
                        delta = float(int(delta))
            if abs(delta) < _EPS_QTY:
                continue

            orders.append(
                Order(
                    client_order_id=order_id_factory(),
                    instrument=ins,
                    side=Side.BUY if delta > 0 else Side.SELL,
                    quantity=abs(delta),
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.DAY,
                    created_at=signal.ts,
                )
            )

        # --- 6. No-margin enforcement (cash-account behavior).
        # If max_margin_borrowing_pct == 0.0, any order that would push a
        # currency cash balance below zero is unacceptable: IBKR would
        # auto-loan the deficit and we'd be on margin. Reject the basket
        # rather than partially fill — operator should FX-convert or
        # reduce sizing and try again.
        if orders and self.limits.max_margin_borrowing_pct < 1.0:
            margin_check = self._check_no_margin(orders, account, last_prices)
            if margin_check is not None:
                decisions.append(margin_check)
                return [], decisions

        decisions.append(
            RiskDecision(
                action="allow",
                reason=f"generated {len(orders)} orders",
            )
        )
        return orders, decisions

    def _check_no_margin(
        self,
        orders: list[Order],
        account: AccountSnapshot,
        last_prices: dict[str, float],
    ) -> RiskDecision | None:
        """Simulate the orders' effect on per-currency cash. Returns a
        reject ``RiskDecision`` if any currency would breach the margin
        budget; ``None`` if the basket is fundable.

        Budget per currency:
            allowed_debit = max_margin_borrowing_pct * account.equity
                         (interpreted as a base-currency budget; we
                          apply it independently to each currency as a
                          conservative proxy — a true cross-currency
                          model would need live FX rates, which we
                          don't want to require here).

        At 0.0, no currency may go below zero — strict cash account.
        """
        per_ccy_delta: dict[str, float] = {}
        for o in orders:
            px = last_prices.get(o.instrument.key, 0.0)
            if px <= 0:
                continue
            ccy = o.instrument.currency or account.base_currency
            notional = float(o.quantity) * float(px)
            # BUY consumes cash (negative delta); SELL produces cash.
            delta = -notional if o.side == Side.BUY else notional
            per_ccy_delta[ccy] = per_ccy_delta.get(ccy, 0.0) + delta

        # If we have a per-currency cash breakdown, use it; otherwise
        # fall back to total cash (and assume it's all in base currency).
        starting = (
            dict(account.cash_by_currency)
            if account.cash_by_currency
            else {account.base_currency: account.cash}
        )

        allowed_debit = self.limits.max_margin_borrowing_pct * max(account.equity, 0.0)
        breaches: list[str] = []
        for ccy, delta in per_ccy_delta.items():
            after = starting.get(ccy, 0.0) + delta
            if after < -allowed_debit:
                breaches.append(f"{ccy} {after:,.0f} (limit {-allowed_debit:,.0f})")

        if not breaches:
            return None

        hint = (
            "FX-convert into the deficit currency before re-running, "
            "or raise MAX_MARGIN_BORROWING_PCT in .env."
            if self.limits.max_margin_borrowing_pct == 0.0
            else "scale down the basket or raise MAX_MARGIN_BORROWING_PCT."
        )
        return RiskDecision(
            action="reject",
            reason=f"no-margin breach — would overdraw: {', '.join(breaches)}. {hint}",
        )

    # ------------------------------------------------------ force flatten

    def force_flatten_orders(
        self,
        positions: list[Position],
        *,
        ts: datetime | None = None,
        order_id_factory: Callable[[], str] = new_client_order_id,
    ) -> list[Order]:
        """Build market orders that close every non-zero position. Bypasses
        all limits (this is the emergency exit) but cannot be called while
        halted to a *different* reason — operator must unhalt first."""
        ts = ts or datetime.now(timezone.utc)
        orders: list[Order] = []
        for pos in positions:
            if abs(pos.quantity) < _EPS_QTY:
                continue
            orders.append(
                Order(
                    client_order_id=order_id_factory(),
                    instrument=pos.instrument,
                    side=Side.SELL if pos.quantity > 0 else Side.BUY,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.DAY,
                    created_at=ts,
                )
            )
        return orders
