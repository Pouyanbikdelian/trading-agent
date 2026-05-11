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

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from trading.core.logging import logger
from trading.core.types import (
    AccountSnapshot,
    Instrument,
    Order,
    OrderStatus,
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
        if self._halt_path and self._halt_path.exists():
            return HaltState.model_validate_json(self._halt_path.read_text())
        return HaltState()

    def _save_state(self) -> None:
        if self._halt_path is None:
            return
        self._halt_path.parent.mkdir(parents=True, exist_ok=True)
        self._halt_path.write_text(self._state.model_dump_json(indent=2))

    # ------------------------------------------------------ halt control

    @property
    def state(self) -> HaltState:
        return self._state

    def is_halted(self) -> bool:
        return self._state.halted

    def halt(self, reason: str) -> None:
        self._state = self._state.replace(
            halted=True, reason=reason, halted_at=datetime.now(timezone.utc),
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
        if self._state.halted:
            return RiskDecision(action="halt", reason=f"already halted: {self._state.reason}")

        # Lazily stamp the daily open if start_of_day() wasn't called — useful
        # for backtests and unit tests that don't model trading sessions.
        if self._state.last_day is None or self._state.daily_equity_open == 0:
            self.start_of_day(account)

        # Update high-water mark.
        if account.equity > self._state.equity_high_watermark:
            self._state = self._state.replace(equity_high_watermark=account.equity)
            self._save_state()

        # Daily-loss kill.
        if self._state.daily_equity_open > 0:
            day_pnl = (account.equity - self._state.daily_equity_open) / self._state.daily_equity_open
            if day_pnl <= -self.limits.max_daily_loss_pct:
                self.halt(
                    f"daily loss {day_pnl:.2%} breaches limit -{self.limits.max_daily_loss_pct:.2%}"
                )
                return RiskDecision(action="halt", reason=self._state.reason)

        # Peak drawdown halt.
        if self._state.equity_high_watermark > 0:
            dd = (account.equity - self._state.equity_high_watermark) / self._state.equity_high_watermark
            if dd <= -self.limits.max_drawdown_pct:
                self.halt(
                    f"drawdown {dd:.2%} breaches limit -{self.limits.max_drawdown_pct:.2%}"
                )
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
    ) -> tuple[list[Order], list[RiskDecision]]:
        """Convert a Signal's target weights into Orders, applying limits.

        Parameters are keyed by ``instrument.key`` (e.g. ``"equity:AAPL"``)
        to match ``Signal.target_weights`` and ``AccountSnapshot.positions``.
        """
        decisions: list[RiskDecision] = []
        if self._state.halted:
            return [], [RiskDecision(action="halt", reason=f"halted: {self._state.reason}")]
        if account.equity <= 0:
            return [], [RiskDecision(action="reject", reason="non-positive equity")]

        # Work on a mutable copy.
        weights: dict[str, float] = dict(signal.target_weights)

        # --- 1. Per-position cap (scale individual weights down if needed).
        for key in list(weights):
            w = weights[key]
            if abs(w) > self.limits.max_position_pct:
                scale = self.limits.max_position_pct / abs(w)
                weights[key] = w * scale
                decisions.append(RiskDecision(
                    action="scale",
                    reason=f"per-position cap on {key}",
                    scale_factor=scale,
                ))

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
                    decisions.append(RiskDecision(
                        action="scale",
                        reason=f"sector cap on {sec}",
                        scale_factor=scale,
                    ))

        # --- 3. Gross exposure cap.
        gross = sum(abs(w) for w in weights.values())
        if gross > self.limits.max_gross_exposure:
            scale = self.limits.max_gross_exposure / gross
            weights = {k: w * scale for k, w in weights.items()}
            decisions.append(RiskDecision(
                action="scale",
                reason="gross exposure cap",
                scale_factor=scale,
            ))

        # --- 4. Net exposure cap.
        net = sum(weights.values())
        if abs(net) > self.limits.max_net_exposure:
            scale = self.limits.max_net_exposure / abs(net)
            weights = {k: w * scale for k, w in weights.items()}
            decisions.append(RiskDecision(
                action="scale",
                reason="net exposure cap",
                scale_factor=scale,
            ))

        # --- 5. Build delta-quantity orders.
        orders: list[Order] = []
        for key, target_w in weights.items():
            if key not in instruments:
                decisions.append(RiskDecision(action="reject",
                                              reason=f"no instrument metadata for {key}"))
                continue
            if key not in last_prices or last_prices[key] <= 0:
                decisions.append(RiskDecision(action="reject",
                                              reason=f"no positive last_price for {key}"))
                continue

            target_value = target_w * account.equity
            target_qty = target_value / last_prices[key]
            current_qty = (
                account.positions[key].quantity if key in account.positions else 0.0
            )
            delta = target_qty - current_qty
            if abs(delta) < _EPS_QTY:
                continue

            orders.append(Order(
                client_order_id=order_id_factory(),
                instrument=instruments[key],
                side=Side.BUY if delta > 0 else Side.SELL,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
                tif=TimeInForce.DAY,
                created_at=signal.ts,
            ))

        decisions.append(RiskDecision(
            action="allow",
            reason=f"generated {len(orders)} orders",
        ))
        return orders, decisions

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
            orders.append(Order(
                client_order_id=order_id_factory(),
                instrument=pos.instrument,
                side=Side.SELL if pos.quantity > 0 else Side.BUY,
                quantity=abs(pos.quantity),
                order_type=OrderType.MARKET,
                tif=TimeInForce.DAY,
                created_at=ts,
            ))
        return orders
