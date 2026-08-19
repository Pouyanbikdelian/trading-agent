"""One iteration of the live loop: fetch → signal → risk → execute → reconcile.

``Cycle`` is broker-agnostic: it takes a ``Broker`` (Simulator or IbkrBroker)
and uses whatever ParquetCache + DataSources the caller wires in. The
runner (``trading.runner.runner.Runner``) builds those dependencies and
schedules ``run_cycle`` via APScheduler.

A cycle that hits an exception writes an ``error`` ``CycleReport``, fires a
critical Telegram alert, and returns — it does **not** raise. APScheduler
suppresses one-off job failures but the operator wouldn't see the bug;
the explicit failure-report + alert is what gets noticed.

Hard-coded design choices (and why)
-----------------------------------
* The cycle only generates **MARKET orders**. Limit/stop logic lives at
  the strategy level (a strategy that needs them emits a different target
  weight; the manager-driven Order construction is uniform here).
* Cycle uses the **simple equal-weight combiner** by default. Inverse-vol
  and min-variance combiners need per-strategy returns history; the
  runner doesn't keep those, so they're explicitly rejected at config
  load (see ``runner.py``).
* Reconciliation polls ``broker.get_fills(since=cycle_start)``. For
  IBKR-async this is eventually-consistent; in paper trades the simulator
  fills happen on the *next* ``step()`` call. The CLI's "paper" mode
  drives ``Simulator.step`` separately so this works.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from trading.core.exec_lock import ExecutionBusyError, execution_lock
from trading.core.logging import logger
from trading.core.types import (
    AccountSnapshot,
    Instrument,
    OrderStatus,
    RiskDecision,
    Signal,
)
from trading.core.universes import load_universe
from trading.data.base import DataSource, Frequency
from trading.data.cache import ParquetCache
from trading.execution.base import Broker
from trading.execution.store import OrderStore
from trading.risk.manager import RiskManager
from trading.runner.alerts import TelegramAlerts
from trading.runner.config import RunnerConfig
from trading.runner.heartbeat import write_heartbeat
from trading.runner.state import RunnerStore
from trading.selection.overlay import vol_target
from trading.strategies.base import get_strategy

CycleStatus = Literal[
    "ok",
    "no_orders",
    "halted",
    "halted_review",
    "error",
    "skipped_locked",
]

# Substrings that mean "we could not reach the broker" rather than "the
# broker said no". Matched on the exception text because ib-async raises
# several unrelated types for the same underlying condition — a plain
# ConnectionError, an asyncio TimeoutError, and its own errors all mean
# the same thing to an operator: the session is down, go tap your phone.
_DISCONNECT_MARKERS = (
    "not connected",
    "connection",
    "connect call failed",
    "timeout",
    "timed out",
    "broken pipe",
    "reset by peer",
    "no route to host",
    "econnrefused",
    "refused",
)


def _looks_like_disconnect(exc: BaseException) -> bool:
    """Distinguish a dead session from a refused order."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _DISCONNECT_MARKERS)


class CycleReport(BaseModel):
    """Outcome of a single cycle. Persisted by RunnerStore and returned to
    the caller / scheduler. Frozen so we can pass it around without aliasing."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    status: CycleStatus
    orders_submitted: int
    fills_received: int
    decisions: list[RiskDecision]
    error: str | None = None
    duration_ms: float = 0.0


def bought_symbols(target_weights: dict[str, float] | None) -> set[str]:
    """The names a signal actually wants held, as bare symbols.

    ``_weights_to_signal`` emits a key for EVERY instrument in the
    universe, the overwhelming majority of them at 0.0. Membership in
    ``target_weights`` therefore means "was considered", not "was
    bought" — a distinction with no consequence anywhere else in the
    cycle, because the risk manager sizes off the weight rather than the
    key set.

    It matters exactly once: labelling shadow rows. Read as membership,
    the first live run marked all 501 universe names ``taken``, leaving
    the counterfactual ledger with no ``passed`` rows and no ability to
    answer the only question it exists for. Hence a named function with
    a test rather than a set comprehension inline.
    """
    return {
        key.split(":")[-1].upper()
        for key, weight in (target_weights or {}).items()
        if abs(float(weight)) > 1e-9
    }


class Cycle:
    """All-in-one bound cycle. Construct once at startup; call ``run_cycle()``
    every bar."""

    #: A display-only state artifact. It is intentionally distinct from
    #: ``cycle_approval_pending.json`` so no bot command can ever mistake a
    #: halted review for executable approval state.
    HALTED_REVIEW_FILE: ClassVar[str] = "cycle_halted_review.json"

    #: A private ``Signal.metadata`` carrier for the PM decision whose
    #: lifecycle may advance after a clean broker acknowledgement.  It is
    #: deliberately data on the signal rather than mutable Cycle instance
    #: state: planning cycles can overlap before they contend on the shared
    #: execution lock, and one cycle must never persist another's PM view.
    _PM_TARGET_KEYS_METADATA: ClassVar[str] = "_pm_target_keys_to_persist"

    def __init__(
        self,
        config: RunnerConfig,
        *,
        cache: ParquetCache,
        source_factory: Callable[[Instrument], DataSource],
        broker: Broker,
        risk_manager: RiskManager,
        order_store: OrderStore,
        runner_store: RunnerStore,
        alerts: TelegramAlerts,
        heartbeat_path: Path | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        order_id_factory: Callable[[], str] | None = None,
        playbook: Any = None,
        regime_label_fn: Callable[[datetime], str] | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.source_factory = source_factory
        self.broker = broker
        self.risk_manager = risk_manager
        self.order_store = order_store
        self.runner_store = runner_store
        self.alerts = alerts
        self.heartbeat_path = heartbeat_path
        self._clock = clock
        self._order_id_factory = order_id_factory
        self._playbook = playbook
        self._regime_label_fn = regime_label_fn
        self._cycle_count = 0
        self._last_regime: str | None = None

    # ------------------------------------------------------- public API

    def run_cycle(self) -> CycleReport:
        """Execute one normal, potentially executable cycle.

        If the intraday gate is active or trips during planning, this remains
        a strictly non-executable review and returns ``halted_review``.  That
        lets an operator inspect the current risk-sized account proposal
        without making a halt an information blackout.
        """
        return self._run_public_cycle(force_review=False)

    def run_review(self) -> CycleReport:
        """Build a deliberately non-executable live-account review.

        This is used by `/cycle` while a halt is already active and by the
        explicit review command.  It never creates an approval window or
        enters a broker submission path, even if the halt is later cleared.
        """
        return self._run_public_cycle(force_review=True)

    def _run_public_cycle(self, *, force_review: bool) -> CycleReport:
        """Shared lifecycle persistence for executable and review cycles."""
        ts_start = self._clock()
        self._cycle_count += 1
        if not force_review:
            # A prior halted card is never an executable proposal. Remove it
            # before building a normal cycle so `/review` cannot keep showing
            # obsolete prices/account state after `/resume`.
            self._clear_halted_review()
        try:
            report = self._run_inner(ts_start, force_review=force_review)
        except Exception as e:
            logger.bind(component="cycle").exception("cycle failed")
            self.alerts.critical(f"cycle failed: {e!r}")
            report = CycleReport(
                ts=ts_start,
                status="error",
                orders_submitted=0,
                fills_received=0,
                decisions=[],
                error=str(e),
                duration_ms=self._elapsed_ms(ts_start),
            )
        # Always persist + heartbeat, even on error.
        try:
            self.runner_store.save_cycle(report)
        except Exception:
            logger.bind(component="cycle").exception("save_cycle failed")
        if self.heartbeat_path is not None:
            try:
                write_heartbeat(
                    self.heartbeat_path,
                    ts=self._clock(),
                    status=report.status,
                    cycle_no=self._cycle_count,
                    extra={
                        "orders_submitted": report.orders_submitted,
                        "fills_received": report.fills_received,
                    },
                )
            except Exception:
                logger.bind(component="cycle").exception("heartbeat write failed")
        return report

    # -------------------------------------------------------- internals

    def _effective_config(self, ts: datetime) -> tuple[RunnerConfig, bool]:
        """Apply the playbook (if configured) to derive this cycle's config.

        Returns ``(cfg, force_flatten)``. If no playbook is wired or the
        regime label doesn't resolve to a rule, the static config is returned
        unchanged. The runner logs every regime transition so the operator
        sees the system rotating between rules in the runner logs.
        """
        if self._playbook is None or self._regime_label_fn is None:
            return self.config, False

        try:
            label = self._regime_label_fn(ts)
        except Exception:
            logger.bind(component="cycle").exception(
                "regime classifier failed; falling back to static config"
            )
            return self.config, False

        # Local import to avoid coupling the cycle module to the playbook
        # types when no playbook is in use.
        from trading.runner.playbook import rule_for

        rule = rule_for(self._playbook, label)
        if rule is None:
            return self.config, False

        if label != self._last_regime:
            logger.bind(component="cycle").info(
                f"regime transition: {self._last_regime} -> {label}"
            )
            self.alerts.info(f"regime: {self._last_regime} -> {label}")
            self._last_regime = label

        # Merge: rule fields override; unset rule fields inherit. Frozen
        # pydantic + model_copy keeps both objects immutable.
        updates: dict[str, Any] = {}
        if rule.strategies:
            updates["strategies"] = list(rule.strategies)
        if rule.universe is not None:
            updates["universe"] = rule.universe
        if rule.vol_target is not None:
            updates["vol_target"] = rule.vol_target
        if rule.strategy_params:
            updates["strategy_params"] = dict(rule.strategy_params)
        return self.config.model_copy(update=updates), bool(rule.force_flatten)

    def _working_orders(self) -> list[Any]:
        """Orders live at the broker right now, or [] if it cannot say.

        Degrading to [] restores the old un-netted behaviour rather than
        blocking — this feeds the flatten path, and a panic button that
        refuses during a broker wobble is worse than one that may
        double-sell.
        """
        try:
            return list(self.broker.get_open_orders() or [])
        except Exception:
            logger.bind(component="cycle").warning(
                "could not read working orders; flatten will not be netted"
            )
            return []

    def _run_force_flatten(self, ts_start: datetime) -> CycleReport:
        """Playbook said this regime = stay flat. Generate closing orders for
        every open position, submit, and skip strategy generation entirely."""
        account = self._fetch_account(ts_start)
        # A playbook is automatic policy, not an explicit operator exit.
        # It must satisfy the same live session-baseline gate as a normal
        # cycle; otherwise a noon restart could bypass the protective review
        # mode merely by selecting a ``force_flatten`` regime.
        from trading.core.config import settings as _settings

        if _settings.is_live_armed():
            from trading.runtime.nyse_session import current_nyse_session_label

            session_risk = self.risk_manager.evaluate_session_risk(
                account,
                session_label=current_nyse_session_label(account.ts),
            )
            if session_risk.action != "allow":
                return self._run_force_flatten_review(
                    ts_start,
                    account=account,
                    reason=session_risk.reason,
                )
        # A playbook is automated policy, not an operator's explicit
        # reduce-only command.  It must obey the same hard halt as every
        # other automatic submission path.
        self.risk_manager._reload_halt_state()
        if self.risk_manager.is_halted():
            self.alerts.warning(
                "🛑 HALTED — playbook force-flatten was not submitted. "
                "Use the explicit reduce-only close/flatten command if you want to lower exposure."
            )
            return CycleReport(
                ts=ts_start,
                status="halted",
                orders_submitted=0,
                fills_received=0,
                decisions=[
                    RiskDecision(
                        action="halt",
                        reason=f"halted: {self.risk_manager.state.reason}",
                    )
                ],
                duration_ms=self._elapsed_ms(ts_start),
            )
        positions = list(account.positions.values())
        orders = self.risk_manager.force_flatten_orders(
            positions,
            ts=ts_start,
            working_orders=self._working_orders(),
            **({"order_id_factory": self._order_id_factory} if self._order_id_factory else {}),
        )

        orders_submitted = 0
        # Force-flatten is the operator's emergency exit, so it takes the
        # lock with a LONGER deadline than a routine cycle: if a cycle is
        # mid-submit we would rather wait for it than fail to flatten.
        # It still refuses to wait forever — a wedged holder must not
        # trap the one command that reduces risk.
        halted_during_submit = False
        with execution_lock(self._state_dir(), holder="force_flatten", timeout=120.0):
            # The initial check above is not sufficient: waiting for the
            # shared execution lock can take up to two minutes, during which
            # the operator may halt the system.  Never let the playbook's
            # automatic path cross that newly-set boundary.
            with self.risk_manager.submission_gate() as permitted:
                if permitted:
                    permitted = not self.risk_manager.is_halted()
            if not permitted:
                self.alerts.warning(
                    "🛑 HALTED — force-flatten stood down at the final submit gate."
                )
                return CycleReport(
                    ts=ts_start,
                    status="halted",
                    orders_submitted=0,
                    fills_received=0,
                    decisions=[
                        RiskDecision(
                            action="halt",
                            reason=f"halted: {self.risk_manager.state.reason}",
                        )
                    ],
                    duration_ms=self._elapsed_ms(ts_start),
                )
            for order in orders:
                with self.risk_manager.submission_gate() as permitted:
                    if not permitted:
                        halted_during_submit = True
                        break
                    try:
                        self.order_store.save_order(order)
                        self.broker.submit_order(order)
                        self.order_store.update_status(order.client_order_id, OrderStatus.SUBMITTED)
                        orders_submitted += 1
                    except Exception as e:
                        logger.bind(component="cycle").exception(
                            f"force-flatten submit failed for {order.client_order_id}"
                        )
                        self.order_store.update_status(order.client_order_id, OrderStatus.REJECTED)
                        self.alerts.error(f"force-flatten submit failed: {e!r}")

            fills = self.broker.get_fills(since=self._reconcile_from(ts_start))
            for fill in fills:
                try:
                    self.order_store.save_fill(fill, client_order_id=fill.order_id)
                    # Derive the status from cumulative fills, not from the
                    # arrival of one execution — see store.settle_status.
                    self.order_store.settle_status(fill.order_id)
                except Exception:
                    logger.bind(component="cycle").exception("save_fill failed")
        try:
            self.runner_store.save_snapshot(self.broker.get_account())
        except Exception:
            logger.bind(component="cycle").exception("snapshot persistence failed")

        return CycleReport(
            ts=ts_start,
            status="halted"
            if halted_during_submit
            else "ok"
            if orders_submitted > 0
            else "no_orders",
            orders_submitted=orders_submitted,
            fills_received=len(fills),
            decisions=[],
            duration_ms=self._elapsed_ms(ts_start),
        )

    def _run_force_flatten_review(
        self,
        ts_start: datetime,
        *,
        account: Any | None = None,
        reason: str | None = None,
    ) -> CycleReport:
        """Describe a playbook flatten without calling any broker mutation.

        This is intentionally compact because the normal live-account review
        has fresh price/risk sizing.  A force-flatten playbook is dormant in
        production, but it must never become a backdoor around review mode.
        """
        account = account or self._fetch_account(ts_start)
        orders = self.risk_manager.force_flatten_orders(
            list(account.positions.values()),
            ts=ts_start,
            working_orders=self._working_orders(),
            order_id_factory=lambda: f"review-flatten-{uuid.uuid4().hex[:12]}",
        )
        reason = reason or "playbook requested force flatten; review mode forbids submission"
        payload = {
            "schema_version": 1,
            "id": uuid.uuid4().hex[:12],
            "kind": "halted_review",
            "built_at": self._clock().isoformat(),
            "halted": self.risk_manager.is_halted(),
            "halt_reason": reason,
            "account_snapshot_ts": getattr(account, "ts", ts_start).isoformat(),
            "account": {
                "base_currency": getattr(account, "base_currency", "USD"),
                "equity": float(getattr(account, "equity", 0.0) or 0.0),
                "position_count": len(getattr(account, "positions", {}) or {}),
            },
            "plan": {
                "base_currency": getattr(account, "base_currency", "USD"),
                "order_count": len(orders),
                "prices_available": False,
                "orders": [
                    {
                        "symbol": order.instrument.symbol,
                        "side": order.side.value.upper(),
                        "quantity": float(order.quantity),
                    }
                    for order in orders
                ],
            },
            "will_never_auto_execute": True,
        }
        self._write_halted_review(payload)
        self.alerts.warning(
            "🛑 *HALTED — FORCE-FLATTEN REVIEW ONLY*\n"
            f"{len(orders)} position close(s) identified, but zero orders were submitted.\n"
            "_This review cannot be approved or queued. Use explicit reduce-only controls "
            "if you choose to lower exposure while halted._"
        )
        return CycleReport(
            ts=ts_start,
            status="halted_review",
            orders_submitted=0,
            fills_received=0,
            decisions=[RiskDecision(action="halt", reason=reason)],
            duration_ms=self._elapsed_ms(ts_start),
        )

    def _run_inner(self, ts_start: datetime, *, force_review: bool = False) -> CycleReport:
        # 0. Apply the regime playbook if one is configured. The playbook can
        # swap the universe, strategies, vol target, and per-strategy params.
        # `force_flatten: true` short-circuits to a flatten-everything cycle.
        cfg, force_flatten = self._effective_config(ts_start)
        if force_flatten and force_review:
            return self._run_force_flatten_review(ts_start)
        if force_flatten:
            return self._run_force_flatten(ts_start)

        # 1. Load instruments for the universe.
        instruments = load_universe(cfg.universe)

        # 1b. Pre-strategy screens (liquidity / quality / sector momentum).
        # Cuts the universe before strategies and the price load do the
        # heavy work, so the screen is essentially free.
        if cfg.screens is not None:
            instruments = self._apply_screens(instruments, cfg)

        # 1c. The Agent PM's own picks, when the bridge is live.
        #
        # The PM allocates across single names AND up to three ETFs, which
        # is how it expresses a hedge or a sector view. The cycle prices
        # only its configured universe, so under UNIVERSE=sp500 every ETF
        # target is unpriceable and gets dropped. Measured on the first
        # real decision: GLD 0.15 + XLV 0.12 out of 0.67 gross — FORTY
        # PERCENT of what the PM wanted to hold would have silently become
        # cash, and the digest would still have read like a fully invested
        # book.
        #
        # Added AFTER the screens deliberately: the screens exist to cut
        # illiquid or low-quality single names, and have nothing useful to
        # say about SPY. Only when the bridge is actually on, so the
        # mechanical strategy's universe is unchanged at sleeve 0.
        instruments = self._add_pm_targets(instruments)

        # 2. Build wide-format price frame.
        prices = self._load_prices(instruments, ts_start)
        if prices.empty or len(prices) < 2:
            self.alerts.warning(f"insufficient price history for {cfg.universe}")
            return CycleReport(
                ts=ts_start,
                status="no_orders",
                orders_submitted=0,
                fills_received=0,
                decisions=[],
                duration_ms=self._elapsed_ms(ts_start),
            )

        # 3. Get the latest broker account view.
        logger.bind(component="cycle").info("fetching broker account snapshot")
        account = self._fetch_account(ts_start)
        logger.bind(component="cycle").info(
            f"account: cash=${getattr(account, 'cash', 0):,.0f} "
            f"equity=${getattr(account, 'equity', 0):,.0f} "
            f"positions={len(getattr(account, 'positions', {}) or {})}"
        )

        # 4. Intraday kill switches (daily-loss, drawdown).  A request that
        # was explicitly marked review-only must not mutate the baseline or
        # execution state — it only observes the current halt.  A normal
        # cycle still evaluates risk exactly as before; when that trips, we
        # carry on as a non-executable review rather than hiding the current
        # proposal behind an early return.
        if force_review:
            self.risk_manager._reload_halt_state()
            if self.risk_manager.is_halted():
                intraday = RiskDecision(
                    action="halt", reason=f"already halted: {self.risk_manager.state.reason}"
                )
            else:
                intraday = RiskDecision(action="allow", reason="operator requested review-only")
            review_only = True
        else:
            # A live account may execute only against a baseline captured at
            # the actual NYSE open.  Research/paper keeps the legacy path so
            # its historical fixtures and simulations do not acquire a live
            # exchange-session dependency.  A missing trusted live baseline
            # becomes a review, never a noon re-baseline or order path.
            from trading.core.config import settings as _settings

            if _settings.is_live_armed():
                from trading.runtime.nyse_session import current_nyse_session_label

                intraday = self.risk_manager.evaluate_session_risk(
                    account,
                    session_label=current_nyse_session_label(account.ts),
                )
            else:
                intraday = self.risk_manager.evaluate_intraday(account)
            review_only = intraday.action in {"halt", "reject"}
        # A repaired baseline is not an error, but it IS the loudest thing
        # that happened this cycle: it means the figure the kill switches
        # were about to measure against belonged to another account. The
        # alternative — repairing it silently — is how the 2026-08-07
        # baseline poisoning would have gone unnoticed on a day when it
        # did not happen to trip a limit.
        if not force_review:
            try:
                note = self.risk_manager.take_baseline_note()
                if note:
                    self.alerts.critical(f"⚠️ Risk baseline repaired — {note}")
            except Exception:
                logger.bind(component="cycle").exception("baseline note alert failed")
        if intraday.action == "halt":
            if not force_review:
                self.alerts.critical(f"HALT: {intraday.reason}")
            logger.bind(component="cycle").warning(
                "risk halt active: building non-executable live-account review"
            )
        elif review_only:
            logger.bind(component="cycle").warning(
                f"execution safety gate rejected cycle: {intraday.reason}; "
                "building non-executable live-account review"
            )

        # 5. Generate strategy weights and combine.
        logger.bind(component="cycle").info(
            f"generating weights: strategies={cfg.strategies}, "
            f"combiner={cfg.combiner}, prices_shape={prices.shape}"
        )
        position_symbols = {
            p.instrument.symbol.upper() for p in (getattr(account, "positions", {}) or {}).values()
        }
        weights = self._generate_combined_weights(
            prices, cfg=cfg, position_symbols=position_symbols
        )

        # 6. Optional vol-target overlay.
        if cfg.vol_target is not None:
            weights = vol_target(
                weights,
                prices,
                target_vol=cfg.vol_target,
                lookback=cfg.vol_lookback,
                periods_per_year=cfg.periods_per_year,
                max_leverage=cfg.max_leverage,
            )

        # 6b. Operator mode overlay. Reads state/mode.json (default NEUTRAL,
        #     pass-through). Lets the operator de-risk via /mode defense
        #     etc. without touching the strategy code.
        weights = self._apply_operator_mode(weights, prices)

        # 7. Build the Signal from the last row of the weights frame.
        signal = self._weights_to_signal(weights, instruments, ts_start, cfg=cfg)
        last_prices = self._last_prices(prices, instruments)
        instruments_by_key = {ins.key: ins for ins in instruments if ins.key in last_prices}

        # 7b. Counterfactual ledger. The ranked ladder exists here whether
        # or not approval mode is on; before this it was computed only for
        # the approval prompt and discarded minutes later. Recording the
        # names below the cut is the only way to ever answer whether the
        # ranking step adds anything — see docs/LEARNING_ARCHITECTURE.md.
        if not review_only:
            self._record_shadow_ladder(prices, signal, last_prices, cfg=cfg)

        # 7c. Agent PM bridge. Deliberately AFTER the shadow ladder, so the
        # counterfactual keeps measuring the mechanical strategy's ranking
        # rather than the blend, and BEFORE the risk manager, so the PM
        # gets no privileged path to the broker. Disabled at sleeve 0.
        # FX rates so the risk manager sizes in base-currency terms (CHF
        # account buying USD stocks — the sizing was off by the USDCHF
        # factor before 2026-07-14). Failure degrades to {}: the manager
        # then sizes at 1.0 and records a decision, never crashes.
        #
        # Fetched BEFORE the PM merge because the PM's hard dollar cap
        # (PM_SLEEVE_CAPITAL_USD) is denominated in USD and the account is
        # not.
        fx_rates: dict[str, float] = {}
        try:
            fx_rates = self.broker.get_fx_rates()
        except Exception as e:
            logger.bind(component="cycle").warning(f"get_fx_rates failed ({e!r}); sizing at 1.0")

        signal = self._merge_pm_signal(
            signal,
            instruments_by_key=instruments_by_key,
            ts=ts_start,
            equity=float(getattr(account, "equity", 0.0) or 0.0),
            base_currency=getattr(account, "base_currency", None) or "USD",
            fx_rates=fx_rates,
        )
        # Sector tags for the risk manager's sector cap. cfg.sector_map wins;
        # otherwise derive key -> sector from the fundamentals cache so the cap
        # actually binds instead of silently no-op'ing on an empty map.
        sector_map = self._resolve_sector_map(instruments_by_key, cfg)

        # Orders already working at the broker — from ANY source (earlier
        # after-hours cycles, guard exits, manual commands). The risk
        # manager nets them into positions so this cycle can't stack on
        # top of them (the 2026-07-15 short-MU/SNDK incident). Failure
        # degrades to [] with a loud warning: stacking risk returns, so
        # the operator should avoid off-cycle /cycle until it recovers.
        pending_orders: list[Any] = []
        try:
            pending_orders = self.broker.get_open_orders()
            if pending_orders:
                pend = ", ".join(
                    f"{o.side.value} {o.quantity:g} {o.instrument.symbol}" for o in pending_orders
                )
                logger.bind(component="cycle").info(f"netting working orders: {pend}")
        except Exception as e:
            logger.bind(component="cycle").warning(
                f"get_open_orders failed ({e!r}); sizing WITHOUT pending-order netting"
            )

        # 8. Risk manager: signal -> orders.
        logger.bind(component="cycle").info(
            f"risk manager: signal has {len(signal.target_weights)} target weights"
            + (f", fx_rates={fx_rates}" if fx_rates else "")
        )
        review_id = uuid.uuid4().hex[:12] if review_only else None
        preview_sequence = 0

        def _preview_order_id() -> str:
            nonlocal preview_sequence
            preview_sequence += 1
            return f"review-{review_id}-{preview_sequence}"

        size_signal = (
            self.risk_manager.preview_signal_to_orders
            if review_only
            else self.risk_manager.signal_to_orders
        )
        orders, decisions = size_signal(
            signal,
            account=account,
            last_prices=last_prices,
            instruments=instruments_by_key,
            sector_map=sector_map,
            fx_rates=fx_rates,
            pending_orders=pending_orders,
            **(
                {"order_id_factory": _preview_order_id}
                if review_only
                else (
                    {"order_id_factory": self._order_id_factory} if self._order_id_factory else {}
                )
            ),
        )

        # 8a-bis. Operator holds (/hold SYM): pinned positions are frozen —
        # the cycle neither sells nor adds to them. We filter ORDERS rather
        # than weights because zeroing a held name's target weight would
        # make signal_to_orders emit a SELL for the existing position,
        # which is exactly what /hold exists to prevent.
        from trading.core.config import settings as _settings_holds
        from trading.runner.holds import filter_held_orders, load_holds

        held_syms = load_holds(_settings_holds.state_dir)
        if held_syms and orders:
            orders, dropped = filter_held_orders(orders, held_syms)
            if dropped:
                names = ", ".join(sorted({o.instrument.symbol for o in dropped}))
                logger.bind(component="cycle").info(
                    f"holds: dropped {len(dropped)} order(s) on pinned symbols: {names}"
                )
                self.alerts.info(
                    f"📌 Holds respected — skipped {len(dropped)} order(s) on "
                    f"pinned position(s): `{names}`. Use `/unhold <sym>` to release."
                )

        # 8b. Buying-power preflight. Estimate notional required vs cash
        # available; warn (don't refuse) if we're going to run short. The
        # broker will issue the actual rejection if margin doesn't permit
        # — this is a heads-up so the operator can intervene with /fx.
        self._preflight_buying_power(orders, account, last_prices)

        if review_only:
            # A halt has to stop the book growing. It does not have to stop
            # the desk from *shrinking* it — that was the complaint on
            # 2026-08-18, when a loss halt turned every command into an
            # information blackout and the only defensive action left was
            # trading by hand in IBKR.
            #
            # So before falling back to the read-only card, ask whether any
            # part of this plan provably reduces exposure. Sells against a
            # live long, buy-backs against a live short, sized to the
            # remaining headroom and never past flat. Buys are dropped, and
            # the operator still has to approve what is left.
            defensive = None
            defensive_reason: str | None = None
            if not force_review:
                # `/review` is an explicit contract: it never executes, even
                # if the risk state clears while it is being built. So the
                # reduce-only question is only asked on the `/cycle` path.
                defensive, defensive_reason = self._plan_defensive_reduction(
                    orders, account=account
                )
            if defensive is not None and defensive.has_orders:
                return self._run_defensive_cycle(
                    ts_start=ts_start,
                    intraday=intraday,
                    decisions=decisions,
                    orders=defensive.kept,
                    dropped=defensive.dropped,
                    clamped=defensive.clamped,
                    account=account,
                    prices=prices,
                    last_prices=last_prices,
                    fx_rates=fx_rates,
                    signal=signal,
                    held_symbols=held_syms,
                )
            return self._publish_halted_review(
                ts_start=ts_start,
                review_id=review_id or "review",
                intraday=intraday,
                decisions=decisions,
                orders=orders,
                account=account,
                last_prices=last_prices,
                fx_rates=fx_rates,
                signal=signal,
                held_symbols=held_syms,
                defensive_note=defensive_reason,
            )

        # 8c. Basket preview — Telegram the planned orders *before* they
        # go to the broker so the operator can see what will trade.
        # Three distinct outcomes need three distinct messages:
        #   1. strategy emitted no weights              → warm-up message
        #   2. strategy emitted weights but ALL rejected → show reject reasons
        #   3. strategy emitted weights, some accepted   → normal basket preview
        # The old code conflated #2 and "you're already there" so a basket
        # rejected by the no-margin check looked identical to a do-nothing
        # cycle. Surface the rejection reason verbatim so the operator
        # actually sees why nothing went through.
        strategy_has_view = bool(signal.target_weights)
        rejections = [d for d in decisions if d.action == "reject"]
        if not orders and not strategy_has_view:
            held = list(getattr(account, "positions", {}).keys())
            self.alerts.info(
                "📊 *Cycle plan: no orders*\n"
                "Strategy emitted no target weights this cycle — "
                "likely still in warm-up (insufficient price history) or "
                "today isn't a rebalance bar.\n\n"
                + (
                    f"_Current positions ({len(held)}) are untouched: "
                    f"`{', '.join(h.split(':')[-1] for h in held[:8])}`._\n"
                    "Use `/flatten` to clear them, `/signal` to see what "
                    "the strategy would pick once it has data."
                    if held
                    else "_Portfolio is flat; nothing to do._"
                )
            )
        elif not orders and rejections:
            # Strategy did decide, but the risk manager refused. Most common:
            # no-margin breach on a CHF-base account buying USD stocks
            # without pre-trade FX. Print every reject reason so the
            # operator knows what to do (typically /fx convert into the
            # deficit currency, then /cycle again).
            reasons = "\n".join(f"  • {r.reason}" for r in rejections[:5])
            n_picks = len(signal.target_weights)
            self.alerts.warning(
                f"⛔ *Cycle plan: refused by risk manager*\n"
                f"Strategy wanted {n_picks} names, but the basket was "
                f"rejected pre-submission:\n{reasons}\n\n"
                "_No orders sent. Address the cause above (e.g. `/fx 500000 "
                "CHF to USD` for margin breaches) and run `/cycle` again._"
            )
        # An approval-enabled cycle publishes one self-contained decision
        # card below. Sending the raw order table first made operators read
        # the same plan twice and mistook gross turnover for fresh exposure.
        from trading.core.config import settings as _settings

        if orders and not _settings.require_cycle_approval:
            self._announce_basket(orders, account, last_prices, fx_rates=fx_rates)

        # 8d. Per-cycle operator approval (optional, off by default).
        # When require_cycle_approval=true, block here until the operator
        # /approve's, /reject's, /pick's a different basket, or the
        # timeout fires. Default paper mode skips this entirely.
        if _settings.require_cycle_approval and orders:
            candidates = self._compute_top_candidates(prices, cfg=cfg)

            def _risk_rebuild(revised_signal: Signal) -> tuple[list[Any], list[RiskDecision]]:
                """Re-price an operator revision through the normal risk path.

                An approval edit is still a target portfolio, never a
                hand-filtered list of orders.  In particular, suppressing a
                planned sell can leave a position above the target or a risk
                limit, so every revised target must go back through the same
                sizing, exposure and no-margin checks as the original plan.
                """
                new_orders, new_decisions = self.risk_manager.signal_to_orders(
                    revised_signal,
                    account=account,
                    last_prices=last_prices,
                    instruments=instruments_by_key,
                    sector_map=sector_map,
                    fx_rates=fx_rates,
                    pending_orders=pending_orders,
                    **(
                        {"order_id_factory": self._order_id_factory}
                        if self._order_id_factory
                        else {}
                    ),
                )
                # Holds are an independent operator safety control.  A
                # custom approval must not become a side door around them.
                if held_syms and new_orders:
                    new_orders, dropped = filter_held_orders(new_orders, held_syms)
                    if dropped:
                        names = ", ".join(sorted({o.instrument.symbol for o in dropped}))
                        new_decisions.append(
                            RiskDecision(
                                action="scale",
                                reason=f"operator hold removed revised orders for: {names}",
                                scale_factor=1.0,
                            )
                        )
                return new_orders, new_decisions

            def _rebuild_from_picks(picked_symbols: list[str]) -> list[Any]:
                # Equal-weight the picked names at the per-position cap so
                # the basket still respects risk limits without rebuilding
                # the strategy from scratch. signal_to_orders applies the
                # gross/sector/margin checks on top.
                from trading.core.types import Signal as _Signal

                if not picked_symbols:
                    return []
                weight_each = self.risk_manager.limits.max_position_pct
                picked_keys: dict[str, float] = {}
                for sym in picked_symbols:
                    # match the strategy's symbol → instrument key namespace
                    matching = [k for k in instruments_by_key if k.endswith(f":{sym}") or k == sym]
                    if not matching:
                        continue
                    picked_keys[matching[0]] = weight_each
                if not picked_keys:
                    return []
                new_signal = _Signal(
                    ts=signal.ts,
                    strategy=signal.strategy,
                    target_weights=picked_keys,
                )
                new_orders, _new_decisions = _risk_rebuild(new_signal)
                return new_orders

            def _rebuild_with_frozen_symbols(
                mode: str, selected_symbols: list[str]
            ) -> tuple[list[Any], list[RiskDecision]]:
                """Preserve selected original targets and freeze the rest.

                ``only`` applies just the PM's planned changes for the named
                symbols.  ``except`` freezes the named planned changes.  A
                freeze means target the current position's actual weight,
                not simply delete a post-risk order: that lets the risk
                manager re-evaluate the complete revised book and still
                override a freeze when a hard limit requires it.
                """
                selected = {s.upper() for s in selected_symbols}
                planned_symbols = {o.instrument.symbol.upper() for o in orders}
                if mode not in {"only", "except"} or not selected:
                    raise ValueError("invalid custom approval request")
                if unknown := selected - planned_symbols:
                    raise ValueError(
                        "symbols are not in this cycle's planned changes: "
                        + ", ".join(sorted(unknown))
                    )

                frozen = (planned_symbols - selected) if mode == "only" else selected
                positions_by_symbol = {
                    pos.instrument.symbol.upper(): pos
                    for pos in (getattr(account, "positions", {}) or {}).values()
                }
                revised_weights = dict(signal.target_weights)
                equity = float(getattr(account, "equity", 0.0) or 0.0)
                if equity <= 0:
                    raise ValueError("cannot freeze positions with non-positive equity")

                for key, instrument in instruments_by_key.items():
                    symbol = instrument.symbol.upper()
                    if symbol not in frozen:
                        continue
                    position = positions_by_symbol.get(symbol)
                    if position is None:
                        revised_weights[key] = 0.0
                        continue
                    price = float(last_prices.get(key, 0.0) or 0.0)
                    if price <= 0:
                        raise ValueError(f"no positive price to freeze {symbol}")
                    currency = instrument.currency or getattr(account, "base_currency", "USD")
                    base_currency = getattr(account, "base_currency", "USD")
                    fx_rate = (
                        1.0 if currency == base_currency else float(fx_rates.get(currency, 0.0))
                    )
                    # Mirror RiskManager's documented missing-FX fallback.
                    # A frozen foreign holding must use the same conversion
                    # convention as every other target in the rebuilt plan.
                    if fx_rate <= 0:
                        fx_rate = 1.0
                    revised_weights[key] = float(position.quantity) * price * fx_rate / equity

                revised_signal = signal.model_copy(update={"target_weights": revised_weights})
                return _risk_rebuild(revised_signal)

            orders = self._request_cycle_approval(
                orders,
                account,
                last_prices,
                fx_rates=fx_rates,
                candidates=candidates,
                rebuild_from_picks=_rebuild_from_picks,
                rebuild_with_frozen_symbols=_rebuild_with_frozen_symbols,
            )
            if not orders:
                return CycleReport(
                    ts=ts_start,
                    status="no_orders",
                    orders_submitted=0,
                    fills_received=0,
                    decisions=decisions,
                    duration_ms=self._elapsed_ms(ts_start),
                )
        # 8f. LAST-INSTANT gates before any order leaves the building
        # (external review, 2026-07-15 — "the freeze is not atomic"):
        #
        # (a) Halt re-check. The step-4 check can be minutes old by now
        #     (price load + approval wait). If the operator /halt'ed or a
        #     kill switch fired in another process meanwhile, this cycle
        #     must submit NOTHING.
        # (b) Zombie guard. The runner's asyncio.wait_for cancels the
        #     awaiting coroutine on timeout but NOT this worker thread —
        #     a "timed out" cycle keeps running and could submit long
        #     after the operator was told it aborted. If we're past the
        #     runner's cycle timeout, stand down.
        self.risk_manager._reload_halt_state()
        if self.risk_manager.is_halted():
            self.alerts.critical(
                "⛔ submit gate: halt engaged after planning — dropping "
                f"{len(orders)} planned order(s), submitting nothing."
            )
            return CycleReport(
                ts=ts_start,
                status="halted",
                orders_submitted=0,
                fills_received=0,
                decisions=decisions,
                duration_ms=self._elapsed_ms(ts_start),
            )
        if self._elapsed_ms(ts_start) > 290_000:  # runner aborts awaiting at 300s
            self.alerts.critical(
                "⛔ submit gate: cycle exceeded its timeout before submission "
                f"— dropping {len(orders)} planned order(s) (zombie-cycle guard)."
            )
            return CycleReport(
                ts=ts_start,
                status="error",
                orders_submitted=0,
                fills_received=0,
                decisions=decisions,
                error="zombie-cycle guard: planning outlived the cycle timeout",
                duration_ms=self._elapsed_ms(ts_start),
            )

        # 9. Submit each order, under the single execution lock.
        #
        #    Four paths can reach the broker — this cycle, the on-demand
        #    cycle, the approval flow and manual commands — and until
        #    2026-08-06 they shared no mutex, only per-job locks that stop
        #    a job racing ITSELF. The system already shipped one incident
        #    from that gap (2026-07-14/15 order stacking took two names
        #    short). The lock is held around the broker mutation only:
        #    submit plus reconcile, seconds, not the whole cycle.
        try:
            with execution_lock(self._state_dir(), holder="cycle"):
                # The outer last-instant check can be stale while this cycle
                # waits for another execution path. Recheck under the lock;
                # the lower-level routine repeats an atomic halt check before
                # every individual broker submission.
                with self.risk_manager.submission_gate() as permitted:
                    if not permitted:
                        self.alerts.critical(
                            "⛔ submit gate: halt engaged while waiting for the execution lock "
                            f"— dropping {len(orders)} planned order(s), submitting nothing."
                        )
                        return CycleReport(
                            ts=ts_start,
                            status="halted",
                            orders_submitted=0,
                            fills_received=0,
                            decisions=decisions,
                            duration_ms=self._elapsed_ms(ts_start),
                        )
                (
                    orders_submitted,
                    fills,
                    halted_during_submit,
                    submission_failed,
                ) = self._submit_and_reconcile(
                    orders,
                    prices,
                    ts_start,
                )
        except ExecutionBusyError as e:
            # Skipping a cycle is recoverable; double-submitting is not.
            logger.bind(component="cycle").warning(f"cycle skipped: {e}")
            self.alerts.error(
                f"⚠️ cycle skipped — another path is trading: {e.holder}\n"
                "_No orders submitted. The next scheduled cycle will retry._"
            )
            return CycleReport(
                ts=ts_start,
                status="skipped_locked",
                orders_submitted=0,
                fills_received=0,
                decisions=decisions,
                duration_ms=self._elapsed_ms(ts_start),
            )

        if halted_during_submit:
            self.alerts.critical(
                "⛔ halt engaged during order batch — stopped remaining planned orders. "
                f"{orders_submitted} order(s) had already been submitted."
            )
            return CycleReport(
                ts=ts_start,
                status="halted",
                orders_submitted=orders_submitted,
                fills_received=len(fills),
                decisions=decisions,
                duration_ms=self._elapsed_ms(ts_start),
            )

        self._persist_pm_targets_after_successful_submission(
            signal,
            orders_submitted=orders_submitted,
            submission_failed=submission_failed,
        )

        # 10b. Telegram a fill summary so the operator doesn't have to
        # poll /positions. Aggregated to keep noise low even when 8
        # names fill at once. Silent if no fills this cycle.
        if fills:
            self._announce_fills(fills)

        return self._finish_cycle(ts_start, decisions, orders_submitted, fills)

    def _submit_and_reconcile(
        self,
        orders: list[Any],
        prices: pd.DataFrame,
        ts_start: datetime,
        *,
        allow_when_halted: bool = False,
    ) -> tuple[int, list[Any], bool, bool]:
        """Everything that mutates broker state. Runs under the execution
        lock — keep it short, and keep anything slow (price refresh, LLM
        calls, snapshots) outside it.

        ``allow_when_halted`` is used **only** by the defensive reduce-only
        cycle, whose orders have been independently proven to lower exposure
        against a fresh broker snapshot. Every other caller leaves it False,
        so a halt landing mid-batch still stops the remaining orders."""
        orders_submitted = 0
        halted_during_submit = False
        submission_failed = False
        for order in orders:
            # ``submission_gate`` shares the halt-state transaction with
            # Telegram/CLI /halt. A hard stop therefore takes effect between
            # orders in this batch, rather than only after all submissions.
            with self.risk_manager.submission_gate(
                allow_when_halted=allow_when_halted
            ) as permitted:
                if not permitted:
                    halted_during_submit = True
                    break
                try:
                    self.order_store.save_order(order)
                    self.broker.submit_order(order)
                    self.order_store.update_status(order.client_order_id, OrderStatus.SUBMITTED)
                    orders_submitted += 1
                except Exception as e:
                    submission_failed = True
                    # Surface broker rejections explicitly in Telegram so the
                    # operator sees the reason — not just a generic stack.
                    err_str = f"{type(e).__name__}: {e}"[:400]
                    logger.bind(component="cycle").exception(
                        f"submit failed for {order.client_order_id}"
                    )
                    self.order_store.update_status(order.client_order_id, OrderStatus.REJECTED)
                    # A CONNECTIVITY failure is a different event from a
                    # rejected order and needs a different reaction. "Order
                    # rejected: insufficient margin" is information; "cannot
                    # reach the broker" means the whole batch is dead and a
                    # human has to do something — usually approve a 2FA
                    # prompt. Escalate it so it does not scroll past looking
                    # like one more per-name rejection.
                    if _looks_like_disconnect(e):
                        self.alerts.critical(
                            "🔴 *Broker unreachable while submitting orders*\n"
                            f"`{err_str}`\n"
                            f"Failed on {order.instrument.symbol}; the rest of this "
                            "batch will fail too.\n"
                            "_Most likely an IBKR Mobile 2FA prompt after a gateway "
                            "restart. Approve it, then `/cycle` to retry._"
                        )
                    else:
                        self.alerts.error(
                            f"❌ order rejected: {order.side.value} {order.quantity:g} "
                            f"{order.instrument.symbol} — {err_str}"
                        )

        # 9b. Drive the broker's internal clock so paper-trade fills
        # materialize in the same cycle they were submitted in. No-op for
        # IbkrBroker; the Simulator uses this to fill at the next bar's open.
        latest_bars = self._build_latest_bars(prices, ts_start)
        try:
            self.broker.tick(ts_start, latest_bars)
        except Exception:
            logger.bind(component="cycle").exception("broker tick failed")

        # 10. Reconcile fills — from the oldest STILL-OPEN order, not from
        #     the start of this cycle.
        #
        #     `since=ts_start` only ever saw fills that arrived during the
        #     submitting cycle. An order placed after the close that filled
        #     at the next open was invisible: by the next cycle, `since`
        #     was already past it. Those rows stayed `submitted` forever
        #     and the local position view silently diverged from the
        #     broker's — which on live means the system can believe it
        #     holds nothing and buy the same name again.
        fills = self.broker.get_fills(since=self._reconcile_from(ts_start))
        for fill in fills:
            try:
                self.order_store.save_fill(fill, client_order_id=fill.order_id)
                # FILLED only when the cumulative fills cover the order.
                # Marking it terminal on the first execution stopped
                # oldest_open_created_at from holding the reconciliation
                # window open, so the rest of a partial fill could arrive
                # after the window closed and never be recorded.
                self.order_store.settle_status(fill.order_id)
            except Exception:
                logger.bind(component="cycle").exception("save_fill failed")
        self._warn_on_stale_open_orders(ts_start)
        return orders_submitted, fills, halted_during_submit, submission_failed

    def _finish_cycle(
        self,
        ts_start: datetime,
        decisions: list[Any],
        orders_submitted: int,
        fills: list[Any],
    ) -> CycleReport:
        """Snapshot, announce and report — deliberately OUTSIDE the
        execution lock, so a slow broker snapshot cannot block the
        operator's /flatten."""
        # 11. Persist the post-trade snapshot AND announce the new state.
        # Operator asked for an immediate portfolio print after every cycle
        # that did anything — so they don't have to /positions + /balances
        # to verify what changed.
        post_snap = None
        try:
            post_snap = self.broker.get_account()
            self.runner_store.save_snapshot(post_snap)
        except Exception:
            logger.bind(component="cycle").exception("snapshot persistence failed")

        if orders_submitted > 0 and post_snap is not None:
            self._announce_post_cycle_state(post_snap)

        status: CycleStatus = "ok" if orders_submitted > 0 else "no_orders"
        return CycleReport(
            ts=ts_start,
            status=status,
            orders_submitted=orders_submitted,
            fills_received=len(fills),
            decisions=decisions,
            duration_ms=self._elapsed_ms(ts_start),
        )

    def _publish_halted_review(
        self,
        *,
        ts_start: datetime,
        review_id: str,
        intraday: RiskDecision,
        decisions: list[RiskDecision],
        orders: list[Any],
        account: Any,
        last_prices: dict[str, float],
        fx_rates: dict[str, float],
        signal: Signal,
        held_symbols: set[str],
        defensive_note: str | None = None,
    ) -> CycleReport:
        """Persist and announce a real-account plan that cannot execute.

        The PM's research book is deliberately *not* reused as this screen.
        ``build_plan`` works from the current broker account, live FX and the
        same risk-sized order deltas an ordinary cycle would use.  The sole
        difference is that this branch has no approval, execution lock,
        OrderStore write, broker submit/tick/fill call, shadow write or PM
        target-state write.
        """
        from trading.runner.approval_view import build_plan, format_trade_review

        plan = build_plan(orders, account, last_prices, fx_rates=fx_rates)
        metadata = dict(signal.metadata or {})
        base_currency = str(getattr(account, "base_currency", "USD") or "USD").upper()
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        actual_halt = self.risk_manager.is_halted()
        adjustment_reasons = [
            decision.reason for decision in decisions if decision.action != "allow"
        ]
        payload: dict[str, Any] = {
            "schema_version": 1,
            "id": review_id,
            "kind": "halted_review",
            "built_at": self._clock().isoformat(),
            "halted": actual_halt,
            "halt_reason": intraday.reason,
            "account_snapshot_ts": getattr(account, "ts", ts_start).isoformat(),
            "account": {
                "base_currency": base_currency,
                "equity": round(equity, 2),
                "cash": round(float(getattr(account, "cash", 0.0) or 0.0), 2),
                "position_count": len(getattr(account, "positions", {}) or {}),
            },
            "plan": plan,
            "risk_adjustments": adjustment_reasons,
            "held_symbols": sorted(held_symbols),
            "combined_target_weights": {
                key: float(weight) for key, weight in signal.target_weights.items()
            },
            "pm": {
                "decided_at": metadata.get("pm_decided_at", ""),
                "sleeve_pct": metadata.get("pm_sleeve_pct", ""),
                "strategy_sleeve_pct": metadata.get("strategy_sleeve_pct", ""),
                "dropped": metadata.get("pm_dropped", ""),
            },
            "will_never_auto_execute": True,
        }
        try:
            self._write_halted_review(payload)
        except Exception:
            logger.bind(component="cycle").exception("could not persist halted review")

        title = (
            "🛑 *HALTED — LIVE-ACCOUNT REVIEW ONLY*"
            if actual_halt
            else "🧪 *LIVE-ACCOUNT REVIEW ONLY*"
        )
        headline = [
            title,
            f"Reason: `{intraday.reason}`",
            (
                f"Live account snapshot: {base_currency} {equity:,.0f} equity · "
                f"{len(getattr(account, 'positions', {}) or {})} current position(s)."
            ),
        ]
        if orders:
            rendered = format_trade_review(payload, heading=title)
        else:
            rejections = [reason for reason in adjustment_reasons if reason]
            rendered = "\n".join(
                [
                    "*No hypothetical position changes passed the current sizing gates.*",
                    *(
                        [f"Risk result: {rejections[0]}"]
                        if rejections
                        else ["The current account already matches the computed target."]
                    ),
                ]
            )
        notes: list[str] = []
        if held_symbols:
            notes.append("Holds preserved: `" + ", ".join(sorted(held_symbols)) + "`.")
        pm_decided_at = str(metadata.get("pm_decided_at") or "")
        if pm_decided_at:
            notes.append(
                "PM research was translated through the live account, sleeve, cash, FX, "
                "risk caps and holds (PM decision: " + pm_decided_at + ")."
            )
        if defensive_note:
            notes.append(
                "No part of this plan could be proven to reduce exposure — "
                f"{defensive_note}. Nothing here is approvable."
            )
        notes.append(
            "*0 orders submitted. This review cannot be approved, queued or executed. "
            "After `/resume`, run a brand-new `/cycle` for a fresh executable approval.*"
        )
        self.alerts.warning("\n\n".join(["\n".join(headline), rendered, "\n".join(notes)]))

        return CycleReport(
            ts=ts_start,
            status="halted_review",
            orders_submitted=0,
            fills_received=0,
            decisions=[intraday, *decisions],
            duration_ms=self._elapsed_ms(ts_start),
        )

    # -------------------------------------------- defensive (reduce-only)

    def _plan_defensive_reduction(
        self,
        orders: list[Any],
        *,
        account: Any,
    ) -> tuple[Any | None, str]:
        """Extract the provably-reducing subset of a blocked plan.

        Returns ``(filter_result, reason)``.  ``filter_result`` is ``None``
        when the defensive path is switched off or the account view cannot
        be trusted; ``reason`` always explains what the operator is seeing,
        because "nothing happened" is the message that gets ignored.
        """
        from trading.core.config import settings as _settings

        if not bool(getattr(_settings, "allow_defensive_cycle_when_halted", True)):
            return (
                None,
                "the defensive reduce-only path is disabled (ALLOW_DEFENSIVE_CYCLE_WHEN_HALTED=false)",
            )
        if not orders:
            return None, "the risk-sized plan contained no orders at all"

        positions = list((getattr(account, "positions", {}) or {}).values())
        if not positions:
            return None, "the account holds no positions to reduce"
        try:
            working = self._working_orders()
        except Exception:
            logger.bind(component="cycle").exception(
                "could not read working orders; refusing a defensive reduction"
            )
            return (
                None,
                "the broker's working orders could not be read, so no exit could be proven safe",
            )

        from dataclasses import replace

        from trading.execution.ibkr import new_client_order_id
        from trading.risk.reduce_only import filter_reduce_only

        try:
            result = filter_reduce_only(list(orders), positions, list(working or []))
        except Exception:
            logger.bind(component="cycle").exception("reduce-only filter failed")
            return None, "the reduce-only check itself failed"
        if not result.has_orders:
            return result, "every planned order would have opened or increased a position"

        # The orders arrived from ``preview_signal_to_orders`` and carry
        # display-only ``review-…`` client order ids. These are about to
        # reach the broker and the ledger, so re-stamp them with real ones;
        # a fill labelled "review" is a reconciliation problem later and a
        # lie in the audit trail now.
        factory = getattr(self, "_order_id_factory", None) or (lambda: new_client_order_id("dfns"))
        result = replace(
            result,
            kept=[o.model_copy(update={"client_order_id": factory()}) for o in result.kept],
        )
        return result, "reduce-only subset available"

    def _run_defensive_cycle(
        self,
        *,
        ts_start: datetime,
        intraday: RiskDecision,
        decisions: list[RiskDecision],
        orders: list[Any],
        dropped: list[tuple[str, str]],
        clamped: list[tuple[str, float, float]],
        account: Any,
        prices: pd.DataFrame,
        last_prices: dict[str, float],
        fx_rates: dict[str, float],
        signal: Signal,
        held_symbols: set[str],
    ) -> CycleReport:
        """Approve-and-submit a strictly exposure-reducing basket while halted.

        Three properties make this safe to run inside a kill switch:

        1. every order was verified against the live book to reduce, never
           open or flip, a position (``trading.risk.reduce_only``);
        2. approval is *mandatory* here regardless of
           ``REQUIRE_CYCLE_APPROVAL`` — a halted system never trades without
           a human saying so;
        3. the verification is repeated against a **fresh** broker snapshot
           under the execution lock, after the approval wait, because the
           position it was built from can be minutes old by then.

        Anything that fails re-verification is dropped, not resized upward.
        """
        base_currency = str(getattr(account, "base_currency", "USD") or "USD").upper()
        equity = float(getattr(account, "equity", 0.0) or 0.0)

        detail: list[str] = []
        if clamped:
            detail.append(
                "Trimmed to the remaining headroom: "
                + ", ".join(f"`{sym}` {want:g}→{got:g}" for sym, want, got in clamped)
                + "."
            )
        if dropped:
            shown = "; ".join(f"`{sym}` — {why}" for sym, why in dropped[:6])
            more = f" (+{len(dropped) - 6} more)" if len(dropped) > 6 else ""
            detail.append(f"Excluded as exposure-increasing: {shown}{more}.")
        if held_symbols:
            detail.append("Holds preserved: `" + ", ".join(sorted(held_symbols)) + "`.")

        self.alerts.warning(
            "\n".join(
                [
                    "🛡 *DEFENSIVE CYCLE — REDUCE-ONLY* (execution is halted)",
                    f"Reason: `{intraday.reason}`",
                    f"Live account: {base_currency} {equity:,.0f} equity · "
                    f"{len(getattr(account, 'positions', {}) or {})} position(s).",
                    "",
                    f"{len(orders)} order(s) that can only *lower* exposure are proposed below. "
                    "Buys, rotations and new names were removed. Approval is required — "
                    "nothing is sent unless you approve it.",
                ]
                + (["", *detail] if detail else [])
            )
        )

        approved = self._request_cycle_approval(
            orders,
            account,
            last_prices,
            fx_rates=fx_rates,
            defensive=True,
        )
        if not approved:
            return CycleReport(
                ts=ts_start,
                status="no_orders",
                orders_submitted=0,
                fills_received=0,
                decisions=[intraday, *decisions],
                duration_ms=self._elapsed_ms(ts_start),
            )

        if self._elapsed_ms(ts_start) > 290_000:
            self.alerts.critical(
                "⛔ defensive cycle exceeded its timeout before submission — "
                f"dropping {len(approved)} order(s) (zombie-cycle guard)."
            )
            return CycleReport(
                ts=ts_start,
                status="error",
                orders_submitted=0,
                fills_received=0,
                decisions=[intraday, *decisions],
                error="zombie-cycle guard: defensive planning outlived the cycle timeout",
                duration_ms=self._elapsed_ms(ts_start),
            )

        try:
            with execution_lock(self._state_dir(), holder="cycle:defensive"):
                final = self._reverify_reduce_only(approved)
                if not final:
                    self.alerts.warning(
                        "⛔ defensive cycle stood down — the live book changed while you were "
                        "reviewing, so no order could still be proven reduce-only. "
                        "Nothing was submitted."
                    )
                    return CycleReport(
                        ts=ts_start,
                        status="no_orders",
                        orders_submitted=0,
                        fills_received=0,
                        decisions=[intraday, *decisions],
                        duration_ms=self._elapsed_ms(ts_start),
                    )
                (
                    orders_submitted,
                    fills,
                    _halted_during_submit,
                    _submission_failed,
                ) = self._submit_and_reconcile(
                    final,
                    prices,
                    ts_start,
                    allow_when_halted=True,
                )
        except ExecutionBusyError as e:
            logger.bind(component="cycle").warning(f"defensive cycle skipped: {e}")
            self.alerts.warning(f"⏳ defensive cycle skipped — {e}")
            return CycleReport(
                ts=ts_start,
                status="skipped_locked",
                orders_submitted=0,
                fills_received=0,
                decisions=[intraday, *decisions],
                duration_ms=self._elapsed_ms(ts_start),
            )

        # A defensive cycle deliberately does NOT advance the PM's exit
        # history: it traded a filtered subset, not the PM's target book,
        # and next cycle must still see the PM's real previous targets.
        self.alerts.info(
            f"🛡 Defensive reduce-only cycle complete — {orders_submitted} order(s) submitted, "
            f"{len(fills)} fill(s). The account remains halted; no new exposure was added."
        )
        if fills:
            try:
                self._announce_fills(fills)
            except Exception:
                logger.bind(component="cycle").exception("fill summary failed")

        return CycleReport(
            ts=ts_start,
            status="ok" if orders_submitted > 0 else "no_orders",
            orders_submitted=orders_submitted,
            fills_received=len(fills),
            decisions=[intraday, *decisions],
            duration_ms=self._elapsed_ms(ts_start),
        )

    def _reverify_reduce_only(self, orders: list[Any]) -> list[Any]:
        """Re-prove every approved order against a fresh broker snapshot.

        The approval window can be minutes long, and a trailing-stop guard
        or a manual `/close` may have exited part of the book meanwhile.
        Submitting the stale quantity would then cross flat and open a
        short — precisely the outcome the halt exists to prevent.
        """
        from trading.risk.reduce_only import filter_reduce_only

        try:
            account = self._fetch_account(self._clock())
            positions = list((getattr(account, "positions", {}) or {}).values())
            working = list(self._working_orders() or [])
        except Exception:
            logger.bind(component="cycle").exception(
                "could not refresh the broker view before a defensive submit"
            )
            self.alerts.critical(
                "⛔ could not re-read the live book before submitting the defensive basket — "
                "nothing was sent."
            )
            return []
        result = filter_reduce_only(list(orders), positions, working)
        if result.dropped:
            names = ", ".join(f"`{sym}`" for sym, _ in result.dropped)
            self.alerts.warning(
                f"⚠️ dropped at the final reduce-only re-check (the book moved): {names}."
            )
        if result.clamped:
            trims = ", ".join(f"`{sym}` {want:g}→{got:g}" for sym, want, got in result.clamped)
            self.alerts.warning(f"⚠️ resized down at the final reduce-only re-check: {trims}.")
        return result.kept

    def _write_halted_review(self, payload: dict[str, Any]) -> None:
        """Atomically publish the display-only review artifact.

        The bot is a concurrent reader in another container.  ``os.replace``
        prevents it from ever seeing a half-written plan, while keeping the
        artifact intentionally separate from the executable approval files.
        """
        path = self._state_dir() / self.HALTED_REVIEW_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
            os.replace(tmp_name, path)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise

    def _clear_halted_review(self) -> None:
        """Discard a display-only review once a new normal cycle begins."""
        path = self._state_dir() / self.HALTED_REVIEW_FILE
        with suppress(FileNotFoundError):
            path.unlink()

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _state_dir() -> Path:
        """Where the shared execution lock lives. Read at call time, not
        at construction, so a test or a live/paper split that overrides
        STATE_DIR is honoured."""
        from trading.core.config import settings as _s

        return Path(_s.state_dir)

    def _elapsed_ms(self, ts_start: datetime) -> float:
        return (self._clock() - ts_start).total_seconds() * 1000.0

    #: How far back fill reconciliation may reach. An order open longer
    #: than this is a problem to alert on, not a window to keep widening —
    #: without a ceiling one wedged row would make every cycle ask the
    #: broker for a year of fills.
    RECONCILE_LOOKBACK = timedelta(days=14)

    #: An order still open past this is stale. Two sessions covers a
    #: normal after-hours order filling at the next open, plus a weekend.
    STALE_ORDER_AGE = timedelta(days=3)

    def _add_pm_targets(self, instruments: list[Any]) -> list[Any]:
        """Make the PM's actual picks priceable. No-op at sleeve 0, never
        raises.

        The PM chooses from a much wider set than the cycle prices: its
        whitelist is sp500 + nasdaq100 + russell1000 + us_large_cap (~1000
        names) plus a 21-ETF shelf, while the cycle loads only
        ``UNIVERSE`` — 503 names under sp500. Anything it picks outside
        that is unpriceable, gets dropped, and the weight silently becomes
        cash. Measured on the first real decision: GLD 0.15 + XLV 0.12 out
        of 0.67 gross — forty percent of the intended book, gone, with the
        digest still reading fully invested.

        The obvious fix, loading the PM's whole whitelist, is the wrong
        one: 503 symbols took 124s, so ~1000 would approach the cycle's
        300s timeout every week to price names it will not hold.

        So add only what the PM ACTUALLY PICKED — typically 7 or 8 symbols.
        Reading its decision here, before the price load, costs one small
        JSON read and makes the bridge's real constraint the whitelist
        (its anti-hallucination guard) rather than an unrelated universe
        setting.
        """
        try:
            from trading.core.config import settings as _s

            if float(getattr(_s, "agent_pm_sleeve_pct", 0.0) or 0.0) <= 0:
                return instruments

            from trading.agents.pm import UNIVERSE as PM_ETF_SHELF
            from trading.agents.pm_signal import load_previous_targets, pm_decision_path
            from trading.core.types import AssetClass, Instrument

            raw = json.loads(pm_decision_path(_s.state_dir).read_text())
            if not raw.get("ok"):
                return instruments
            targets = {str(s).upper().strip() for s in (raw.get("weights") or {})}
            # Also price what the PM held LAST cycle. A name it has
            # dropped needs a zero target to be sold, and a zero target on
            # an unpriced instrument is rejected ("no positive last_price")
            # — so without this the exit is computed and then discarded.
            targets |= {k.split(":", 1)[-1] for k in load_previous_targets(_s.state_dir)}
        except FileNotFoundError:
            return instruments
        except Exception:
            logger.bind(component="cycle").exception(
                "could not read PM targets; its off-universe picks will be dropped this cycle"
            )
            return instruments

        have = {getattr(i, "symbol", "").upper() for i in instruments}
        extra = [
            Instrument(
                symbol=s,
                asset_class=AssetClass.ETF if s in PM_ETF_SHELF else AssetClass.EQUITY,
            )
            for s in sorted(targets - have)
        ]
        if extra:
            logger.bind(component="cycle").info(
                f"PM bridge: pricing {len(extra)} PM target(s) outside the universe: "
                f"{[i.symbol for i in extra]}"
            )
        return [*instruments, *extra]

    def _pm_capital_scale(
        self,
        pm_weights: dict[str, float],
        *,
        equity: float,
        base_currency: str,
        fx_rates: dict[str, float],
    ) -> float:
        """Scale factor enforcing PM_SLEEVE_CAPITAL_USD — the HARD cap.

        This setting was added when the bridge was still hypothetical, with
        a comment saying it existed so that "the bridge, when built, sizes
        PM target weights against min(PM_SLEEVE_CAPITAL_USD, allotted
        equity)". The bridge was built on 2026-08-07 and nothing ever read
        it. GO_LIVE.md §4 records "$20,000 USD, hard cap" as a locked
        decision, and the July notes record it as shipped. It did not
        exist: the only thing bounding the PM was AGENT_PM_SLEEVE_PCT, a
        FRACTION — so the allocation grew with the account, which is
        exactly what a dollar cap is for.

        Returns 1.0 when the cap does not bind, or when we cannot value
        the book (no equity yet). Never raises.

        Currency: the cap is USD, the account may not be. ``fx_rates`` maps
        currency -> base units per 1 unit, so USD 20,000 on a CHF account
        at 0.81 is 16,200 CHF. With no rate available we apply the number
        as base currency and say so — on a CHF account that is slightly
        LOOSER than intended, which is why it is logged rather than
        silently assumed.
        """
        try:
            from trading.core.config import settings as _s

            cap_usd = float(getattr(_s, "pm_sleeve_capital_usd", 0.0) or 0.0)
            if cap_usd <= 0 or equity <= 0:
                return 1.0

            if base_currency.upper() == "USD":
                cap_base = cap_usd
            else:
                rate = float((fx_rates or {}).get("USD", 0.0) or 0.0)
                if rate > 0:
                    cap_base = cap_usd * rate
                else:
                    cap_base = cap_usd
                    logger.bind(component="cycle").warning(
                        f"no USD rate for the PM capital cap; treating "
                        f"{cap_usd:,.0f} USD as {base_currency} — verify get_fx_rates()"
                    )

            wanted_base = sum(pm_weights.values()) * equity
            if wanted_base <= cap_base:
                return 1.0
            return max(0.0, cap_base / wanted_base)
        except Exception:
            logger.bind(component="cycle").exception(
                "PM capital cap could not be computed; leaving the sleeve unscaled"
            )
            return 1.0

    def _merge_pm_signal(
        self,
        signal: Any,
        *,
        instruments_by_key: dict[str, Any],
        ts: datetime,
        equity: float = 0.0,
        base_currency: str = "USD",
        fx_rates: dict[str, float] | None = None,
    ) -> Any:
        """Blend the Agent PM's decision into the strategy signal.

        Two INDEPENDENT knobs, ``AGENT_PM_SLEEVE_PCT`` and
        ``STRATEGY_SLEEVE_PCT``, each the fraction of the account that book
        represents:

            pm 0.00, strat 1.00 -> mechanical only (the default)
            pm 0.11, strat 0.00 -> PM manages ~11% of the account, the
                                   strategy computes but never trades
            pm 0.50, strat 0.50 -> half each

        An earlier version derived the strategy's share as ``1 - pm``. That
        conflated two meanings and broke the case this exists for: the PM's
        weights are fractions of ITS OWN book, so its sleeve says "how much
        of the account that book represents". Setting it to 0.11 for a 10k
        sleeve on an 88k account then silently handed the strategy the
        other 89%. Independent knobs; the operator says what they mean.

        Weights are NOT renormalised. The PM holding 67% invested and 33%
        cash is a decision, and scaling its gross up to fill the sleeve
        would overrule it. Its 0.15 in GLD becomes 0.15 * sleeve of the
        account, and its cash stays cash.

        Overlapping names ADD rather than overwrite: if both books want
        AAPL, the account holds the sum of two independent convictions.

        Never raises and never silently no-ops. A cycle where the PM was
        expected to trade and did not is indistinguishable, in a position
        report, from a PM that chose to hold everything — so every refusal
        is announced.
        """
        from trading.agents.pm_signal import format_bridge_note, load_pm_signal
        from trading.core.config import settings as _s

        try:
            result = load_pm_signal(_s.state_dir, now=ts, tradeable_keys=set(instruments_by_key))
        except Exception:
            logger.bind(component="cycle").exception(
                "PM bridge raised; continuing on the mechanical strategy alone"
            )
            return signal

        strat = float(getattr(_s, "strategy_sleeve_pct", 1.0) or 0.0)

        if result.signal is None:
            # Silence only for the off switch. Every other refusal is an
            # event the operator needs, because it means the PM was meant
            # to reach the market this cycle and did not.
            if "disabled" not in result.reason:
                logger.bind(component="cycle").warning(f"PM bridge: {result.reason}")
                self.alerts.error(format_bridge_note(result))
            if strat >= 1.0:
                return signal
            # The strategy's own share still applies even when the PM does
            # not trade. Skipping this on the refusal path would hand the
            # whole account to a strategy the operator had deliberately
            # sized down — or, at strat 0, trade a book meant to be
            # benchmark-only.
            return signal.model_copy(
                update={"target_weights": {k: w * strat for k, w in signal.target_weights.items()}}
            )

        sleeve = result.sleeve_pct
        merged: dict[str, float] = {k: w * strat for k, w in signal.target_weights.items()}
        # The operator's mode applies to the PM sleeve too. Step 6b
        # reshapes the MECHANICAL strategy's weights and the PM is merged
        # in here, afterwards — so before this, `/mode flatten` zeroed the
        # strategy, left the PM fully invested, and still replied
        # "mode active: flatten".
        #
        # At STRATEGY_SLEEVE_PCT=0.0 — the live configuration — that was
        # not a partial gap. The strategy contributes nothing, the PM is
        # the only book that trades, and the one command that reads as
        # "get out of the market" did nothing whatsoever.
        #
        # Scaled HERE rather than on the combined book, because the
        # strategy side has already had its scale applied and doing it
        # again downstream would cut it twice.
        # PM_SLEEVE_CAPITAL_USD — the documented hard dollar cap, which
        # until now nothing read. Applied BEFORE the mode scale so the two
        # compose: the cap bounds what the PM may ever direct, the mode
        # cuts it further when the operator has de-risked.
        cap_scale = self._pm_capital_scale(
            result.signal.target_weights,
            equity=equity,
            base_currency=base_currency,
            fx_rates=fx_rates or {},
        )
        if cap_scale < 1.0:
            from trading.core.config import settings as _cap_s

            logger.bind(component="cycle").warning(
                f"PM capital cap binding: scaling the sleeve by {cap_scale:.3f} "
                f"(cap {float(_cap_s.pm_sleeve_capital_usd):,.0f} USD)"
            )
            self.alerts.info(
                f"💵 PM sleeve capped at {float(_cap_s.pm_sleeve_capital_usd):,.0f} USD "
                f"— targets scaled by {cap_scale:.2f}"
            )

        mode_scale, mode_name = self._pm_mode_scale()
        for k, w in result.signal.target_weights.items():
            w = w * cap_scale
            merged[k] = merged.get(k, 0.0) + (w if mode_scale is None else w * mode_scale)
        if mode_scale is not None:
            pm_gross = sum(result.signal.target_weights.values())
            logger.bind(component="cycle").warning(
                f"mode {mode_name}: PM sleeve scaled by {mode_scale:.2f} "
                f"({pm_gross:.1%} → {pm_gross * mode_scale:.1%} of account)"
            )
            self.alerts.info(
                f"🛡 mode *{mode_name}* also applied to the PM sleeve: "
                f"{pm_gross:.1%} → {pm_gross * mode_scale:.1%} of account"
            )

        # Names the PM held last cycle and no longer wants. Absent is not
        # the same as flat: an absent key gets no delta, so no order, so
        # the position stays. Only ever ADD a zero — if the mechanical
        # strategy still wants the name, its weight is the answer and the
        # PM exiting is not a reason to sell the strategy's position.
        exited: list[str] = []
        for k in sorted(result.exiting):
            if k in merged:
                continue
            if k not in instruments_by_key:
                # Unpriceable this cycle; the zero would be rejected
                # downstream anyway. Say so — a position nothing can reach
                # is exactly the thing that goes unnoticed for months.
                logger.bind(component="cycle").warning(
                    f"PM wants out of {k} but it is not priceable this cycle — "
                    "position left untouched"
                )
                continue
            merged[k] = 0.0
            exited.append(k)
        if exited:
            logger.bind(component="cycle").info(f"PM exit targets set to zero: {exited}")

        logger.bind(component="cycle").info(
            f"PM bridge: pm sleeve {sleeve:.1%}, strategy sleeve {strat:.1%}, "
            f"{len(result.signal.target_weights)} PM name(s), "
            f"combined gross {sum(merged.values()):.1%}"
        )
        self.alerts.info(format_bridge_note(result))

        return signal.model_copy(
            update={
                "target_weights": merged,
                "strategy": ("agent_pm" if strat <= 0 else f"{signal.strategy}+agent_pm"),
                "metadata": {
                    **(signal.metadata or {}),
                    "pm_sleeve_pct": f"{sleeve:.4f}",
                    "strategy_sleeve_pct": f"{strat:.4f}",
                    "pm_decided_at": result.signal.metadata.get("decided_at", ""),
                    "pm_dropped": ",".join(result.dropped),
                    # This is a candidate only. It is committed after a
                    # clean normal-cycle broker acknowledgement, never while
                    # merely planning a review or waiting for approval.
                    self._PM_TARGET_KEYS_METADATA: json.dumps(
                        sorted(result.signal.target_weights), separators=(",", ":")
                    ),
                },
            }
        )

    def _persist_pm_targets_after_successful_submission(
        self,
        signal: Signal,
        *,
        orders_submitted: int,
        submission_failed: bool,
    ) -> None:
        """Advance PM exit history only after a clean broker acknowledgement.

        ``last_targets.json`` drives later PM exits, so changing it while a
        proposal is only hypothetical can turn an untraded name into a
        phantom exit.  The candidate travels in this cycle's immutable
        signal; every early return (review, reject, timeout, halt or lock
        contention) bypasses this method.  A partially failed batch is also
        deliberately left unchanged: retrying with the known previous state
        is safer than claiming the proposed PM book reached the broker.
        """
        if orders_submitted <= 0 or submission_failed:
            return
        encoded_keys = signal.metadata.get(self._PM_TARGET_KEYS_METADATA)
        if not encoded_keys:
            return
        try:
            parsed_keys = json.loads(encoded_keys)
            if (
                not isinstance(parsed_keys, list)
                or not parsed_keys
                or any(not isinstance(key, str) or not key for key in parsed_keys)
            ):
                raise ValueError("invalid PM target-key metadata")
            keys = set(parsed_keys)
        except Exception:
            logger.bind(component="cycle").error(
                "refusing to persist malformed PM target-key metadata"
            )
            return
        try:
            from trading.agents.pm_signal import save_targets

            save_targets(self._state_dir(), keys)
        except Exception:
            logger.bind(component="cycle").exception(
                "could not record PM targets after broker acknowledgement; "
                "next cycle retains the prior PM exit state"
            )

    def _reconcile_from(self, ts_start: datetime) -> datetime:
        """Earliest timestamp worth asking the broker for fills from.

        The oldest still-open order, floored at RECONCILE_LOOKBACK and
        never later than this cycle's start. Self-healing: while an order
        is open we keep looking for its fill; once it goes terminal the
        window closes again on its own. No watermark file to corrupt, and
        the existing stuck rows get back-filled on the first run.
        """
        floor = ts_start - self.RECONCILE_LOOKBACK
        try:
            oldest = self.order_store.oldest_open_created_at()
        except Exception:
            logger.bind(component="cycle").exception("open-order lookup failed")
            return ts_start
        if oldest is None:
            return ts_start
        return max(min(oldest, ts_start), floor)

    def _warn_on_stale_open_orders(self, ts_start: datetime) -> None:
        """Alert on orders that have been open too long.

        The reconciliation fix stops NEW orders getting stuck, but an
        order the broker never acted on still needs a human. Silence when
        healthy; one message listing the offenders when not.
        """
        try:
            stale = self.order_store.open_orders_older_than(ts_start - self.STALE_ORDER_AGE)
        except Exception:
            logger.bind(component="cycle").exception("stale-order check failed")
            return
        if not stale:
            return
        lines = [
            f"  `{o.instrument.symbol}` {o.side.value} {o.quantity:g} "
            f"({status.value}, {(ts_start - o.created_at).days}d old)"
            for o, status, _ in stale[:8]
        ]
        logger.bind(component="cycle").warning(f"{len(stale)} order(s) open past the stale window")
        self.alerts.error(
            f"⚠️ *{len(stale)} order(s) still open* — never reached a terminal state:\n"
            + "\n".join(lines)
            + "\n_Local position view may differ from the broker. Check /orders._"
        )

    # Per-instrument refresh hard ceiling. yfinance has no built-in timeout;
    # before this we silently hung for >5min on the sp500 universe, blowing
    # the cycle's outer 300s budget. 15s per ticker x 500 = max 125min, but
    # in practice most calls return in ~500ms so totals are ~5min worst case.
    # The cycle's own 300s timeout still backstops the whole thing.
    REFRESH_PER_CALL_TIMEOUT_S: ClassVar[float] = 15.0
    REFRESH_PROGRESS_EVERY: ClassVar[int] = 50

    # How many concurrent yfinance fetches to run. yfinance's anonymous
    # rate limit kicks in around 2000 req/h; 8 workers x 503 names runs
    # in ~10-15s when the cache is cold and well under the limit.
    REFRESH_PARALLELISM: ClassVar[int] = 8

    def _load_prices(self, instruments: list[Instrument], ts: datetime) -> pd.DataFrame:
        """Read each instrument's price column from the cache, optionally
        fetching fresh bars first. Returns a wide-format DataFrame aligned
        on the inner intersection of dates.

        Two-phase to parallelize the slow part: submit ALL refresh futures
        upfront with a small thread pool, then iterate as they complete.
        Previously the loop was sequential (max_workers=1, future.result()
        inside the loop), so 503 yfinance calls took ~2 min. With 8
        workers it drops to ~10–15s when the cache is warm.
        """
        import concurrent.futures
        import time

        freq: Frequency = self.config.freq  # type: ignore[assignment]
        # Choose a generous start: we want at least `history_bars` rows. The
        # cache returns whatever it has; downstream code handles short series.
        start = ts.replace(year=ts.year - 5)  # 5 years back is more than enough
        end = ts

        n_total = len(instruments)
        logger.bind(component="cycle").info(
            f"loading prices: {n_total} instruments, auto_refresh="
            f"{self.config.auto_refresh}, freq={freq}, "
            f"workers={self.REFRESH_PARALLELISM}"
        )
        t0 = time.monotonic()
        n_refreshed = 0
        n_refresh_failed = 0
        n_refresh_timeout = 0

        series: dict[str, pd.Series] = {}

        def _fetch_one(ins: Instrument) -> tuple[Instrument, pd.DataFrame]:
            """Refresh (or read-only) a single instrument. Runs on a
            worker thread. Returns (instrument, dataframe)."""
            if self.config.auto_refresh:
                source = self.source_factory(ins)
                return ins, self.cache.get_bars(source, ins, start, end, freq)
            return ins, self.cache.read(ins, freq)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.REFRESH_PARALLELISM,
            thread_name_prefix="prices",
        ) as pool:
            futures = {pool.submit(_fetch_one, ins): ins for ins in instruments}
            done_count = 0
            for done_count, fut in enumerate(
                concurrent.futures.as_completed(
                    futures, timeout=self.REFRESH_PER_CALL_TIMEOUT_S * n_total
                ),
                start=1,
            ):
                ins = futures[fut]
                df = pd.DataFrame()
                try:
                    _ins, df = fut.result(timeout=self.REFRESH_PER_CALL_TIMEOUT_S)
                    if self.config.auto_refresh:
                        n_refreshed += 1
                except concurrent.futures.TimeoutError:
                    n_refresh_timeout += 1
                    logger.bind(symbol=ins.symbol).warning(
                        f"refresh timed out after {self.REFRESH_PER_CALL_TIMEOUT_S:.0f}s; "
                        "falling back to cache"
                    )
                except Exception:
                    n_refresh_failed += 1
                    logger.bind(symbol=ins.symbol).exception(
                        "refresh failed; falling back to cache"
                    )

                if df.empty:
                    df = self.cache.read(ins, freq)
                if df.empty or self.config.price_column not in df.columns:
                    pass
                else:
                    s = df[self.config.price_column].dropna()
                    if not s.empty:
                        series[ins.symbol] = s.iloc[-self.config.history_bars :]

                if done_count % self.REFRESH_PROGRESS_EVERY == 0:
                    elapsed = time.monotonic() - t0
                    logger.bind(component="cycle").info(
                        f"  …prices {done_count}/{n_total} loaded "
                        f"({len(series)} non-empty, {elapsed:.0f}s elapsed, "
                        f"refresh: {n_refreshed} ok / {n_refresh_timeout} timeout / "
                        f"{n_refresh_failed} err)"
                    )

        elapsed = time.monotonic() - t0
        logger.bind(component="cycle").info(
            f"prices loaded: {len(series)}/{n_total} non-empty in {elapsed:.1f}s "
            f"(refresh: {n_refreshed} ok, {n_refresh_timeout} timeout, "
            f"{n_refresh_failed} err)"
        )

        if not series:
            return pd.DataFrame()
        wide = pd.DataFrame(series).sort_index()
        # Drop short-history symbols BEFORE the row-wise dropna. The old
        # inner-join (`dropna(how="any")` alone) let a single recently
        # listed symbol truncate the entire price matrix to ITS length:
        # June 2026 incident — one fresh universe addition with ~26 bars
        # cut 503 symbols down to 26 rows, starved the 126-day momentum
        # lookback, and the paper strategy silently stopped trading for a
        # month ("no orders — portfolio already on target" every cycle).
        # A symbol too young to have real history can't carry a momentum
        # signal anyway — exclude it until it matures.
        counts = wide.notna().sum()
        required = max(int(counts.median() * 0.8), min(60, int(counts.max())))
        short = sorted(counts[counts < required].index)
        if short:
            logger.bind(component="cycle").warning(
                f"dropping {len(short)} short-history symbol(s) "
                f"(<{required} bars, matrix has {int(counts.max())}): "
                f"{', '.join(short[:12])}{'…' if len(short) > 12 else ''}"
            )
            wide = wide.drop(columns=short)
        wide = wide.dropna(how="any")
        return wide

    def _apply_screens(
        self,
        instruments: list[Instrument],
        cfg: RunnerConfig,
    ) -> list[Instrument]:
        """Filter the universe with the configured screens. Each screen reads
        only what it needs from the cache; missing data silently disables a
        screen rather than crashing the cycle. Logs the reduction so the
        operator can see how aggressive the filtering is in practice."""
        from trading.selection.screens import apply_screens

        original = len(instruments)
        sc = cfg.screens
        if sc is None:
            return instruments

        # Build closes + volumes from cached bars. Reads each Parquet once.
        closes: dict[str, pd.Series] = {}
        volumes: dict[str, pd.Series] = {}
        freq: Frequency = cfg.freq  # type: ignore[assignment]
        for ins in instruments:
            df = self.cache.read(ins, freq)
            if df.empty:
                continue
            if "close" in df.columns:
                closes[ins.symbol] = df["close"]
            if "volume" in df.columns:
                volumes[ins.symbol] = df["volume"]
        closes_df = pd.DataFrame(closes).sort_index() if closes else pd.DataFrame()
        volumes_df = pd.DataFrame(volumes).sort_index() if volumes else pd.DataFrame()

        # Optional fundamentals + sector prices.
        fundamentals = None
        if cfg.fundamentals_path:
            from trading.data.fundamentals_source import read_fundamentals_cache

            fundamentals = read_fundamentals_cache(Path(cfg.fundamentals_path))

        sector_prices = None
        if sc.top_n_sectors is not None:
            sector_prices = self._load_sector_prices(freq)

        filtered = apply_screens(
            instruments,
            sc,
            closes=closes_df if not closes_df.empty else None,
            volumes=volumes_df if not volumes_df.empty else None,
            fundamentals=fundamentals,
            sector_prices=sector_prices,
        )

        logger.bind(component="cycle").info(
            f"screens reduced universe: {original} -> {len(filtered)} instruments"
        )
        return filtered

    # SPDR sector ETFs keyed by yfinance's sector strings — used by the
    # sector-momentum screen when the user has no custom sector_etf_map.
    _DEFAULT_SECTOR_ETFS: ClassVar[dict[str, str]] = {
        "Technology": "XLK",
        "Financial Services": "XLF",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP",
        "Industrials": "XLI",
        "Utilities": "XLU",
        "Basic Materials": "XLB",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
    }

    def _resolve_sector_map(
        self, instruments_by_key: dict[str, Instrument], cfg: RunnerConfig
    ) -> dict[str, str] | None:
        """Sector tags for the risk manager's sector cap (``max_sector_exposure``).

        Precedence: an explicit ``cfg.sector_map`` always wins. Otherwise derive
        ``instrument.key -> sector`` from the fundamentals cache, so the cap binds
        automatically wherever fundamentals exist instead of no-op'ing on the
        empty default. Returns ``None`` when neither source yields tags — the cap
        then stays disabled, exactly as before.
        """
        if cfg.sector_map:
            return dict(cfg.sector_map)
        # Fall back to the conventional cache location when no explicit
        # path is configured. Before 2026-07-14 a missing fundamentals_path
        # returned None here and the 30% sector cap silently NEVER bound in
        # production — the book reached ~90% in one correlated sector with
        # the cap "configured" the whole time. Populate the cache with
        # `trading data fundamentals <universe>`.
        path = Path(cfg.fundamentals_path) if cfg.fundamentals_path else None
        if path is None:
            from trading.core.config import settings as _settings

            path = Path(_settings.data_dir) / "fundamentals.parquet"
        if not path.exists():
            self._warn_sector_cap_disabled(f"fundamentals cache missing at {path}")
            return None
        from trading.data.fundamentals_source import read_fundamentals_cache

        funds = read_fundamentals_cache(path)
        out: dict[str, str] = {}
        for key, ins in instruments_by_key.items():
            f = funds.get(ins.symbol)
            if f is not None and f.sector:
                out[key] = str(f.sector)
        if not out:
            self._warn_sector_cap_disabled(f"no sector tags for this universe in {path}")
            return None
        untagged = len(instruments_by_key) - len(out)
        if untagged:
            logger.bind(component="cycle").info(
                f"sector map covers {len(out)}/{len(instruments_by_key)} instruments "
                f"({untagged} untagged — those escape the sector cap)"
            )
        return out

    def _warn_sector_cap_disabled(self, why: str) -> None:
        """The sector cap failing OPEN must be loud, not silent — a limit
        the operator believes in but that never fires is worse than no
        limit. Alert once per process, log every cycle."""
        logger.bind(component="cycle").warning(f"SECTOR CAP DISABLED — {why}")
        if not getattr(self, "_sector_cap_warned", False):
            self._sector_cap_warned = True
            self.alerts.warning(
                f"⚠️ sector cap ({self.risk_manager.limits.max_sector_exposure:.0%}) is "
                f"NOT enforced — {why}. Run `trading data fundamentals <universe>` "
                "to enable it."
            )

    def _load_sector_prices(self, freq: Frequency) -> pd.DataFrame:
        """Read sector-ETF closes from the cache and re-key by sector name
        (not ETF symbol) so the screen can match against ``Fundamentals.sector``."""
        from trading.core.types import AssetClass, Instrument

        series: dict[str, pd.Series] = {}
        for sector_name, etf in self._DEFAULT_SECTOR_ETFS.items():
            ins = Instrument(symbol=etf, asset_class=AssetClass.ETF)
            df = self.cache.read(ins, freq)
            if df.empty or "close" not in df.columns:
                continue
            series[sector_name] = df["close"]
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).sort_index().dropna(how="all")

    def _build_latest_bars(self, prices: pd.DataFrame, ts: datetime) -> dict[str, Any]:
        """Build the {symbol: Bar} snapshot the Simulator needs to fill orders.

        We only have the close column in ``prices`` (the cycle reads one
        column to keep memory small); the cache has the rest. Fall back to
        the close for open/high/low when the cache is unreadable — fills
        will execute at close instead of next bar's open, which is a slight
        bias but never crashes."""
        from trading.core.types import AssetClass, Bar, Instrument

        freq: Frequency = self.config.freq  # type: ignore[assignment]
        bars: dict[str, Any] = {}
        for sym in prices.columns:
            close = float(prices[sym].iloc[-1])
            ins = Instrument(symbol=sym, asset_class=AssetClass.EQUITY)
            try:
                df = self.cache.read(ins, freq)
                row = df.iloc[-1]
                bars[sym] = Bar(
                    ts=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0) or 0.0),
                )
            except Exception:
                bars[sym] = Bar(ts=ts, open=close, high=close, low=close, close=close, volume=0.0)
        return bars

    def _fetch_account(self, ts: datetime) -> AccountSnapshot:
        """Get the broker's account view.

        CRITICAL: if this fails we MUST raise. The previous behavior of
        falling back to a synthetic zero-position snapshot caused real
        money damage in paper: when ``broker.get_account`` timed out the
        cycle thought it owned nothing, bought the full target basket
        again, and stacked positions to 3× target across three cycles.
        Re-raising means the outer cycle handler reports the error and
        skips the rebalance — far safer than trading on stale state.
        """
        return self.broker.get_account()

    def _preflight_buying_power(
        self,
        orders: list[Any],
        account: Any,
        last_prices: dict[str, float],
    ) -> None:
        """Estimate notional cost of BUY orders vs current cash.

        Surfaces a Telegram warning when cash is short. We don't *reject*
        — IBKR's risk margin may permit; the operator can also `/fx
        CHF X` to convert before the actual fill. Sells reduce required
        notional. Skip if last_prices is empty.
        """
        if not orders or not last_prices:
            return
        from trading.core.types import Side as _Side  # local to avoid name clash

        notional_required = 0.0
        for o in orders:
            px = last_prices.get(o.instrument.key)
            if px is None:
                continue
            sign = 1.0 if o.side == _Side.BUY else -1.0
            notional_required += sign * o.quantity * px
        if notional_required <= 0:
            return
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        shortfall = notional_required - cash
        if shortfall <= 0:
            return
        self.alerts.warning(
            f"⚠️ buying-power preflight: need ~${notional_required:,.0f} for "
            f"net buys, have ${cash:,.0f} — short ~${shortfall:,.0f}. "
            "IBKR may reject or auto-margin; consider `/fx` to convert."
        )

    def _announce_fills(self, fills: list[Any]) -> None:
        """Telegram a one-shot summary of fills received this cycle.

        One alert per cycle (never per fill), grouped by order, with
        the instrument's actual currency on each line. FX trades get
        formatted differently from equity trades — "rate" instead of
        "price per share", and side/qty in the instrument's terms.
        """
        if not fills:
            return

        # Look up the orders we submitted recently so we can attach
        # instrument + side context to each fill. The instrument on the
        # Fill object can be None when the broker reports executions
        # without populating it (FX in particular).
        from datetime import timedelta

        orders_by_id: dict[str, Any] = {}
        try:
            recent = self.order_store.load_orders(
                since=datetime.now(tz=timezone.utc) - timedelta(days=1)
            )
            for order, _st, _bid in recent:
                orders_by_id[order.client_order_id] = order
        except Exception:
            logger.bind(component="cycle").debug(
                "order lookup for fills announcement failed; falling back to symbol-only"
            )

        # Group by order_id so partial fills aggregate cleanly.
        by_order: dict[str, list[Any]] = {}
        for f in fills:
            oid = getattr(f, "order_id", None) or "?"
            by_order.setdefault(oid, []).append(f)

        parts: list[str] = ["💰 *Fills this cycle*"]
        total_by_ccy: dict[str, float] = {}
        fee_by_ccy: dict[str, float] = {}

        for oid, entries in by_order.items():
            order = orders_by_id.get(oid)
            instr = getattr(order, "instrument", None) if order else None
            sym = (
                (getattr(instr, "symbol", None) if instr else None)
                or self._fill_symbol(entries[0])
                or "?"
            )
            side = getattr(order, "side", None)
            side_str = getattr(side, "value", None) or "TRD"
            ccy = getattr(instr, "currency", None) or "USD"
            ac = getattr(instr, "asset_class", None)
            asset_str = str(getattr(ac, "value", ac or "")).upper() or "?"
            is_fx = asset_str == "FX"

            qty = sum(float(getattr(e, "quantity", 0.0) or 0.0) for e in entries)
            if qty == 0:
                continue
            notional = sum(
                float(getattr(e, "quantity", 0.0) or 0.0) * float(getattr(e, "price", 0.0) or 0.0)
                for e in entries
            )
            commission = sum(float(getattr(e, "commission", 0.0) or 0.0) for e in entries)
            avg_px = notional / qty if qty > 0 else 0.0
            total_by_ccy[ccy] = total_by_ccy.get(ccy, 0.0) + notional
            fee_by_ccy[ccy] = fee_by_ccy.get(ccy, 0.0) + commission

            parts.append("")  # blank line between entries
            if is_fx:
                parts.append(f"💱 *{sym}* (FX)")
                parts.append(f"   Side:    `{side_str}`")
                parts.append(f"   Qty:     `{qty:,.0f}`")
                parts.append(f"   Rate:    `{avg_px:.5f}`")
                parts.append(f"   Total:   `{ccy} {notional:,.2f}`")
                parts.append(f"   Fee:     `{ccy} {commission:,.2f}`")
            else:
                parts.append(f"📈 *{sym}*  ({asset_str})")
                parts.append(f"   {side_str}:    `{qty:,.0f} @ {ccy} {avg_px:,.2f}`")
                parts.append(f"   Total:   `{ccy} {notional:,.2f}`")
                parts.append(f"   Fee:     `{ccy} {commission:,.4f}`")

        parts.append("")  # separator before summary
        # Summary aggregated per currency so mixed-currency baskets
        # (e.g. USD stocks + a CHF FX leg) are still honest.
        summary_lines = [f"📊 *Summary* — {len(fills)} execution(s)"]
        for ccy in sorted(total_by_ccy):
            summary_lines.append(
                f"   notional: `{ccy} {total_by_ccy[ccy]:,.2f}`  "
                f"fees: `{ccy} {fee_by_ccy.get(ccy, 0):,.2f}`"
            )
        parts.extend(summary_lines)

        self.alerts.info("\n".join(parts))

    def _announce_post_cycle_state(self, snap: Any) -> None:
        """Telegram a full portfolio snapshot after a cycle that did work.

        Operator-requested: when a cycle submits orders, they want an
        immediate confirmation of the new state — total equity, cash by
        currency, and every position with qty + avg cost + % weight —
        without having to ask via /balances and /positions.

        Only fires when orders were submitted; cycles with no orders are
        silent here (the basket-preview already said "no orders").
        """
        equity = float(getattr(snap, "equity", 0.0) or 0.0)
        total_cash = float(getattr(snap, "cash", 0.0) or 0.0)
        positions = getattr(snap, "positions", {}) or {}

        # Per-currency cash breakdown — only available on brokers that
        # implement get_balances (IbkrBroker does; Simulator doesn't).
        per_ccy: dict[str, float] = {}
        if hasattr(self.broker, "get_balances"):
            try:
                per_ccy = self.broker.get_balances() or {}  # type: ignore[attr-defined]
            except Exception:
                logger.bind(component="cycle").exception(
                    "get_balances failed; skipping FX breakdown"
                )

        ccy = getattr(snap, "base_currency", None) or "USD"
        lines: list[str] = ["📈 *Portfolio after this cycle*"]
        lines.append(f"  Equity: {ccy} {equity:,.2f}    Cash: {ccy} {total_cash:,.2f}")

        if per_ccy:
            ccy_parts = [
                f"{ccy} {amt:,.0f}" for ccy, amt in sorted(per_ccy.items()) if abs(amt) >= 1.0
            ]
            if ccy_parts:
                lines.append("  Cash by currency: " + " | ".join(ccy_parts))

        if not positions:
            lines.append("  No open positions.")
            self.alerts.info("\n".join(lines))
            return

        # Holdings table — same shape as /positions but live, not snapshot-read.
        lines.append(f"  Positions: {len(positions)}")
        header = f"  {'Symbol':<7} {'Qty':>9} {'Avg cost':>10} {'Mkt value':>11} {'Weight':>7}"
        sep = "  " + "-" * (len(header) - 2)
        rows: list[str] = []
        for _key, pos in sorted(positions.items()):
            qty = float(pos.quantity)
            avg = float(pos.avg_price)
            mv = qty * avg + float(getattr(pos, "unrealized_pnl", 0.0) or 0.0)
            w = (mv / equity) if equity > 0 else 0.0
            rows.append(
                f"  {pos.instrument.symbol:<7} {qty:>9.2f} {avg:>10,.2f} {mv:>11,.0f} {w:>6.1%}"
            )
        table = "```\n" + "\n".join([header, sep, *rows]) + "\n```"
        lines.append(table)
        self.alerts.info("\n".join(lines))

    @staticmethod
    def _fill_symbol(fill: Any) -> str | None:
        """Best-effort symbol extraction from a Fill object.

        Different broker adapters carry the symbol in different places
        (some have ``instrument.symbol``, some only the ``order_id`` slug).
        Returns None when we genuinely can't tell — caller falls back to '?'.
        """
        instr = getattr(fill, "instrument", None)
        if instr is not None:
            sym = getattr(instr, "symbol", None)
            if sym:
                return str(sym)
        # Fall back: orderRef sometimes embeds the symbol after a dash.
        oid = getattr(fill, "order_id", None) or ""
        if "-" in oid:
            return oid.split("-", 1)[1][:6] or None
        return None

    # -------------------------------------------------- cycle approval gate

    # File paths used to coordinate with the bot process. The cycle (in
    # the trader process) writes APPROVAL_PENDING_FILE and polls for
    # APPROVAL_DECISION_FILE; the bot (a separate process) reads pending
    # and writes the decision when the operator sends /approve, /reject.
    APPROVAL_PENDING_FILE: ClassVar[str] = "cycle_approval_pending.json"
    APPROVAL_DECISION_FILE: ClassVar[str] = "cycle_approval_decision.json"
    APPROVAL_AUDIT_FILE: ClassVar[str] = "cycle_approval_audit.jsonl"
    APPROVAL_LOCK_FILE: ClassVar[str] = "cycle_approval.lock"
    APPROVAL_POLL_INTERVAL_S: ClassVar[float] = 2.0

    def _request_cycle_approval(
        self,
        orders: list[Any],
        account: Any,
        last_prices: dict[str, float],
        *,
        fx_rates: dict[str, float] | None = None,
        candidates: list[tuple[str, float]] | None = None,
        rebuild_from_picks: Callable[[list[str]], list[Any]] | None = None,
        rebuild_with_frozen_symbols: (
            Callable[[str, list[str]], tuple[list[Any], list[RiskDecision]]] | None
        ) = None,
        defensive: bool = False,
    ) -> list[Any]:
        """Block the cycle until the operator decides via Telegram.

        ``defensive`` marks a reduce-only basket built while execution is
        halted. It changes two things: the published window carries
        ``defensive_reduce_only``, which is the only flag that lets the bot
        accept an ``/approve`` during a halt, and the prompt says plainly
        that nothing here can add exposure.

        Writes the basket summary (+ optional top-N candidate scoreboard)
        to ``state/cycle_approval_pending.json`` and waits for
        ``state/cycle_approval_decision.json``. Bot produces the decision
        file in response to /approve, /approve N, /approve flat,
        /pick … or /reject.

        ``candidates`` is the strategy's top-N ranked list (best-first,
        with scores). When present, /pick can override which subset to
        trade — ``rebuild_from_picks`` then rebuilds the orders from
        those picks through the risk manager.

        ``rebuild_with_frozen_symbols`` handles ``/approve only …`` and
        ``/approve all except …``. The first response requests a revised
        plan; this method risk-checks it, shows the exact result, and
        requires a second, preview-bound confirmation.

        Returns the list of orders to submit. Empty = no submission.
        """
        import fcntl as _fcntl
        import hashlib as _hashlib
        import json as _json
        import time as _time
        import uuid as _uuid

        from trading.core.config import settings as _settings

        state_dir = _settings.state_dir
        pending_path = state_dir / self.APPROVAL_PENDING_FILE
        decision_path = state_dir / self.APPROVAL_DECISION_FILE
        audit_path = state_dir / self.APPROVAL_AUDIT_FILE
        approval_lock_path = state_dir / self.APPROVAL_LOCK_FILE

        def _order_payload(plan: list[Any]) -> list[dict[str, Any]]:
            """Stable, human-readable order description for a preview."""
            return [
                {
                    "symbol": str(order.instrument.symbol).upper(),
                    "side": str(getattr(order.side, "value", order.side)).upper(),
                    "quantity": float(order.quantity),
                }
                for order in plan
            ]

        def _fingerprint(plan: list[Any]) -> str:
            canonical = _json.dumps(
                _order_payload(plan), sort_keys=True, separators=(",", ":")
            ).encode()
            return _hashlib.sha256(canonical).hexdigest()

        def _write_pending(payload: dict[str, Any]) -> None:
            """Publish state atomically so the bot never reads half a plan."""
            pending_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = pending_path.with_name(f".{pending_path.name}.tmp")
            tmp_path.write_text(_json.dumps(payload, indent=2))
            tmp_path.replace(pending_path)

        def _write_audit(event: str, **payload: Any) -> bool:
            """Durably record an operator override before it can trade.

            This is deliberately runner-owned: the bot asks for a change,
            but only the process that built the exact risk-checked plan may
            say what was actually confirmed. If the state volume is not
            writable, a custom plan fails closed rather than becoming an
            untraceable execution path.
            """
            import os as _os

            record = {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "event": event,
                "cycle_id": cycle_id,
                **payload,
            }
            try:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as audit_file:
                    audit_file.write(_json.dumps(record, sort_keys=True) + "\n")
                    audit_file.flush()
                    _os.fsync(audit_file.fileno())
            except OSError as exc:
                logger.bind(component="cycle").error(f"approval audit write failed: {exc!r}")
                self.alerts.warning(
                    "⚠️ could not record the custom approval audit trail — no orders submitted."
                )
                return False
            return True

        def _wait_for_decision(deadline: float) -> dict[str, Any] | None:
            """Consume one bot decision, failing closed on malformed JSON."""
            while _time.monotonic() < deadline:
                if decision_path.exists():
                    try:
                        payload = _json.loads(decision_path.read_text())
                    except Exception:
                        self.alerts.warning(
                            "⚠️ malformed cycle-approval decision — no orders submitted."
                        )
                        return None
                    finally:
                        decision_path.unlink(missing_ok=True)
                    if not isinstance(payload, dict):
                        self.alerts.warning(
                            "⚠️ invalid cycle-approval decision — no orders submitted."
                        )
                        return None
                    return payload
                _time.sleep(self.APPROVAL_POLL_INTERVAL_S)
            return None

        def _matches_current_cycle(decision: dict[str, Any], cycle_id: str) -> bool:
            decided_id = str(decision.get("id") or "")
            if not decided_id or decided_id == cycle_id:
                return True
            self.alerts.warning(
                f"⚠️ ignoring an approval for cycle `{decided_id[:8]}` — "
                f"this is cycle `{cycle_id[:8]}`. No orders submitted.\n"
                "_Re-issue against the current basket if you still want it._"
            )
            logger.bind(component="cycle").warning(
                f"approval id mismatch: decision {decided_id!r} vs cycle {cycle_id!r}"
            )
            return False

        cycle_id = self._build_cycle_id(account)
        from trading.runner.approval_view import build_plan, format_trade_review

        ccy = getattr(account, "base_currency", None) or "USD"
        plan = build_plan(orders, account, last_prices, fx_rates=fx_rates)
        gross = float(plan["gross_turnover"])
        equity = float(plan["equity"])
        deploy_pct = float(plan["turnover_pct"])

        # Planned symbols mark the candidate scoreboard and are the only
        # symbols an operator may filter in a custom approval. A request
        # cannot freeze an arbitrary current holding.
        chosen = {getattr(o.instrument, "symbol", "") for o in orders}
        selectable_symbols = sorted(symbol.upper() for symbol in chosen if symbol)
        plan_fingerprint = _fingerprint(orders)

        candidates_payload: list[dict[str, Any]] = []
        if candidates:
            for sym, score in candidates:
                candidates_payload.append(
                    {
                        "symbol": sym,
                        "score": float(score),
                        "in_basket": sym in chosen,
                    }
                )

        # Approval state is deliberately a single active window: it is a
        # global prompt and global decision file, not a queue. The runner's
        # cooldown stops ordinary overlap, but a slow window plus an
        # off-cycle trigger can outlive it. A second cycle must stand down
        # rather than overwrite the plan an operator is reviewing.
        approval_lock_path.parent.mkdir(parents=True, exist_ok=True)
        approval_lock = approval_lock_path.open("a")
        try:
            _fcntl.flock(approval_lock.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            approval_lock.close()
            self.alerts.warning(
                "⚠️ cycle skipped — another cycle is awaiting operator approval. "
                "No orders submitted."
            )
            return []

        def _release_approval_lock() -> None:
            try:
                _fcntl.flock(approval_lock.fileno(), _fcntl.LOCK_UN)
            finally:
                approval_lock.close()

        try:
            # Wipe a stale decision only AFTER winning the single-window
            # lock. Doing this before the lock would let a later cycle
            # erase the decision for the plan currently under review.
            decision_path.unlink(missing_ok=True)
            pending_payload = {
                "id": cycle_id,
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "phase": "initial",
                "plan_fingerprint": plan_fingerprint,
                "n_orders": len(orders),
                "gross": gross,
                "currency": ccy,
                "equity": equity,
                "deploy_pct": deploy_pct,
                "plan": plan,
                "selectable_symbols": selectable_symbols,
                "can_pick": bool(candidates_payload and rebuild_from_picks),
                "candidates": candidates_payload,
                # The bot refuses every approval while halted EXCEPT one
                # carrying this flag. It is written by the runner, which is
                # the only process that verified the orders reduce-only.
                "defensive_reduce_only": bool(defensive),
            }
            _write_pending(pending_payload)
        except OSError as exc:
            _release_approval_lock()
            logger.bind(component="cycle").error(f"could not publish approval window: {exc!r}")
            self.alerts.warning(
                "⚠️ could not publish the cycle approval window — no orders submitted."
            )
            return []

        timeout_s = float(_settings.cycle_approval_timeout_s)
        prompt = format_trade_review(pending_payload, include_controls=True)
        if defensive:
            prompt += (
                "\n🛡 *Reduce-only.* Execution is halted, so this basket can only lower "
                "exposure — every order was checked against the live book and re-checked "
                "immediately before submission.\n"
                f"`/approve` sends it · `/approve 80` sends 80% of each size · `/reject` drops it."
                f"\n⏱ Auto-rejects after {int(timeout_s / 60)} min."
            )
        else:
            prompt += f"\n`/approve 80` scales the whole plan · `/approve flat` replaces it with a full exit.\n⏱ Auto-rejects after {int(timeout_s / 60)} min."

        # Buttons carry the cycle id, so a tap on an older prompt in the
        # chat history is refused rather than applied to this basket.
        from trading.bot.keyboards import approval_keyboard

        self.alerts.info(prompt, buttons=approval_keyboard(cycle_id))

        def _cleanup() -> None:
            try:
                pending_path.unlink(missing_ok=True)
                decision_path.unlink(missing_ok=True)
            finally:
                _release_approval_lock()

        def _handle_custom_approval(request: dict[str, Any], deadline: float) -> list[Any]:
            """Build one filtered plan, then require its exact confirmation."""
            try:
                # Custom edits have no backward-compatible unbound form:
                # they must originate from the current bot and displayed
                # plan, rather than an old process writing a vague action.
                if not str(request.get("id") or ""):
                    self.alerts.warning("⚠️ unbound custom approval request — no orders submitted.")
                    return []
                if str(request.get("plan_fingerprint") or "") != plan_fingerprint:
                    self.alerts.warning(
                        "⚠️ custom approval did not match the displayed plan — no orders submitted."
                    )
                    return []
                mode = str(request.get("mode") or "").lower()
                raw_symbols = request.get("symbols")
                if (
                    mode not in {"only", "except"}
                    or not isinstance(raw_symbols, list)
                    or not raw_symbols
                    or rebuild_with_frozen_symbols is None
                ):
                    self.alerts.warning("⚠️ invalid custom approval request — no orders submitted.")
                    return []
                symbols = [str(symbol).strip().upper() for symbol in raw_symbols]
                if (
                    any(not symbol for symbol in symbols)
                    or len(set(symbols)) != len(symbols)
                    or set(symbols) - set(selectable_symbols)
                ):
                    self.alerts.warning(
                        "⚠️ custom approval named symbols outside this plan — no orders submitted."
                    )
                    return []
                try:
                    revised_orders, revised_decisions = rebuild_with_frozen_symbols(mode, symbols)
                except Exception as exc:
                    logger.bind(component="cycle").warning(
                        f"custom approval rebuild failed: {exc!r}"
                    )
                    self.alerts.warning(
                        f"⚠️ could not build the revised plan ({type(exc).__name__}) — "
                        "no orders submitted."
                    )
                    return []
                if not revised_orders:
                    risk_notes = [
                        decision.reason
                        for decision in revised_decisions
                        if decision.action != "allow"
                    ]
                    detail = f"\n{risk_notes[0]}" if risk_notes else ""
                    self.alerts.warning(
                        f"⛔ revised cycle `{cycle_id[:8]}` produced no executable orders. "
                        f"No orders submitted.{detail}"
                    )
                    return []

                preview_id = _uuid.uuid4().hex[:12]
                revised_plan = build_plan(revised_orders, account, last_prices, fx_rates=fx_rates)
                revised_gross = float(revised_plan["gross_turnover"])
                revised_pct = float(revised_plan["turnover_pct"])
                preview_orders = _order_payload(revised_orders)
                preview_payload = {
                    "id": cycle_id,
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "phase": "custom_preview",
                    "plan_fingerprint": plan_fingerprint,
                    "preview_id": preview_id,
                    "preview_fingerprint": _fingerprint(revised_orders),
                    "mode": mode,
                    "symbols": symbols,
                    "n_orders": len(revised_orders),
                    "gross": revised_gross,
                    "currency": ccy,
                    "equity": equity,
                    "deploy_pct": revised_pct,
                    "plan": revised_plan,
                    "orders": preview_orders,
                }
                _write_pending(preview_payload)
                if not _write_audit(
                    "custom_preview",
                    mode=mode,
                    symbols=symbols,
                    plan_fingerprint=plan_fingerprint,
                    preview_id=preview_id,
                    preview_fingerprint=_fingerprint(revised_orders),
                    orders=preview_orders,
                    risk_adjustments=[
                        decision.reason
                        for decision in revised_decisions
                        if decision.action != "allow"
                    ],
                ):
                    return []
                mode_text = (
                    f"apply only the planned changes for `{', '.join(symbols)}`"
                    if mode == "only"
                    else f"freeze `{', '.join(symbols)}`"
                )
                preview_lines = [
                    format_trade_review(preview_payload, final_confirmation=True),
                    mode_text,
                ]
                risk_notes = [
                    decision.reason for decision in revised_decisions if decision.action != "allow"
                ]
                if risk_notes:
                    preview_lines.extend(
                        ["", "*Risk adjustments:*"] + [f"  • {note}" for note in risk_notes[:3]]
                    )
                preview_lines.extend(
                    [
                        "",
                        f"Confirm with `/approve confirm {preview_id}` or the button below.",
                        "`/reject` cancels this revised plan.",
                    ]
                )
                from trading.bot.keyboards import revised_approval_keyboard

                self.alerts.info(
                    "\n".join(preview_lines),
                    buttons=revised_approval_keyboard(cycle_id, preview_id),
                )

                confirmation = _wait_for_decision(deadline)
                if confirmation is None:
                    self.alerts.warning(
                        f"⏱ revised cycle `{cycle_id[:8]}` auto-rejected — no final confirmation."
                    )
                    return []
                if not _matches_current_cycle(confirmation, cycle_id):
                    return []
                follow_up = str(confirmation.get("action", "reject")).lower()
                if follow_up == "reject":
                    self.alerts.info(f"❌ revised cycle `{cycle_id[:8]}` rejected by operator.")
                    return []
                if (
                    follow_up != "custom_confirm"
                    or str(confirmation.get("preview_id") or "") != preview_id
                    or str(confirmation.get("preview_fingerprint") or "")
                    != _fingerprint(revised_orders)
                ):
                    self.alerts.warning(
                        "⚠️ revised-plan confirmation did not match the displayed preview — "
                        "no orders submitted."
                    )
                    return []
                if not _write_audit(
                    "custom_confirmed",
                    mode=mode,
                    symbols=symbols,
                    plan_fingerprint=plan_fingerprint,
                    preview_id=preview_id,
                    preview_fingerprint=_fingerprint(revised_orders),
                    orders=preview_orders,
                ):
                    return []
                self.alerts.info(
                    f"✅ revised cycle `{cycle_id[:8]}` confirmed; submitting "
                    f"{len(revised_orders)} risk-checked order(s)."
                )
                return revised_orders
            finally:
                _cleanup()

        deadline = _time.monotonic() + timeout_s
        decision = _wait_for_decision(deadline)
        if decision is None:
            _cleanup()
            self.alerts.warning(
                f"⏱ cycle `{cycle_id[:8]}` auto-rejected after "
                f"{int(timeout_s / 60)} min — no operator response."
            )
            return []
        if not _matches_current_cycle(decision, cycle_id):
            _cleanup()
            return []

        action = str(decision.get("action", "reject")).lower()
        if action == "custom_request":
            return _handle_custom_approval(decision, deadline)
        _cleanup()
        if action == "reject":
            self.alerts.info(f"❌ cycle `{cycle_id[:8]}` rejected by operator.")
            return []
        if action == "flatten":
            self.alerts.info(
                f"⏸ cycle `{cycle_id[:8]}` — flatten requested; replacing basket with full close."
            )
            from trading.core.types import Position

            positions: list[Position] = list((account.positions or {}).values())
            return self.risk_manager.force_flatten_orders(
                positions,
                ts=datetime.now(tz=timezone.utc),
                working_orders=self._working_orders(),
            )
        if action == "scale":
            factor = float(decision.get("scale_factor", 1.0))
            factor = max(0.0, min(factor, 1.0))
            if factor < 0.001:
                self.alerts.info(f"❌ cycle `{cycle_id[:8]}` scaled to ~0% — equivalent to reject.")
                return []
            scaled_orders = self._rescale_orders(orders, factor)
            self.alerts.info(
                format_trade_review(
                    {
                        "id": cycle_id,
                        "plan": build_plan(scaled_orders, account, last_prices, fx_rates=fx_rates),
                    },
                    heading=f"🔁 *Scaled trade review* — submitting at {factor * 100:.0f}%",
                )
            )
            return scaled_orders
        if action == "pick":
            picked_symbols = [str(s) for s in decision.get("picked_symbols", []) if s]
            if not picked_symbols or rebuild_from_picks is None:
                self.alerts.info(
                    f"❌ cycle `{cycle_id[:8]}` /pick had no valid symbols — skipping."
                )
                return []
            picked_orders = rebuild_from_picks(picked_symbols)
            if picked_orders:
                self.alerts.info(
                    format_trade_review(
                        {
                            "id": cycle_id,
                            "plan": build_plan(
                                picked_orders, account, last_prices, fx_rates=fx_rates
                            ),
                        },
                        heading="🔁 *Replacement trade review* — submitting",
                    )
                )
            return picked_orders
        # action == "approve"
        self.alerts.info(f"✅ cycle `{cycle_id[:8]}` approved as-is.")
        return orders

    def _compute_top_candidates(
        self,
        prices: pd.DataFrame,
        *,
        cfg: RunnerConfig | None = None,
        top_n: int = 20,
    ) -> list[tuple[str, float]] | None:
        """Ask the first strategy that supports it for its top-N
        candidates. Used by the approval prompt to give the operator
        a larger pool than just the K finally chosen.

        For multi-strategy cycles, we use the first strategy's view —
        a richer aggregation is possible but the operator override
        (``/pick``) currently equal-weights selected names anyway, so
        a single ranked source is enough.
        """
        from trading.strategies.base import get_strategy

        cfg = cfg or self.config
        for name in cfg.strategies:
            try:
                cls = get_strategy(name)
                params = cls.Params(**cfg.strategy_params.get(name, {}))
                from trading.core.config import settings as _settings_k2
                from trading.runner.holds import apply_runtime_overrides

                params, _ = apply_runtime_overrides(params, _settings_k2.state_dir)
                ranked = cls(params=params).top_candidates(prices, top_n=top_n)
                if ranked:
                    return ranked
            except Exception:
                logger.bind(component="cycle").exception(
                    f"top_candidates failed for strategy {name!r}"
                )
        return None

    def _record_shadow_ladder(
        self,
        prices: pd.DataFrame,
        signal: Any,
        last_prices: dict[str, float],
        *,
        cfg: RunnerConfig | None = None,
    ) -> None:
        """Write every ranked candidate to the counterfactual ledger.

        Both sides of the cut, deliberately. The names above it are what
        the desk did; the names below it are the control group, and
        without a control group a track record is just a return series in
        whatever market happened to occur.

        Swallows everything. This is instrumentation — a failure here must
        never cost a cycle.
        """
        try:
            ranked = self._compute_top_candidates(prices, cfg=cfg, top_n=30)
            if not ranked:
                return
            chosen = bought_symbols(signal.target_weights)
            px_by_symbol = {k.split(":")[-1].upper(): v for k, v in (last_prices or {}).items()}
            conditions = self._regime_fingerprint(prices)
            pctiles = self._entry_percentiles(prices)

            from trading.memory.store import default_store

            mem = default_store()
            taken = 0
            try:
                for i, (raw_symbol, score) in enumerate(ranked, start=1):
                    sym = str(raw_symbol).split(":")[-1].upper()
                    is_taken = sym in chosen
                    taken += is_taken
                    mem.add_shadow(
                        symbol=sym,
                        origin="ladder",
                        disposition="taken" if is_taken else "passed",
                        rank=i,
                        score=float(score),
                        why=f"rank {i} of {len(ranked)}",
                        conditions=conditions,
                        px_at=px_by_symbol.get(sym),
                        pctile_52w=pctiles.get(sym),
                    )
            finally:
                mem.close()
            # Count the rows actually written, not the size of ``chosen``.
            # The two differ (a held name can sit outside the top 30), and
            # logging the latter is what hid the all-taken bug behind a
            # number too large to be a basket.
            logger.bind(component="cycle").info(
                f"shadow ledger: recorded {len(ranked)} candidate(s), "
                f"{taken} taken / {len(ranked) - taken} passed"
            )
        except Exception:
            logger.bind(component="cycle").exception("shadow ladder write failed")

    def _entry_percentiles(self, prices: pd.DataFrame, window: int = 252) -> dict[str, float]:
        """Where each name sits in its 52-week range today, 0=low, 1=high.

        Stored on the shadow row so the ledger can later test whether
        buying near the top of the range actually worked. The Quant
        charter already asserts it does not; this is what would let that
        assertion be checked rather than believed.
        """
        try:
            closes = prices["close"] if "close" in getattr(prices, "columns", []) else prices
            recent = closes.iloc[-window:]
            lo, hi, last = recent.min(), recent.max(), closes.iloc[-1]
            span = hi - lo
            out: dict[str, float] = {}
            for sym in getattr(closes, "columns", []):
                s = float(span.get(sym, 0.0) or 0.0)
                if s <= 0:  # flat or single-price history tells us nothing
                    continue
                out[str(sym).split(":")[-1].upper()] = float(
                    min(1.0, max(0.0, (last[sym] - lo[sym]) / s))
                )
            return out
        except Exception:
            return {}

    def _regime_fingerprint(self, prices: pd.DataFrame) -> dict[str, Any]:
        """A coarse description of what the market looked like today.

        Stamped on every shadow row so the ledger can later be sliced by
        conditions — "our passes beat our picks, but only in high
        dispersion" is a usable finding; "our passes beat our picks" on
        its own is not actionable. Kept deliberately cheap and derived
        only from the price frame already in hand: a fingerprint that
        needs a network call would get skipped on exactly the days worth
        recording.
        """
        try:
            import numpy as np

            closes = prices["close"] if "close" in getattr(prices, "columns", []) else prices
            rets = closes.pct_change().dropna(how="all")
            if rets.empty:
                return {}
            recent = rets.iloc[-21:]
            vol = float(np.nanstd(recent.to_numpy()) * np.sqrt(252))
            dispersion = float(np.nanmean(np.nanstd(recent.to_numpy(), axis=1)))
            above_200 = None
            if len(closes) >= 200:
                ma = closes.rolling(200).mean().iloc[-1]
                last = closes.iloc[-1]
                above_200 = float((last > ma).mean())
            return {
                "vol_ann": round(vol, 4),
                "vol_bucket": (
                    "low"
                    if vol < 0.12
                    else "normal"
                    if vol < 0.20
                    else "elevated"
                    if vol < 0.30
                    else "stress"
                ),
                "dispersion": round(dispersion, 4),
                "breadth_above_200d": None if above_200 is None else round(above_200, 3),
            }
        except Exception:
            return {}

    def _build_cycle_id(self, account: Any) -> str:
        """Stable-ish id for this approval round. Uses the snapshot ts
        if available so the operator can correlate with /balances."""
        import uuid as _uuid

        ts = getattr(account, "ts", None)
        if ts is None:
            return _uuid.uuid4().hex
        return f"{ts.strftime('%Y%m%d-%H%M%S')}-{_uuid.uuid4().hex[:6]}"

    def _rescale_orders(self, orders: list[Any], factor: float) -> list[Any]:
        """Return new orders with quantities scaled by ``factor``.

        For whole-share instruments we round (not truncate) so a scale
        of 0.8 on 169 shares = 135.2 → 135 (vs int() → 135). Orders
        whose scaled quantity rounds to zero are dropped.
        """
        from trading.core.types import AssetClass

        out: list[Any] = []
        for o in orders:
            new_qty = float(o.quantity) * factor
            if o.instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF):
                new_qty = float(round(new_qty))
            if abs(new_qty) < 1e-9:
                continue
            out.append(o.model_copy(update={"quantity": new_qty}))
        return out

    def _announce_basket(
        self,
        orders: list[Any],
        account: Any,
        last_prices: dict[str, float],
        *,
        fx_rates: dict[str, float] | None = None,
    ) -> None:
        """Telegram-announce the planned order list before broker submission.

        Lets the operator see *what* the strategy decided this cycle (which
        names, what size, what % of equity) without having to wait for
        fills or grep order_store.
        """
        if not orders:
            self.alerts.info("📊 cycle plan: no orders — portfolio already on target.")
            return

        from trading.runner.approval_view import build_plan, format_trade_review

        self.alerts.info(
            format_trade_review(
                {"plan": build_plan(orders, account, last_prices, fx_rates=fx_rates)},
                heading="📊 *Cycle plan*",
            )
        )

    def _apply_operator_mode(self, weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Apply the operator-set mode from ``state/mode.json``.

        Default mode is NEUTRAL (pass-through) — this only reshapes
        weights when the operator has set DEFENSE / BEAR / FLATTEN via
        the Telegram bot or CLI.
        """
        try:
            from trading.core.config import settings
            from trading.runtime.mode import Mode, read_mode
            from trading.selection.mode_overlay import apply_mode
        except ImportError:
            return weights  # graceful: if anything is wrong with imports, skip

        mode_path = settings.state_dir / "mode.json"
        state = read_mode(mode_path)
        if state.mode in (Mode.BULL, Mode.NEUTRAL):
            return weights  # fast path — no reshape
        adjusted = apply_mode(weights, prices, state.mode)
        self.alerts.info(
            f"mode active: {state.mode.value} "
            f"(set by {state.set_by} at {state.set_at[:19] if state.set_at else '?'})"
        )
        return adjusted

    #: Fraction of its weight each name keeps under DEFENSE / BEAR, applied
    #: to the COMBINED book after the PM merge. Deliberately reuses the
    #: strategy-side gross targets from ModePolicy so "defense" means one
    #: thing across both books.
    def _mode_scale(self, mode: Any) -> float | None:
        """Scale factor this mode implies for an already-built book.

        FLATTEN -> 0.0. DEFENSE / BEAR -> their strategy gross target.
        BULL / NEUTRAL -> None (leave the book alone).

        Note this applies the SCALING half of the mode only. The
        defensive-ETF sleeve that ``apply_mode`` adds is a strategy-side
        construction that needs a price frame; rotating the PM's
        discretionary book into ETFs on its behalf would be inventing a
        decision it did not make. Cutting exposure is a risk action and
        belongs to the operator; choosing replacements is not.
        """
        from trading.runtime.mode import Mode
        from trading.selection.mode_overlay import ModePolicy

        policy = ModePolicy()
        if mode == Mode.FLATTEN:
            return 0.0
        if mode == Mode.DEFENSE:
            return float(policy.defense_strategy_gross)
        if mode == Mode.BEAR:
            return float(policy.bear_strategy_gross)
        return None

    def _pm_mode_scale(self) -> tuple[float | None, str]:
        """The active mode's scale for the PM sleeve, and its name.

        Applied to the PM's contribution ONLY, inside the merge. The
        strategy side is already reshaped at step 6b — scaling the
        combined book afterwards would scale the strategy twice.

        Never raises: the operator sets a mode precisely when they are
        already nervous, and an overlay that throws must not stop a cycle.
        """
        try:
            from trading.core.config import settings as _s
            from trading.runtime.mode import read_mode

            state = read_mode(Path(_s.state_dir) / "mode.json")
            return self._mode_scale(state.mode), state.mode.value
        except Exception:
            logger.bind(component="cycle").exception(
                "could not read operator mode; PM sleeve left unscaled"
            )
            return None, "unknown"

    def _generate_combined_weights(
        self,
        prices: pd.DataFrame,
        *,
        cfg: RunnerConfig | None = None,
        position_symbols: set[str] | None = None,
    ) -> pd.DataFrame:
        """Run each configured strategy on ``prices`` and combine the
        per-strategy weight frames.

        For risk-aware combiners (inverse_vol, min_variance, dsr_weighted),
        we also compute each strategy's recent OOS returns by running an
        in-process backtest on the same price history. That's cheap (~ms
        per strategy on a 252-bar daily frame) and avoids carrying a
        separate per-strategy returns table.
        """
        from trading.backtest import ZERO_COSTS, run_vectorized
        from trading.selection.combine import (
            dsr_weighted,
            equal_weight,
            inverse_vol,
            min_variance,
        )

        cfg = cfg or self.config
        weights_by_strategy: dict[str, pd.DataFrame] = {}
        for name in cfg.strategies:
            cls = get_strategy(name)
            params = cls.Params(**cfg.strategy_params.get(name, {}))
            # Operator runtime state: /k override + held-position slot
            # reservation. No-op for strategies without a k parameter.
            from trading.core.config import settings as _settings_k
            from trading.runner.holds import apply_runtime_overrides

            params, notes = apply_runtime_overrides(
                params, _settings_k.state_dir, position_symbols=position_symbols
            )
            for note in notes:
                logger.bind(component="cycle", strategy=name).info(note)
            strat = cls(params=params)
            weights_by_strategy[name] = strat.generate(prices)

        if len(weights_by_strategy) == 1:
            return next(iter(weights_by_strategy.values()))

        combiner = cfg.combiner
        if combiner == "equal_weight":
            return equal_weight(weights_by_strategy)

        # Risk-aware combiners need per-strategy returns; compute them with
        # the vectorized engine on the same price frame.
        returns_by_strategy: dict[str, pd.Series] = {}
        for name, w in weights_by_strategy.items():
            result = run_vectorized(prices, w, costs=ZERO_COSTS)
            returns_by_strategy[name] = result.returns

        lookback = cfg.combiner_lookback
        if combiner == "inverse_vol":
            return inverse_vol(weights_by_strategy, returns_by_strategy, lookback=lookback)
        if combiner == "min_variance":
            return min_variance(weights_by_strategy, returns_by_strategy, lookback=lookback)
        if combiner == "dsr_weighted":
            return dsr_weighted(
                weights_by_strategy,
                returns_by_strategy,
                periods_per_year=cfg.periods_per_year,
            )
        raise ValueError(f"unknown combiner={combiner!r}")

    def _weights_to_signal(
        self,
        weights: pd.DataFrame,
        instruments: list[Instrument],
        ts: datetime,
        *,
        cfg: RunnerConfig | None = None,
    ) -> Signal:
        last_row = weights.iloc[-1]
        # Match column names to instrument.symbol -> instrument.key.
        sym_to_key = {ins.symbol: ins.key for ins in instruments}
        target_weights = {
            sym_to_key[sym]: float(last_row[sym]) for sym in last_row.index if sym in sym_to_key
        }
        active = (cfg or self.config).strategies
        return Signal(
            ts=ts,
            strategy="+".join(active),
            target_weights=target_weights,
        )

    def _last_prices(
        self,
        prices: pd.DataFrame,
        instruments: list[Instrument],
    ) -> dict[str, float]:
        last_row = prices.iloc[-1]
        return {
            ins.key: float(last_row[ins.symbol])
            for ins in instruments
            if ins.symbol in last_row.index
        }
