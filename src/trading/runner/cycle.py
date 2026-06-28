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

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

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

CycleStatus = Literal["ok", "no_orders", "halted", "error"]


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


class Cycle:
    """All-in-one bound cycle. Construct once at startup; call ``run_cycle()``
    every bar."""

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
        """Execute one full cycle. Never raises; failures become ``error``
        reports."""
        ts_start = self._clock()
        self._cycle_count += 1
        try:
            report = self._run_inner(ts_start)
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

    def _run_force_flatten(self, ts_start: datetime) -> CycleReport:
        """Playbook said this regime = stay flat. Generate closing orders for
        every open position, submit, and skip strategy generation entirely."""
        account = self._fetch_account(ts_start)
        positions = list(account.positions.values())
        orders = self.risk_manager.force_flatten_orders(
            positions,
            ts=ts_start,
            **({"order_id_factory": self._order_id_factory} if self._order_id_factory else {}),
        )

        orders_submitted = 0
        for order in orders:
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

        fills = self.broker.get_fills(since=ts_start)
        for fill in fills:
            try:
                self.order_store.save_fill(fill, client_order_id=fill.order_id)
                self.order_store.update_status(fill.order_id, OrderStatus.FILLED)
            except Exception:
                logger.bind(component="cycle").exception("save_fill failed")
        try:
            self.runner_store.save_snapshot(self.broker.get_account())
        except Exception:
            logger.bind(component="cycle").exception("snapshot persistence failed")

        return CycleReport(
            ts=ts_start,
            status="ok" if orders_submitted > 0 else "no_orders",
            orders_submitted=orders_submitted,
            fills_received=len(fills),
            decisions=[],
            duration_ms=self._elapsed_ms(ts_start),
        )

    def _run_inner(self, ts_start: datetime) -> CycleReport:
        # 0. Apply the regime playbook if one is configured. The playbook can
        # swap the universe, strategies, vol target, and per-strategy params.
        # `force_flatten: true` short-circuits to a flatten-everything cycle.
        cfg, force_flatten = self._effective_config(ts_start)
        if force_flatten:
            return self._run_force_flatten(ts_start)

        # 1. Load instruments for the universe.
        instruments = load_universe(cfg.universe)

        # 1b. Pre-strategy screens (liquidity / quality / sector momentum).
        # Cuts the universe before strategies and the price load do the
        # heavy work, so the screen is essentially free.
        if cfg.screens is not None:
            instruments = self._apply_screens(instruments, cfg)

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

        # 4. Intraday kill switches (daily-loss, drawdown).
        intraday = self.risk_manager.evaluate_intraday(account)
        if intraday.action == "halt":
            self.alerts.critical(f"HALT: {intraday.reason}")
            return CycleReport(
                ts=ts_start,
                status="halted",
                orders_submitted=0,
                fills_received=0,
                decisions=[intraday],
                duration_ms=self._elapsed_ms(ts_start),
            )

        # 5. Generate strategy weights and combine.
        logger.bind(component="cycle").info(
            f"generating weights: strategies={cfg.strategies}, "
            f"combiner={cfg.combiner}, prices_shape={prices.shape}"
        )
        weights = self._generate_combined_weights(prices, cfg=cfg)

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
        # Sector tags for the risk manager's sector cap. cfg.sector_map wins;
        # otherwise derive key -> sector from the fundamentals cache so the cap
        # actually binds instead of silently no-op'ing on an empty map.
        sector_map = self._resolve_sector_map(instruments_by_key, cfg)

        # 8. Risk manager: signal -> orders.
        logger.bind(component="cycle").info(
            f"risk manager: signal has {len(signal.target_weights)} target weights"
        )
        orders, decisions = self.risk_manager.signal_to_orders(
            signal,
            account=account,
            last_prices=last_prices,
            instruments=instruments_by_key,
            sector_map=sector_map,
            **({"order_id_factory": self._order_id_factory} if self._order_id_factory else {}),
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
        else:
            self._announce_basket(orders, account, last_prices)

        # 8d. Per-cycle operator approval (optional, off by default).
        # When require_cycle_approval=true, block here until the operator
        # /approve's, /reject's, /pick's a different basket, or the
        # timeout fires. Default paper mode skips this entirely.
        from trading.core.config import settings as _settings

        if _settings.require_cycle_approval and orders:
            candidates = self._compute_top_candidates(prices, cfg=cfg)

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
                new_signal = _Signal(ts=signal.ts, target_weights=picked_keys)
                new_orders, _new_decisions = self.risk_manager.signal_to_orders(
                    new_signal,
                    account=account,
                    last_prices=last_prices,
                    instruments=instruments_by_key,
                    sector_map=sector_map,
                    **(
                        {"order_id_factory": self._order_id_factory}
                        if self._order_id_factory
                        else {}
                    ),
                )
                return new_orders

            orders = self._request_cycle_approval(
                orders,
                account,
                last_prices,
                candidates=candidates,
                rebuild_from_picks=_rebuild_from_picks,
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
            # Re-announce the modified basket if the operator /picked or
            # /scaled, so the Telegram timeline shows the actual orders
            # that will trade.
            self._announce_basket(orders, account, last_prices)

        # 9. Submit each order. The cycle stops at first hard failure to
        #    avoid partial portfolio bringups, but logs and alerts.
        orders_submitted = 0
        for order in orders:
            try:
                self.order_store.save_order(order)
                self.broker.submit_order(order)
                self.order_store.update_status(order.client_order_id, OrderStatus.SUBMITTED)
                orders_submitted += 1
            except Exception as e:
                # Surface broker rejections explicitly in Telegram so the
                # operator sees the reason — not just a generic stack.
                err_str = f"{type(e).__name__}: {e}"[:400]
                logger.bind(component="cycle").exception(
                    f"submit failed for {order.client_order_id}"
                )
                self.order_store.update_status(order.client_order_id, OrderStatus.REJECTED)
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

        # 10. Reconcile fills since the start of this cycle.
        fills = self.broker.get_fills(since=ts_start)
        for fill in fills:
            try:
                self.order_store.save_fill(fill, client_order_id=fill.order_id)
                self.order_store.update_status(fill.order_id, OrderStatus.FILLED)
            except Exception:
                logger.bind(component="cycle").exception("save_fill failed")

        # 10b. Telegram a fill summary so the operator doesn't have to
        # poll /positions. Aggregated to keep noise low even when 8
        # names fill at once. Silent if no fills this cycle.
        if fills:
            self._announce_fills(fills)

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

    # ---------------------------------------------------------- helpers

    def _elapsed_ms(self, ts_start: datetime) -> float:
        return (self._clock() - ts_start).total_seconds() * 1000.0

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
        wide = pd.DataFrame(series).sort_index().dropna(how="any")
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
        if not cfg.fundamentals_path:
            return None
        from trading.data.fundamentals_source import read_fundamentals_cache

        funds = read_fundamentals_cache(Path(cfg.fundamentals_path))
        out: dict[str, str] = {}
        for key, ins in instruments_by_key.items():
            f = funds.get(ins.symbol)
            if f is not None and f.sector:
                out[key] = str(f.sector)
        return out or None

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
    APPROVAL_POLL_INTERVAL_S: ClassVar[float] = 2.0

    def _request_cycle_approval(
        self,
        orders: list[Any],
        account: Any,
        last_prices: dict[str, float],
        *,
        candidates: list[tuple[str, float]] | None = None,
        rebuild_from_picks: Callable[[list[str]], list[Any]] | None = None,
    ) -> list[Any]:
        """Block the cycle until the operator decides via Telegram.

        Writes the basket summary (+ optional top-N candidate scoreboard)
        to ``state/cycle_approval_pending.json`` and waits for
        ``state/cycle_approval_decision.json``. Bot produces the decision
        file in response to /approve, /approve N, /approve flat,
        /pick … or /reject.

        ``candidates`` is the strategy's top-N ranked list (best-first,
        with scores). When present, /pick can override which subset to
        trade — ``rebuild_from_picks`` then rebuilds the orders from
        those picks through the risk manager.

        Returns the list of orders to submit. Empty = no submission.
        """
        import json as _json
        import time as _time

        from trading.core.config import settings as _settings

        state_dir = _settings.state_dir
        pending_path = state_dir / self.APPROVAL_PENDING_FILE
        decision_path = state_dir / self.APPROVAL_DECISION_FILE

        # Wipe any stale decision file from a previous cycle.
        decision_path.unlink(missing_ok=True)

        cycle_id = self._build_cycle_id(account)
        ccy = getattr(account, "base_currency", None) or "USD"
        gross = sum(
            float(o.quantity) * float(last_prices.get(o.instrument.key, 0.0)) for o in orders
        )
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        deploy_pct = (gross / equity * 100.0) if equity > 0 else 0.0

        # Symbols the strategy decided to trade — used so the bot can
        # mark them on the candidate scoreboard.
        chosen = {getattr(o.instrument, "symbol", "") for o in orders}

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

        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(
            _json.dumps(
                {
                    "id": cycle_id,
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "n_orders": len(orders),
                    "gross": gross,
                    "currency": ccy,
                    "equity": equity,
                    "deploy_pct": deploy_pct,
                    "can_pick": bool(candidates_payload and rebuild_from_picks),
                    "candidates": candidates_payload,
                },
                indent=2,
            )
        )

        timeout_s = float(_settings.cycle_approval_timeout_s)
        prompt_lines = [
            f"⏸ *Awaiting approval* — cycle `{cycle_id[:8]}`",
            f"{len(orders)} orders · {deploy_pct:.0f}% of equity",
            "",
            "Reply with one of:",
            "  `/approve` — submit as-is",
            "  `/approve 80` — scale to 80% of basket size",
        ]
        if candidates_payload and rebuild_from_picks is not None:
            prompt_lines.append("  `/pick 1 3 5 8 11` — replace basket with these ranks")
        prompt_lines.extend(
            [
                "  `/approve flat` — flatten everything instead",
                "  `/reject` — skip this cycle (no orders)",
                f"⏱ Auto-rejects after {int(timeout_s / 60)} min.",
            ]
        )
        if candidates_payload:
            prompt_lines.append("")
            prompt_lines.append("*Top candidates this cycle:*")
            for i, c in enumerate(candidates_payload[:20], start=1):
                mark = "✅" if c["in_basket"] else "  "
                prompt_lines.append(f"  `{i:>2}` {mark} `{c['symbol']:<6}` {c['score']:+.1%}")

        self.alerts.info("\n".join(prompt_lines))

        deadline = _time.monotonic() + timeout_s
        decision: dict[str, Any] | None = None
        while _time.monotonic() < deadline:
            if decision_path.exists():
                try:
                    decision = _json.loads(decision_path.read_text())
                except Exception:
                    decision = None
                break
            _time.sleep(self.APPROVAL_POLL_INTERVAL_S)

        # Clean up the coordination files regardless of outcome.
        pending_path.unlink(missing_ok=True)
        decision_path.unlink(missing_ok=True)

        if decision is None:
            self.alerts.warning(
                f"⏱ cycle `{cycle_id[:8]}` auto-rejected after "
                f"{int(timeout_s / 60)} min — no operator response."
            )
            return []

        action = str(decision.get("action", "reject")).lower()
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
                positions, ts=datetime.now(tz=timezone.utc)
            )
        if action == "scale":
            factor = float(decision.get("scale_factor", 1.0))
            factor = max(0.0, min(factor, 1.0))
            if factor < 0.001:
                self.alerts.info(f"❌ cycle `{cycle_id[:8]}` scaled to ~0% — equivalent to reject.")
                return []
            self.alerts.info(
                f"✅ cycle `{cycle_id[:8]}` approved at {factor * 100:.0f}% — "
                "rescaling order quantities."
            )
            return self._rescale_orders(orders, factor)
        if action == "pick":
            picked_symbols = [str(s) for s in decision.get("picked_symbols", []) if s]
            if not picked_symbols or rebuild_from_picks is None:
                self.alerts.info(
                    f"❌ cycle `{cycle_id[:8]}` /pick had no valid symbols — skipping."
                )
                return []
            self.alerts.info(
                f"✅ cycle `{cycle_id[:8]}` /pick → {len(picked_symbols)} names: "
                f"`{', '.join(picked_symbols)}`. Rebuilding basket…"
            )
            return rebuild_from_picks(picked_symbols)
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
    ) -> None:
        """Telegram-announce the planned order list before broker submission.

        Lets the operator see *what* the strategy decided this cycle (which
        names, what size, what % of equity) without having to wait for
        fills or grep order_store.
        """
        if not orders:
            self.alerts.info("📊 cycle plan: no orders — portfolio already on target.")
            return

        from trading.core.types import Side as _Side

        equity = float(getattr(account, "equity", 0.0) or 0.0)
        cash = float(getattr(account, "cash", 0.0) or 0.0)
        ccy = getattr(account, "base_currency", None) or "USD"

        buy_lines: list[str] = []
        sell_lines: list[str] = []
        gross = 0.0
        net = 0.0
        for o in orders:
            px = last_prices.get(o.instrument.key, 0.0)
            notional = float(o.quantity) * float(px)
            gross += abs(notional)
            pct = (notional / equity * 100.0) if equity > 0 else 0.0
            # Per-share prices and per-order notionals are in the instrument's
            # currency (USD for US equities). Account totals further down are
            # in the account's base currency. We keep them visually distinct
            # so the operator knows which is which.
            line = (
                f"  {o.instrument.symbol:<6} {o.quantity:>6g} @ {px:,.2f} "
                f"= {notional:>10,.0f}  ({pct:>4.1f}%)"
            )
            if o.side == _Side.BUY:
                buy_lines.append(line)
                net += notional
            else:
                sell_lines.append(line)
                net -= notional

        deploy_pct = (gross / equity * 100.0) if equity > 0 else 0.0
        parts: list[str] = [
            f"📊 *Cycle plan* — {len(orders)} order(s)",
            f"Deploy: *{deploy_pct:.0f}%* of equity "
            f"(gross {gross:,.0f} · equity {ccy} {equity:,.0f} · cash {ccy} {cash:,.0f})",
        ]
        if buy_lines:
            parts.append("\n*BUY*")
            parts.extend(buy_lines)
        if sell_lines:
            parts.append("\n*SELL*")
            parts.extend(sell_lines)
        parts.append(f"\nNet buy: {net:,.0f} {ccy} (positive = uses cash / margin)")

        self.alerts.info("\n".join(parts))

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

    def _generate_combined_weights(
        self,
        prices: pd.DataFrame,
        *,
        cfg: RunnerConfig | None = None,
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

            params, notes = apply_runtime_overrides(params, _settings_k.state_dir)
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
