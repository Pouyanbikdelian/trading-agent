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
        account = self._fetch_account(ts_start)

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

        # 7. Build the Signal from the last row of the weights frame.
        signal = self._weights_to_signal(weights, instruments, ts_start, cfg=cfg)
        last_prices = self._last_prices(prices, instruments)
        instruments_by_key = {ins.key: ins for ins in instruments if ins.key in last_prices}

        # 8. Risk manager: signal -> orders.
        orders, decisions = self.risk_manager.signal_to_orders(
            signal,
            account=account,
            last_prices=last_prices,
            instruments=instruments_by_key,
            sector_map=cfg.sector_map or None,
            **({"order_id_factory": self._order_id_factory} if self._order_id_factory else {}),
        )

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
                logger.bind(component="cycle").exception(
                    f"submit failed for {order.client_order_id}"
                )
                self.order_store.update_status(order.client_order_id, OrderStatus.REJECTED)
                self.alerts.error(f"order submit failed: {e!r}")

        # 10. Reconcile fills since the start of this cycle.
        fills = self.broker.get_fills(since=ts_start)
        for fill in fills:
            try:
                self.order_store.save_fill(fill, client_order_id=fill.order_id)
                self.order_store.update_status(fill.order_id, OrderStatus.FILLED)
            except Exception:
                logger.bind(component="cycle").exception("save_fill failed")

        # 11. Persist the post-trade snapshot.
        try:
            post_snap = self.broker.get_account()
            self.runner_store.save_snapshot(post_snap)
        except Exception:
            logger.bind(component="cycle").exception("snapshot persistence failed")

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

    def _load_prices(self, instruments: list[Instrument], ts: datetime) -> pd.DataFrame:
        """Read each instrument's price column from the cache, optionally
        fetching fresh bars first. Returns a wide-format DataFrame aligned
        on the inner intersection of dates."""
        freq: Frequency = self.config.freq  # type: ignore[assignment]
        # Choose a generous start: we want at least `history_bars` rows. The
        # cache returns whatever it has; downstream code handles short series.
        start = ts.replace(year=ts.year - 5)  # 5 years back is more than enough
        end = ts

        series: dict[str, pd.Series] = {}
        for ins in instruments:
            df = pd.DataFrame()
            if self.config.auto_refresh:
                try:
                    source = self.source_factory(ins)
                    df = self.cache.get_bars(source, ins, start, end, freq)
                except Exception:
                    logger.bind(symbol=ins.symbol).exception(
                        "refresh failed; falling back to cache"
                    )
            if df.empty:
                df = self.cache.read(ins, freq)
            if df.empty or self.config.price_column not in df.columns:
                continue
            s = df[self.config.price_column].dropna()
            if not s.empty:
                series[ins.symbol] = s.iloc[-self.config.history_bars :]

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

    def _fetch_account(self, ts: datetime) -> AccountSnapshot:
        """Get the broker's account view. For brokers that need a step()
        call before reporting (Simulator), the runner drives that in
        ``Runner.before_cycle``; here we just ask."""
        try:
            return self.broker.get_account()
        except Exception:
            # Fall back to a fresh zero-position snapshot. This is mostly for
            # the very first cycle after process start.
            return AccountSnapshot(
                ts=ts, cash=self.config.initial_cash, equity=self.config.initial_cash
            )

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
