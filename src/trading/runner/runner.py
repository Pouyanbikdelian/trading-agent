"""Top-level Runner — wires a Cycle to APScheduler.

The Runner builds the heavyweight dependencies (cache, store, risk manager,
broker, alerts) from a ``RunnerConfig``, then schedules ``Cycle.run_cycle``
on a crontab trigger. Two execution modes:

* ``run_forever()`` — start the scheduler and block on ``asyncio.run``.
  Used by the CLI ``trading paper`` / ``trading live`` commands.
* ``run_once()`` — fire a single cycle synchronously without the scheduler.
  Used by tests, by ``--once`` CLI invocation, and as a way to validate
  the wiring before scheduling.

Construction goes through ``Runner.from_config(cfg)`` so we keep the
constructor dependency-injectable for tests but the production path picks
sane defaults from ``settings``.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from trading.core.config import settings
from trading.core.logging import logger
from trading.core.types import AssetClass, Instrument
from trading.data.base import DataSource
from trading.data.cache import ParquetCache
from trading.execution.base import Broker
from trading.execution.simulator import Simulator
from trading.execution.store import OrderStore
from trading.risk.limits import RiskLimits
from trading.risk.manager import RiskManager
from trading.runner.alerts import TelegramAlerts
from trading.runner.config import RunnerConfig
from trading.runner.cycle import Cycle, CycleReport
from trading.runner.state import RunnerStore


def _default_source_factory(instrument: Instrument) -> DataSource:
    """Pick the right adapter per asset class. Mirrors cli._source_for so
    the runner and the CLI fetch from identical sources."""
    cls = instrument.asset_class
    if cls in (AssetClass.EQUITY, AssetClass.ETF):
        from trading.data.yfinance_source import YFinanceSource

        return YFinanceSource()
    if cls == AssetClass.CRYPTO:
        from trading.data.ccxt_source import CcxtSource

        return CcxtSource(exchange_id=instrument.exchange or "binance")
    if cls == AssetClass.FX:
        from trading.data.ibkr_source import IbkrSource

        return IbkrSource()
    raise ValueError(f"no DataSource configured for asset_class={cls.value}")


def _fetch_spy_vix(lookback_days: int = 260) -> tuple[Any, Any]:
    """Pull SPY + ^VIX daily series from yfinance for the advisor.

    Lightweight — only two symbols. Returns (spy_series, vix_series).
    Either may be empty/None if the fetch failed; the caller treats
    failure as "no data, skip this poll."
    """
    try:
        import pandas as pd
        import yfinance as yf
    except Exception:
        return None, None
    try:
        raw = yf.download(
            "SPY ^VIX",
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
        )
        if isinstance(raw.columns, pd.MultiIndex):
            spy = raw["SPY"]["Close"].dropna()
            vix = raw["^VIX"]["Close"].dropna()
        else:
            spy = raw["Close"].dropna()
            vix = None
        if spy.index.tz is None:
            spy.index = spy.index.tz_localize("UTC")
        if vix is not None and vix.index.tz is None:
            vix.index = vix.index.tz_localize("UTC")
        return spy.sort_index(), (vix.sort_index() if vix is not None else None)
    except Exception:
        return None, None


class Runner:
    """Coordinates one Cycle on an APScheduler crontab. Holds no state of
    its own — restart safety comes from the SQLite stores and halt file."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        cycle: Cycle,
        broker: Broker,
        alerts: TelegramAlerts,
    ) -> None:
        self.config = config
        self.cycle = cycle
        self.broker = broker
        self.alerts = alerts
        self._scheduler: Any = None

    # -------------------------------------------------- factory

    @classmethod
    def from_config(
        cls,
        config: RunnerConfig,
        *,
        broker: Broker | None = None,
        alerts: TelegramAlerts | None = None,
        source_factory: Callable[[Instrument], DataSource] | None = None,
    ) -> Runner:
        settings.ensure_dirs()
        state_dir = settings.state_dir

        cache = ParquetCache(settings.data_dir)
        order_store = OrderStore(config.order_db_path or (state_dir / "orders.db"))
        runner_store = RunnerStore(config.state_db_path or (state_dir / "runner.db"))
        risk_manager = RiskManager(
            RiskLimits.from_settings(settings),
            halt_state_path=Path(config.halt_state_path or (state_dir / "halt.json")),
        )

        if broker is None:
            broker = Simulator(initial_cash=config.initial_cash)
            broker.connect()
        if alerts is None:
            alerts = TelegramAlerts(
                token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
                enabled=bool(settings.telegram_bot_token and settings.telegram_chat_id),
            )

        heartbeat_path = Path(config.heartbeat_path or (state_dir / "heartbeat.json"))

        # Optional playbook: load the YAML and build a VIX-based regime
        # provider. The cycle treats playbook == None as the static path.
        playbook = None
        regime_label_fn = None
        if config.playbook_path:
            from trading.runner.playbook import load_playbook

            playbook = load_playbook(config.playbook_path)
            regime_label_fn = _build_regime_label_fn(playbook)

        cycle = Cycle(
            config,
            cache=cache,
            source_factory=source_factory or _default_source_factory,
            broker=broker,
            risk_manager=risk_manager,
            order_store=order_store,
            runner_store=runner_store,
            alerts=alerts,
            heartbeat_path=heartbeat_path,
            playbook=playbook,
            regime_label_fn=regime_label_fn,
        )
        return cls(config, cycle=cycle, broker=broker, alerts=alerts)

    # -------------------------------------------------- single-shot

    def run_once(self) -> CycleReport:
        """Fire one cycle synchronously. The simulator path will need its
        own ``step(ts, bars)`` call from the caller; this method assumes
        the broker is already up-to-date."""
        return self.cycle.run_cycle()

    # -------------------------------------------------- scheduler

    async def run_forever(self) -> None:
        """Start APScheduler and block until SIGINT / SIGTERM."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        self._scheduler = AsyncIOScheduler(timezone=self.config.schedule_tz)
        trigger = CronTrigger.from_crontab(
            self.config.schedule_cron,
            timezone=self.config.schedule_tz,
        )
        self._scheduler.add_job(self._run_cycle_async, trigger, id="cycle", replace_existing=True)

        # Off-cycle trigger watcher: polls state/trigger_now.flag every 30s
        # and fires a cycle when the operator (via /mode confirm or
        # `trading mode set X --now`) drops the flag.
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler.add_job(
            self._check_trigger_flag,
            IntervalTrigger(seconds=30),
            id="trigger_watcher",
            replace_existing=True,
        )

        # Advisory risk monitor: hourly poll of SPY+VIX, push a Telegram
        # alert on new triggers. NEVER auto-applies a mode change — only
        # informs the operator. Disabled if Telegram isn't configured.
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self._scheduler.add_job(
                self._run_advisor_async,
                IntervalTrigger(hours=1),
                id="risk_advisor",
                replace_existing=True,
            )
            # HMM regime advisor: once daily after US close (≈ 22:00 UTC).
            # Slow signal — daily granularity is enough. Complements the
            # hourly SMA/VIX advisor without spamming.
            self._scheduler.add_job(
                self._run_hmm_advisor_async,
                CronTrigger(hour=22, minute=15, timezone="UTC"),
                id="hmm_advisor",
                replace_existing=True,
            )

        self._scheduler.start()
        self.alerts.info(
            f"runner started — universe={self.config.universe} "
            f"strategies={self.config.strategies} cron={self.config.schedule_cron}"
        )
        logger.bind(component="runner").info(
            f"scheduler started — next run: {self._scheduler.get_job('cycle').next_run_time}"
        )

        # Park until cancelled.
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # Windows / restricted env
                loop.add_signal_handler(sig, stop_event.set)
        try:
            await stop_event.wait()
        finally:
            await self._shutdown()

    async def _run_cycle_async(self) -> None:
        # APScheduler will execute coroutine jobs natively; we run the
        # synchronous cycle in a thread so a slow run doesn't block the loop.
        report = await asyncio.to_thread(self.cycle.run_cycle)
        if report.status == "error":
            logger.bind(component="runner").error(f"cycle error: {report.error}")
        elif report.status == "halted":
            logger.bind(component="runner").warning("cycle halted by risk manager")

    async def _run_hmm_advisor_async(self) -> None:
        """Daily: refit a 3-state Gaussian HMM on the last ~5 years of
        SPY log-returns and push a Telegram alert when the labeled
        regime (bear/neutral/bull) changes. Advisory only — never writes
        ``mode.json``. Best-effort: any failure is logged and swallowed.
        """
        try:
            spy, _vix = await asyncio.to_thread(_fetch_spy_vix, 1300)
            if spy is None or len(spy) < 300:
                logger.bind(component="hmm_advisor").info(
                    "HMM advisor skipped — not enough SPY history yet"
                )
                return
            import numpy as np

            log_ret = np.log(spy).diff().dropna()
            log_ret.name = "SPY"
            from trading.runtime.hmm_advisor import poll_and_alert

            await poll_and_alert(spy_returns=log_ret)
        except Exception:
            logger.bind(component="hmm_advisor").exception("HMM advisor failed")

    async def _run_advisor_async(self) -> None:
        """Hourly: poll SPY+VIX, push Telegram alert on new risk events.

        Never modifies mode.json. Pure advisory. Failure is logged and
        swallowed — a flaky network mustn't break the runner.
        """
        try:
            spy, vix = await asyncio.to_thread(_fetch_spy_vix)
            if spy is None or spy.empty:
                return
            from trading.runtime.advisor import poll_and_alert

            await poll_and_alert(spy=spy, vix=vix)
        except Exception:
            logger.bind(component="advisor").exception("advisor poll failed")

    async def _check_trigger_flag(self) -> None:
        """Off-cycle trigger watcher.

        When the bot writes ``state/trigger_now.flag`` (typically after
        a mode-change confirmation), we fire one cycle immediately,
        outside the cron schedule. The flag is consumed (deleted) before
        we run so a slow cycle doesn't get re-triggered.
        """
        from trading.core.config import settings

        flag_path = settings.state_dir / "trigger_now.flag"
        if not flag_path.exists():
            return
        try:
            payload = flag_path.read_text()
            flag_path.unlink()  # consume first — re-entry safe
            logger.bind(component="runner").info(f"off-cycle trigger fired: {payload[:120]}")
            await self._run_cycle_async()
        except Exception:
            logger.bind(component="runner").exception("off-cycle trigger failed")

    async def _shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
        self.alerts.info("runner stopped")
        logger.bind(component="runner").info("scheduler stopped")
        try:
            self.broker.disconnect()
        except Exception:
            logger.bind(component="runner").exception("broker disconnect failed")


def _build_regime_label_fn(playbook: Any) -> Callable[[datetime], str]:
    """Build a callable that returns the current regime label.

    For ``classifier: vix``, we fit a VixRegime classifier once at runner
    construction time, cache it, and re-use across cycles. The VIX history
    is fetched lazily and re-fetched at most once per UTC day — yfinance
    is rate-limited and we don't want a heavy call every 5-minute cycle.
    """
    if playbook.classifier == "vix":
        return _vix_regime_label_fn()
    raise ValueError(
        f"playbook.classifier={playbook.classifier!r} not wired yet; only 'vix' is supported"
    )


def _vix_regime_label_fn() -> Callable[[datetime], str]:
    """Closure around a lazily-fit VixRegime + a once-per-day refresh."""
    from trading.regime.vix import DEFAULT_VIX_LABELS, VixRegime, fetch_vix_levels

    state: dict[str, Any] = {"classifier": None, "last_refresh": None, "levels": None}

    def _label(ts: datetime) -> str:
        # Refresh at most once per UTC day.
        today = ts.date()
        if state["last_refresh"] != today or state["classifier"] is None:
            levels = fetch_vix_levels(end=ts)
            classifier = VixRegime().fit(levels)
            state["classifier"] = classifier
            state["levels"] = levels
            state["last_refresh"] = today

        # Predict on the latest VIX observation; fall back to "mid_vol" if
        # the levels series happens to be empty (network blip, weekend).
        levels = state["levels"]
        if levels is None or len(levels) == 0:
            return DEFAULT_VIX_LABELS[1]
        labels = state["classifier"].predict(levels.iloc[-1:])
        label_id = int(labels.iloc[-1])
        return DEFAULT_VIX_LABELS.get(label_id, f"state_{label_id}")

    return _label
