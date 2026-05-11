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
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from trading.runner.alerts import NullAlerts, TelegramAlerts
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
    ) -> "Runner":
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
        self._scheduler.add_job(self._run_cycle_async, trigger,
                                 id="cycle", replace_existing=True)
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
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:   # Windows / restricted env
                pass
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

    async def _shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
        self.alerts.info("runner stopped")
        logger.bind(component="runner").info("scheduler stopped")
        try:
            self.broker.disconnect()
        except Exception:   # noqa: BLE001
            logger.bind(component="runner").exception("broker disconnect failed")
