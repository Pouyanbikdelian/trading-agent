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
import json
import os
import signal
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.core.config import settings
from trading.core.logging import logger
from trading.core.types import AccountSnapshot, AssetClass, Instrument
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


_CRON_DOW_NAMES = {
    "MON": "Mondays",
    "TUE": "Tuesdays",
    "WED": "Wednesdays",
    "THU": "Thursdays",
    "FRI": "Fridays",
    "SAT": "Saturdays",
    "SUN": "Sundays",
}


def _precycle_trigger(cron: str, tz: str, *, lead_minutes: int = 60) -> Any:
    """A cron trigger ``lead_minutes`` before ``cron``, or None.

    DERIVED from the cycle's own schedule rather than hardcoded, so the
    two cannot drift: change CRON in .env and the pre-cycle warning
    follows it. A readiness check that fires an hour before a cycle that
    has since moved is worse than none — it would report "all clear" and
    then the cycle would run unchecked.

    Only handles the fixed ``m h * * DOW`` shape the runner actually uses.
    Anything cleverer (step values, lists in the minute field) returns
    None and the job is simply not scheduled, which is honest.
    """
    # Imported here, not at module scope: apscheduler is only needed when
    # a runner is actually being wired, and the rest of this module is
    # importable without it.
    from apscheduler.triggers.cron import CronTrigger

    parts = cron.split()
    if len(parts) != 5:
        return None
    minute, hour, dom, month, dow = parts
    try:
        m, h = int(minute), int(hour)
    except ValueError:
        return None  # ranges/steps in minute or hour — not our shape

    total = h * 60 + m - lead_minutes
    if total < 0:
        # Crossing midnight backwards would also shift the day-of-week,
        # and getting that subtly wrong is worse than not running.
        logger.bind(component="runner").warning(
            f"pre-cycle check skipped: {cron} minus {lead_minutes}m crosses midnight"
        )
        return None
    return CronTrigger(
        minute=total % 60, hour=total // 60, day=dom, month=month, day_of_week=dow, timezone=tz
    )


def _historian_trigger() -> Any:
    """Tuesday and Friday distillation, after the nightly grader.

    Keep the cadence in one helper so the schedule has a direct, testable
    representation rather than being buried among the runner's jobs.
    """
    from apscheduler.triggers.cron import CronTrigger

    # Keep the full post-close learning chain on New York wall time.  A UTC
    # literal put the winter grader before the cache refresh, then let the
    # historian distil that incomplete result.
    return CronTrigger(day_of_week="tue,fri", hour=19, minute=0, timezone="America/New_York")


def _add_scorecard_backfill_targets(
    symbols: set[str],
    starts: dict[str, datetime],
    targets: list[dict[str, Any]],
    *,
    default_start: datetime,
) -> None:
    """Add cacheable ungraded subjects without letting one bad row block all.

    A scorecard call may name an ETF or an index proxy outside the trading
    universe. It must join both collections: ``starts`` determines the
    requested history, while ``symbols`` controls which requests are made.
    """
    from trading.runtime.portfolio_stats import cache_symbol_for_subject

    for target in targets:
        try:
            symbol = cache_symbol_for_subject(str(target["subject"]))
            if (
                symbol in {"PORTFOLIO", "BOOK", "MARKET"}
                or not symbol.replace("-", "").replace(".", "").isalnum()
            ):
                continue  # aggregate prose such as "portfolio" is not a price series
            earliest = datetime.fromtimestamp(float(target["earliest_ts"]), tz=timezone.utc)
        except (KeyError, TypeError, ValueError, OverflowError):
            logger.bind(component="data").warning("invalid scorecard refresh target skipped")
            continue
        # ``starts`` alone is inert: the fetch loop iterates ``symbols``.
        symbols.add(symbol)
        starts[symbol] = min(starts.get(symbol, default_start), earliest - timedelta(days=3))


def _humanize_cron(expr: str) -> str:
    """Translate a 5-field cron string into something humans read.

    Only handles the common case "M H * * DOW" — falls back to the raw
    expression for anything more exotic (the operator can read cron;
    the goal is just to avoid surprising the user with `5 21 * * FRI`).
    """
    parts = expr.split()
    if len(parts) != 5:
        return expr
    minute, hour, dom, mon, dow = parts
    try:
        m = int(minute)
        h = int(hour)
    except ValueError:
        return expr
    time_s = f"{h:02d}:{m:02d} UTC"
    if dom == "*" and mon == "*" and dow == "*":
        return f"daily at {time_s}"
    if dom == "*" and mon == "*" and dow.upper() in _CRON_DOW_NAMES:
        return f"{_CRON_DOW_NAMES[dow.upper()]} {time_s}"
    return expr


def _humanize_strategy(slug: str, params: dict[str, Any]) -> str:
    """Translate a strategy slug + params into a one-line description.

    Falls back to the raw slug for strategies we haven't pretty-printed
    yet; safe to extend without coupling.
    """
    if slug == "top_k_momentum":
        k = params.get("k", 8)
        lookback = params.get("lookback", 126)
        skip = params.get("skip", 21)
        rebal = params.get("rebalance", 63)
        return (
            f"Top-{k} momentum (lookback {lookback}d, skip {skip}d, rebalance every {rebal} bars)"
        )
    if not params:
        return slug
    p = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{slug} ({p})"


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
        # Health-tracking state. Reset to 0 on a successful cycle.
        # Persisted so a container restart doesn't silently reset the counter
        # below the auto-halt threshold (audit fix #8).
        self._error_counter_path = settings.state_dir / "consecutive_errors.json"
        self._consecutive_errors: int = self._load_error_counter()
        self._last_success_ts: datetime | None = None
        # Cycle cooldown: refuse to start another cycle within this window
        # of the previous one starting (audit fix #11). Prevents overlap
        # if the cron and an off-cycle trigger fire near-simultaneously.
        self._last_cycle_start_ts: datetime | None = None
        # A best-effort, read-only market-watch repair after a late restart.
        # Keep its task so it cannot be garbage-collected while its network
        # collection is in flight and can be cancelled during shutdown.
        self._startup_market_watch_task: asyncio.Task[None] | None = None
        # The cron trigger and a post-restart catch-up can meet just after
        # the scheduled slot.  One async lock keeps that benign race from
        # issuing duplicate yfinance batches or racing the state artifact.
        self._market_watch_lock: asyncio.Lock | None = None
        # Deduplicate the operational warning emitted when a live process
        # starts/restarts after the narrow NYSE-open capture window.  The
        # safety block remains active; only the repeated Telegram noise is
        # suppressed until the reason changes or a trusted baseline arrives.
        self._last_risk_monitor_reject_reason: str | None = None

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

        # Before anything reads a baseline off disk: does this directory
        # belong to the environment we are running as? A paper cycle that
        # wrote into state/live is what halted the first live session at
        # -91.82% (2026-08-07). Raises StateEnvMismatchError — deliberately
        # fatal, deliberately not self-healing.
        from trading.core.state_env import assert_state_dir_env

        assert_state_dir_env(state_dir, settings.trading_env)

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

    def _register_core_background_jobs(self) -> None:
        """Register data, safety, and ops work independent of main-chat setup.

        The primary Telegram bot is an operator interface, not a dependency
        of cache maintenance, scorecard grading, position guards, or the
        separate ops channel.  Their own alert calls already degrade to
        no-ops when no chat is configured.
        """
        from apscheduler.triggers.cron import CronTrigger

        from trading.runtime.guards import enabled as guards_enabled
        from trading.runtime.market_watch import (
            SCHEDULE_HOUR,
            SCHEDULE_MINUTE,
            SCHEDULE_TIMEZONE,
        )

        # Position guards: ATR trailing stops + profit ratchet. Same RTH
        # cadence as the sentinel, offset 5 min. This is safety work, not a
        # notification feature, so a missing chat configuration must not
        # disable it.
        if guards_enabled():
            self._scheduler.add_job(
                self._run_guards_async,
                CronTrigger(day_of_week="mon-fri", hour="13-20", minute="5-59/15", timezone="UTC"),
                id="guards",
                replace_existing=True,
                max_instances=1,
            )

        # Broker readiness, ONE HOUR before the cycle. IBKR mandates 2FA
        # for every user and re-prompts after its daily restart and weekly
        # shutdown; IBC retries a missed prompt, but only a human can answer
        # it. This remains useful with an ops-only Telegram channel.
        try:
            pre = _precycle_trigger(self.config.schedule_cron, self.config.schedule_tz)
            if pre is None:
                logger.bind(component="runner").warning(
                    "pre-cycle broker check NOT scheduled: cannot derive a lead time from "
                    f"cron {self.config.schedule_cron!r} (crosses midnight, or an "
                    "unsupported shape). A dead gateway will only be discovered AT the cycle."
                )
            else:
                self._scheduler.add_job(
                    self._check_broker_ready_async,
                    pre,
                    id="broker_ready",
                    replace_existing=True,
                    max_instances=1,
                )
                logger.bind(component="runner").info(
                    f"pre-cycle broker check scheduled {self.PRECYCLE_LEAD_MINUTES}min "
                    f"before the cycle: {pre}"
                )
        except Exception:
            logger.bind(component="runner").exception(
                "pre-cycle broker check could not be scheduled — a dead gateway will "
                "only be discovered at the cycle itself"
            )

        # Refresh at 17:40 New York time, leaving the free daily source room
        # to publish final bars after the regular close. Market watch runs
        # later from this refreshed cache.
        self._scheduler.add_job(
            self._refresh_price_cache_async,
            CronTrigger(day_of_week="mon-fri", hour=17, minute=40, timezone="America/New_York"),
            id="price_cache_refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        # Index constituents are trading inputs, not an alert-only feature.
        self._scheduler.add_job(
            self._refresh_universes_async,
            CronTrigger(day_of_week="sun", hour=3, minute=0, timezone="UTC"),
            id="universe_refresh",
            replace_existing=True,
            max_instances=1,
        )

        # Ops watchdog: a dedicated OPS_TELEGRAM_* channel is sufficient.
        self._scheduler.add_job(
            self._run_ops_watch_async,
            CronTrigger(minute="*/5", timezone="UTC"),
            id="ops_watch",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        # Scorecard grading is local learning-state maintenance and should
        # not stop just because the interactive chat is unavailable.  It
        # starts 65 minutes after the cache refresh, after the two-hour
        # post-close data-settlement window in both DST seasons.
        self._scheduler.add_job(
            self._run_memory_grader_async,
            CronTrigger(hour=18, minute=45, timezone="America/New_York"),
            id="memory_grader",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        # The Historian owns persistent learning state, rather than the
        # interactive Telegram chat.  Preserve its existing explicit agent
        # and API-key gate, but do not make a missing primary chat disable a
        # working OPS-only deployment.
        if os.getenv("AGENTS_ENABLED", "false").lower() in ("true", "1", "yes") and (
            os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        ):
            self._scheduler.add_job(
                self._run_historian_async,
                _historian_trigger(),
                id="historian",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

        # Market watch follows the cache by 70 minutes. The local constants
        # also drive restart catch-up, preventing DST drift between paths.
        self._scheduler.add_job(
            self._run_market_watch_async,
            CronTrigger(
                day_of_week="mon-fri",
                hour=SCHEDULE_HOUR,
                minute=SCHEDULE_MINUTE,
                timezone=SCHEDULE_TIMEZONE,
            ),
            id="market_watch",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    # -------------------------------------------------- scheduler

    def preflight_unheld(self, *, require_reachable: bool = False) -> bool:
        """Refuse to arm live while positions sit unprotected.

        Only on the live path, and only once at startup — a paper runner
        trading a simulated book has nothing personal to protect, and a
        mid-session check would block a restart during market hours.

        Returns ``False`` only when a strict bootstrap could not verify the
        account's positions.  An actual unheld position still writes the
        existing halt and returns ``True``: the process remains observable
        and no risk path can submit an order while halted.
        """
        from trading.runtime.broker_ready import check_unheld_positions, format_unheld_alert

        result = check_unheld_positions(
            self.cycle.broker,
            state_dir=settings.state_dir,
            require_reachable=require_reachable,
        )
        if require_reachable and not result.get("reachable", False):
            logger.bind(component="preflight").warning(
                "live bootstrap waiting for a verified broker position read: "
                + str(result.get("reason", "unknown failure"))
            )
            return False
        if result["ok"]:
            return True
        # ALERT AND HALT — do not raise.
        #
        # This raised until 2026-08-11, and raising inside a service with
        # `restart: unless-stopped` is what turns "we noticed something"
        # into an outage: the container exits, docker restarts it, the
        # check fails identically, and the operator gets one CRITICAL
        # message per attempt. It happened three times in one afternoon —
        # after the first cycle bought the PM basket, and again the moment
        # the operator deliberately /unhold'ed a position.
        #
        # Worse than the noise: a refusal to start takes down the guards
        # protecting every OTHER position, so one unprotected name
        # disarmed the whole book.
        #
        # Halting achieves what the raise was reaching for — nothing can
        # trade, including guard exits, which route through the same
        # halt-aware command pipeline — while the runner stays up,
        # observable, and able to receive the /hold or /resume that
        # clears it.
        unheld = ", ".join(result["unheld"])
        if self._announce_unheld(set(result["unheld"])):
            try:
                self.alerts.critical(format_unheld_alert(result))
            except Exception:
                logger.bind(component="preflight").exception("unheld alert failed to send")

        from trading.risk.halt_file import set_halted

        set_halted(
            settings.state_dir,
            halted=True,
            reason=f"unprotected positions: {unheld}",
        )
        logger.bind(component="preflight").warning(
            f"starting HALTED — unprotected positions: {unheld}. "
            "/hold them or `trading preflight ack`, then /resume."
        )
        return True

    #: Remembers the last set we alerted about, so a restart with the same
    #: unprotected names is silent. A CHANGED set re-alerts: a new
    #: unprotected position is new information.
    UNHELD_ALERT_FILE = "unheld_alerted.json"

    def _announce_unheld(self, unheld: set[str]) -> bool:
        """True when this exact set has not already been announced.

        Squelching on the symbol set rather than on time: the operator
        should hear immediately about a position that appeared, and never
        twice about one they have already been told about and chosen to
        leave alone.
        """
        import json

        path = settings.state_dir / self.UNHELD_ALERT_FILE
        try:
            seen = set(json.loads(path.read_text()).get("symbols", []))
        except Exception:
            seen = set()
        if seen == unheld:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"symbols": sorted(unheld)}))
        except Exception:
            logger.bind(component="preflight").exception("could not record the unheld alert")
        return True

    BROKER_BOOTSTRAP_RETRY_SECONDS: float = 60.0

    def _write_bootstrap_heartbeat(self, detail: str) -> None:
        """Mark the process alive while authenticated broker bootstrap waits.

        Docker's heartbeat healthcheck should distinguish a live process
        waiting for 2FA from a dead process.  The separate liveness artifact
        remains red, and the watchdog reports it; this heartbeat never claims
        that the broker itself is usable.
        """
        import json
        import tempfile

        path = settings.state_dir / "heartbeat.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "source": "broker_bootstrap",
            "status": "broker_unavailable",
            "detail": detail[:240],
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    async def _report_live_bootstrap_failure(
        self,
        detail: str,
        *,
        probe: str,
        already_recorded: bool = False,
    ) -> None:
        """Persist and alert a no-orders bootstrap failure, then back off."""
        if not already_recorded:
            try:
                from trading.runtime.broker_liveness import record_broker_liveness_failure

                await asyncio.to_thread(
                    record_broker_liveness_failure,
                    settings.state_dir,
                    detail,
                    probe=probe,
                )
            except Exception:
                logger.bind(component="broker_liveness").exception(
                    "could not record live bootstrap failure"
                )
        try:
            await asyncio.to_thread(self._write_bootstrap_heartbeat, detail)
        except Exception:
            logger.bind(component="runner").exception("could not write bootstrap heartbeat")
        try:
            from trading.runtime.ops_watch import run_ops_watch

            await asyncio.to_thread(
                run_ops_watch,
                settings.state_dir,
                log_dir=settings.log_dir,
            )
        except Exception:
            logger.bind(component="ops_watch").exception("bootstrap ops-watch pass failed")
        logger.bind(component="runner").warning(
            f"live broker bootstrap waiting ({probe}): {detail[:240]}"
        )
        await asyncio.sleep(self.BROKER_BOOTSTRAP_RETRY_SECONDS)

    async def _bootstrap_live_broker(self) -> None:
        """Connect, prove authentication, and verify positions before jobs exist.

        A lost IBKR login must not merely look like a stale account cache.
        Until all three checks succeed, this loop does not construct or start
        APScheduler: no cycle, flag consumer, queued command, guard, or
        recovery path can submit an order.  It deliberately relies on the
        IBKR adapter's non-recovering liveness and strict-position seams, so
        this observer never invokes automatic gateway restarts; deliberate
        bounded connection retries remain visible through the liveness file.
        """
        from trading.runtime.broker_liveness import record_broker_liveness

        while True:
            try:
                await asyncio.to_thread(self.broker.connect)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._report_live_bootstrap_failure(
                    f"{type(exc).__name__}: {exc}", probe="connect"
                )
                continue

            try:
                observation = await asyncio.to_thread(
                    record_broker_liveness, self.broker, settings.state_dir
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._report_live_bootstrap_failure(
                    f"{type(exc).__name__}: {exc}", probe="reqCurrentTime"
                )
                continue
            if not isinstance(observation, dict) or observation.get("ready") is not True:
                detail = (
                    str(observation.get("detail", "authenticated liveness probe unavailable"))
                    if isinstance(observation, dict)
                    else "authenticated liveness probe unsupported"
                )
                probe = (
                    str(observation.get("probe", "reqCurrentTime"))
                    if observation
                    else "reqCurrentTime"
                )
                await self._report_live_bootstrap_failure(
                    detail,
                    probe=probe,
                    already_recorded=isinstance(observation, dict),
                )
                continue

            if self.preflight_unheld(require_reachable=True):
                return
            await self._report_live_bootstrap_failure(
                "position inventory unavailable after an authenticated liveness probe",
                probe="positions",
            )

    async def run_forever(self) -> None:
        """Start APScheduler and block until SIGINT / SIGTERM."""
        if settings.is_live_armed():
            await self._bootstrap_live_broker()

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

        # Manual-command watcher. The Telegram bot writes JSON commands
        # (BUY / SELL / FLATTEN / FX_CONVERT / CANCEL_ORDER / ...) into
        # state/commands/pending/. This watcher executes them via the
        # broker on a single thread so they never race the cycle.
        self._scheduler.add_job(
            self._process_pending_commands,
            IntervalTrigger(seconds=5),
            id="command_processor",
            replace_existing=True,
            max_instances=1,
        )

        # Heartbeat watchdog: every 6h, check that we've had a successful
        # cycle in the last HEARTBEAT_WATCHDOG_HOURS. Sends a Telegram
        # nudge if we haven't. Does NOT halt — that's the operator's call.
        self._scheduler.add_job(
            self._watchdog,
            IntervalTrigger(hours=6),
            id="watchdog",
            replace_existing=True,
        )

        # Live snapshot refresh: every 60s, pull a fresh broker account
        # snapshot and persist it. Without this the Telegram /balances and
        # /positions commands read stale data — only as fresh as the last
        # successful cycle, which can be many hours ago between Friday
        # rebalances. ib-async's wrapper keeps account/position dicts
        # push-updated server-side, so get_account is a cache read and
        # cheap to repeat at 1Hz. max_instances=1 + coalesce skips ticks
        # while a previous refresh is still in flight (e.g. during a
        # broker reconnect).
        self._scheduler.add_job(
            self._refresh_account_snapshot,
            IntervalTrigger(seconds=60),
            id="snapshot_refresh",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
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
            # HMM regime advisor: once daily after the cache is refreshed.
            # Slow signal — daily granularity is enough. Complements the
            # hourly SMA/VIX advisor without spamming.
            self._scheduler.add_job(
                self._run_hmm_advisor_async,
                CronTrigger(hour=18, minute=15, timezone="America/New_York"),
                id="hmm_advisor",
                replace_existing=True,
            )
            # Options-structure monitor: twice per US session (post-open,
            # pre-close). Watches SPY's IV level/skew/term slope and
            # put-call OI for stress signatures the spot-only advisors
            # can't see. Advisory only — debounced like the others.
            self._scheduler.add_job(
                self._run_options_monitor_async,
                CronTrigger(hour="15,19", minute=45, timezone="UTC"),
                id="options_monitor",
                replace_existing=True,
            )
            # Agent committee: TWICE weekly — Mon & Fri, mid-session NYSE time
            # (default 13:00 ET: prices settled, well clear of the noisy open).
            # NYSE tz so it tracks the US session across DST. Env-tunable via
            # AGENTS_COMMITTEE_CRON. Advisory only; requires AGENTS_ENABLED=true
            # + an LLM API key in .env.
            import os as _os

            if _os.getenv("AGENTS_ENABLED", "false").lower() in ("true", "1", "yes") and (
                _os.getenv("ANTHROPIC_API_KEY") or _os.getenv("OPENAI_API_KEY")
            ):
                self._scheduler.add_job(
                    self._run_committee_async,
                    CronTrigger.from_crontab(
                        _os.getenv("AGENTS_COMMITTEE_CRON", "0 13 * * MON,FRI"),
                        timezone="America/New_York",
                    ),
                    id="agent_committee",
                    replace_existing=True,
                )
                # Economy watch: slow FRED series (CPI, claims, HY OAS...).
                # Weekdays 11:00 UTC — well before the committee.
                self._scheduler.add_job(
                    self._run_econ_watch_async,
                    CronTrigger(day_of_week="mon-fri", hour=11, minute=0, timezone="UTC"),
                    id="econ_watch",
                    replace_existing=True,
                )
                # News watch: feeds the daily scout dashboard. 13:40 UTC
                # weekdays — fresh headlines + sector momentum. The on-demand
                # /committee path refreshes this again before it debates.
                # Pure RSS/yfinance; failures degrade, not break.
                self._scheduler.add_job(
                    self._run_news_watch_async,
                    CronTrigger(day_of_week="mon-fri", hour=13, minute=40, timezone="UTC"),
                    id="news_watch",
                    replace_existing=True,
                )
                # On-demand convening via /committee (flag file, 30s poll).
                self._scheduler.add_job(
                    self._check_committee_flag,
                    IntervalTrigger(seconds=30),
                    id="committee_trigger",
                    replace_existing=True,
                    max_instances=1,
                )
                # Agent PM: the committee's trading arm.
                #
                # Ran Mondays 14:30 UTC while it was simulation-only, when
                # nothing depended on WHEN it decided. With the execution
                # bridge (AGENT_PM_SLEEVE_PCT > 0) that stops being true:
                # a Monday decision executed by Friday's cycle is a
                # four-day-old view applied to a tape that has moved, and
                # the bridge's freshness guard would refuse it every single
                # week. A bridge that declines every week is worse than no
                # bridge — it looks like a PM that keeps choosing to hold.
                #
                # So derive the PM run from the cycle's own cron, exactly
                # as the broker-readiness check does, and put it 45 minutes
                # ahead: long enough for the committee round trip, short
                # enough to stay inside the 6h freshness window. Change
                # CRON and both move together.
                #
                # Falls back to the Monday slot when the cron shape cannot
                # be offset (crosses midnight, or is not a simple daily
                # time) — simulation keeps running; the bridge will refuse
                # on freshness and say so, which is the safe direction.
                _pm_trigger = _precycle_trigger(
                    self.config.schedule_cron,
                    self.config.schedule_tz,
                    lead_minutes=settings.pm_pre_cycle_lead_minutes,
                ) or CronTrigger(day_of_week="mon", hour=14, minute=30, timezone="UTC")
                self._scheduler.add_job(
                    self._run_agent_pm_async,
                    _pm_trigger,
                    id="agent_pm",
                    replace_existing=True,
                )
                self._scheduler.add_job(
                    self._check_agent_pm_flag,
                    IntervalTrigger(seconds=30),
                    id="agent_pm_trigger",
                    replace_existing=True,
                    max_instances=1,
                )
                # Daily PM mark-to-market: weekdays 17:15 New York, after
                # the US close. No LLM — one price fetch so the simulated
                # sleeve has a daily equity curve and SPY benchmark.
                self._scheduler.add_job(
                    self._mark_agent_pm_async,
                    CronTrigger(
                        day_of_week="mon-fri", hour=17, minute=15, timezone="America/New_York"
                    ),
                    id="agent_pm_mark",
                    replace_existing=True,
                )
                # Sentinel: intraday tripwires every 15 min during US RTH.
                # 13:30-20:00 UTC covers 9:30-16:00 ET in summer (shifts an
                # hour in winter — acceptable for a tripwire). Mechanical
                # checks are free; the LLM runs only when a wire trips.
                # INFORMATION ONLY — it alerts, it never convenes the committee.
                self._scheduler.add_job(
                    self._run_sentinel_async,
                    CronTrigger(day_of_week="mon-fri", hour="13-20", minute="*/15", timezone="UTC"),
                    id="sentinel",
                    replace_existing=True,
                    max_instances=1,
                )
                # Late-day de-risk: ONE check ~50 min before the close (default
                # 15:10 ET). If a holding is down >= SENTINEL_DERISK_DROP_PCT on
                # the day by then, convene the committee once. The noisy open is
                # excluded by design; nothing fires after the close. NYSE tz.
                self._scheduler.add_job(
                    self._run_lateday_derisk_async,
                    CronTrigger.from_crontab(
                        _os.getenv("SENTINEL_DERISK_CRON", "10 15 * * MON-FRI"),
                        timezone="America/New_York",
                    ),
                    id="committee_lateday",
                    replace_existing=True,
                    max_instances=1,
                )
            # Daily P&L note: just after the US close, in the same NYSE
            # wall-clock time across DST. One read of
            # runner.db + one Telegram message — negligible load.
            self._scheduler.add_job(
                self._run_daily_summary_async,
                CronTrigger(day_of_week="mon-fri", hour=16, minute=10, timezone="America/New_York"),
                id="daily_summary",
                replace_existing=True,
            )
            # Macro financial-conditions monitor: daily 13:30 UTC
            # (pre-US-open, after Europe has priced overnight macro).
            # Rates/dollar/energy/BTC z-score dial from the 2018-2026
            # lead-lag study (docs/research_macro_leadlag.md). Advisory.
            self._scheduler.add_job(
                self._run_macro_monitor_async,
                CronTrigger(hour=13, minute=30, timezone="UTC"),
                id="macro_monitor",
                replace_existing=True,
            )
            # Style-rotation advisor: weekly, Sunday 12:00 UTC (market
            # closed, cache warm from Friday's cycle). Ranks all
            # registered strategies on trailing 3/6/9-month Sharpe and
            # proposes a switch when the leader changes. NEVER applies
            # anything — the deployed strategy only changes via .env.
            self._scheduler.add_job(
                self._run_style_advisor_async,
                CronTrigger(day_of_week="sun", hour=12, minute=0, timezone="UTC"),
                id="style_advisor",
                replace_existing=True,
            )

        self._register_core_background_jobs()

        self._scheduler.start()
        try:
            # Do not spend the five-minute NYSE-open baseline window waiting
            # for APScheduler's first 60-second interval tick after a
            # restart.  This is read-only broker observation plus local state
            # persistence; no cycle or order path is started here.
            await self._refresh_account_snapshot()
            self.alerts.info(self._format_runner_started_message())
            logger.bind(component="runner").info(
                f"scheduler started — next run: {self._scheduler.get_job('cycle').next_run_time}"
            )
            # Inventory every registered job. Several are conditional on env
            # flags or on parsing the cron, and until now the only way to know
            # whether one had actually armed was to wait and see whether it
            # ever fired. One line at startup answers it instead.
            for _job in sorted(self._scheduler.get_jobs(), key=lambda j: j.id):
                logger.bind(component="runner").info(
                    f"  job {_job.id:<18} next={_job.next_run_time}"
                )

            # APScheduler does not replay missed cron slots on restart.  The
            # market-watch collector is safe to replay (same-day writes replace
            # rather than append), so a restart after its post-close slot should
            # repair the dashboard's missing reading immediately.
            self._start_startup_market_watch_catchup()

            # Startup reconciliation. Today (May 2026) we shipped a bug where
            # broker.get_account silently returned a zero-position snapshot on
            # IBKR timeout, and three cycles stacked to 3x target. The cycle
            # itself now fails-closed on that path, but a sibling failure
            # mode is: container restarts mid-cycle, local order_store is
            # empty, broker still holds positions, next cycle thinks book is
            # flat and re-buys. We can't *fix* that automatically (the
            # safest thing is to make the operator notice + decide), but we
            # CAN make the drift loud at startup so they intervene.
            try:
                await asyncio.to_thread(self._reconcile_startup)
            except Exception:
                logger.bind(component="runner").exception("startup reconciliation failed")

            # Park until cancelled.
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError):  # Windows / restricted env
                    loop.add_signal_handler(sig, stop_event.set)
            await stop_event.wait()
        finally:
            await self._shutdown()

    # Hard upper bound on cycle duration. If the cycle is still running
    # past this, we abort and notify the operator. Generous enough for
    # a cold data refresh + ~10 IBKR API calls; tight enough that the
    # operator hears about a wedged gateway within minutes, not hours.
    CYCLE_TIMEOUT_SECONDS: float = 300.0  # 5 minutes

    # Minimum gap between consecutive cycle starts (audit fix #11). With
    # both a cron schedule AND an off-cycle trigger watcher polling every
    # 30s, in pathological cases a cron firing could overlap with an
    # operator-triggered cycle. The risk manager re-evaluates the same
    # signal in both, producing duplicate orders. This cooldown refuses
    # the second cycle, logs, and leaves the first to complete.
    CYCLE_COOLDOWN_SECONDS: float = 10.0

    def _load_error_counter(self) -> int:
        """Read the persisted consecutive-error count, default to 0 on any
        failure. Persistence keeps the auto-halt threshold honest across
        container restarts (audit fix #8)."""
        try:
            if self._error_counter_path.exists():
                import json as _json

                return int(_json.loads(self._error_counter_path.read_text()).get("count", 0))
        except Exception:
            logger.bind(component="runner").exception(
                "consecutive_errors.json unreadable; defaulting to 0"
            )
        return 0

    def _save_error_counter(self) -> None:
        """Atomically persist the counter. Best-effort: a failed write
        logs but doesn't crash the runner (the cycle already errored
        once, we don't want to compound)."""
        try:
            import json as _json
            import os as _os
            import tempfile as _tmp

            self._error_counter_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = _tmp.mkstemp(
                dir=self._error_counter_path.parent,
                prefix=f"{self._error_counter_path.name}.",
            )
            with _os.fdopen(fd, "w") as f:
                _json.dump({"count": self._consecutive_errors}, f)
            _os.replace(tmp, self._error_counter_path)
        except Exception:
            logger.bind(component="runner").exception("failed to persist error counter")

    def _reconcile_startup(self) -> None:
        """At startup, compare the broker's positions to the last persisted
        snapshot. If they differ, alert the operator loudly — they may need
        to manually flatten or sell down before the next cycle.

        Why this matters: the cycle uses broker.get_account() each cycle,
        but if the operator just restarted the container right after a
        partial-fill run, the local snapshot might be stale and the
        broker could be holding positions that were never persisted. We
        don't auto-rebalance because the safe action depends on intent
        — *operator must decide*.
        """
        try:
            broker_positions = self.broker.get_positions()
        except Exception as e:
            logger.bind(component="runner").warning(
                f"startup reconciliation: broker.get_positions failed ({e!r}); skipping drift check"
            )
            return

        snap = None
        with contextlib.suppress(Exception):
            snap = self.cycle.runner_store.latest_snapshot()
        snap_positions = list(snap.positions.values()) if snap else []

        # Build a key -> quantity mapping for both sides.
        broker_map = {p.instrument.key: float(p.quantity) for p in broker_positions}
        snap_map = {p.instrument.key: float(p.quantity) for p in snap_positions}

        drifted: list[str] = []
        all_keys = set(broker_map) | set(snap_map)
        for k in all_keys:
            b = broker_map.get(k, 0.0)
            s = snap_map.get(k, 0.0)
            if abs(b - s) < 1e-6:
                continue
            sym = k.split(":", 1)[1] if ":" in k else k
            drifted.append(f"{sym}: broker={b:g}, snapshot={s:g}")

        if not drifted:
            self.alerts.info(
                f"✅ startup reconciliation: broker matches last snapshot "
                f"({len(broker_positions)} position(s))"
            )
            return

        snap_age = "(no prior snapshot)"
        if snap is not None:
            # Both sides in UTC. `datetime.now()` is naive LOCAL time, so
            # stripping the tz off a UTC snapshot compared wall clocks in
            # two zones — on a CEST machine this reported the snapshot as
            # two hours off, in the one alert that asks a human to decide
            # whether the broker and our state have diverged.
            age = (datetime.now(tz=timezone.utc) - snap.ts).total_seconds()
            snap_age = f"(snapshot is {age / 60:.0f} min old)"
        body = "\n".join(f"  • {line}" for line in drifted)
        self.alerts.critical(
            "⚠️ startup reconciliation: BROKER POSITIONS DIFFER FROM SNAPSHOT "
            f"{snap_age}\n{body}\n"
            "→ review with /positions and either /flatten or accept the broker state."
        )
        logger.bind(component="runner").warning(
            f"startup drift: {len(drifted)} symbol(s) — {drifted}"
        )

    def _format_runner_started_message(self) -> str:
        """Build the human-readable startup alert.

        Expands the internal strategy slug + cron expression into something
        the operator can scan on Telegram without thinking — no Python
        ``['x']`` reprs, no raw cron strings.
        """
        cfg = self.config
        parts: list[str] = []
        for slug in cfg.strategies:
            params = cfg.strategy_params.get(slug, {}) if cfg.strategy_params else {}
            parts.append(_humanize_strategy(slug, params))
        strat_line = "; ".join(parts) if parts else "(none)"

        try:
            next_run = self._scheduler.get_job("cycle").next_run_time
            next_run_s = next_run.strftime("%Y-%m-%d %H:%M %Z") if next_run else "?"
        except Exception:
            next_run_s = "?"

        lines = [
            "🤖 Runner online",
            f"  Strategy:    {strat_line}",
            f"  Universe:    {cfg.universe.upper()}",
            f"  Rebalance:   {_humanize_cron(cfg.schedule_cron)}",
            f"  Next run:    {next_run_s}",
        ]
        if cfg.vol_target is not None:
            lines.append(
                f"  Vol target:  {cfg.vol_target:.0%} annualized "
                f"(max leverage {cfg.max_leverage:g}x)"
            )
        return "\n".join(lines)

    async def _run_cycle_async(self, *, review_only: bool = False) -> None:
        """Run one trading cycle with a hard timeout + Telegram-friendly
        error reporting.

        APScheduler executes coroutine jobs natively; we run the synchronous
        cycle in a worker thread so the event loop stays responsive (the
        trigger watcher, the HMM advisor, etc. continue to fire).

        ``review_only`` follows the same read/sizing path but calls the
        Cycle's non-executable review method. It never submits orders or
        creates an approval window; the flag is used when the bot sees an
        active halt.

        If the cycle exceeds ``CYCLE_TIMEOUT_SECONDS`` — almost always
        because the IBKR gateway has a dead broker session and an API call
        is wedged — we abort with a clear error message and Telegram alert.
        The worker thread continues running in the background; it'll wind
        down on its own when its ib-async call eventually times out
        internally. Crucially the runner is unblocked and ready for the
        next scheduled cycle.
        """
        # Cycle cooldown gate. Audit fix #11: refuse a cycle started within
        # CYCLE_COOLDOWN_SECONDS of the previous one — protects against
        # cron + off-cycle trigger near-simultaneous fires.
        now = datetime.now()
        if self._last_cycle_start_ts is not None:
            gap = (now - self._last_cycle_start_ts).total_seconds()
            if gap < self.CYCLE_COOLDOWN_SECONDS:
                logger.bind(component="runner").warning(
                    f"cycle suppressed by cooldown ({gap:.1f}s < "
                    f"{self.CYCLE_COOLDOWN_SECONDS:.0f}s since last start)"
                )
                return
        self._last_cycle_start_ts = now

        # Re-read the consecutive-error counter from disk. The Telegram bot's
        # /resume writes a fresh zero into consecutive_errors.json — without
        # this reload, the runner's in-memory counter stays at its old high
        # value and the next single failure re-triggers auto-halt.
        disk_count = self._load_error_counter()
        if disk_count != self._consecutive_errors:
            self._consecutive_errors = disk_count

        try:
            cycle_fn = self.cycle.run_review if review_only else self.cycle.run_cycle
            report = await asyncio.wait_for(
                asyncio.to_thread(cycle_fn),
                timeout=self.CYCLE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._consecutive_errors += 1
            self._save_error_counter()
            msg = (
                f"⏱️ cycle aborted after {self.CYCLE_TIMEOUT_SECONDS:.0f}s "
                f"(error #{self._consecutive_errors}/{self.AUTO_HALT_AFTER}) — "
                "likely a wedged IBKR Gateway."
            )
            logger.bind(component="runner").error(msg)
            self.alerts.critical(msg)
            self._maybe_auto_halt("cycle timeout")
            return
        except Exception as e:
            self._consecutive_errors += 1
            self._save_error_counter()
            msg = (
                f"❌ cycle crashed: {type(e).__name__}: {e} "
                f"(error #{self._consecutive_errors}/{self.AUTO_HALT_AFTER})"
            )
            logger.bind(component="runner").exception("cycle crashed")
            self.alerts.critical(msg)
            self._maybe_auto_halt(f"cycle crash: {type(e).__name__}")
            return

        if report.status == "error":
            self._consecutive_errors += 1
            self._save_error_counter()
            err = (report.error or "unknown error").strip()
            logger.bind(component="runner").error(f"cycle error: {err}")
            self.alerts.error(
                f"❌ cycle error ({self._consecutive_errors}/{self.AUTO_HALT_AFTER}): {err[:300]}"
            )
            self._maybe_auto_halt(f"cycle error: {err[:100]}")
        elif report.status == "halted":
            logger.bind(component="runner").warning("cycle halted by risk manager")
            self.alerts.warning("⚠️ cycle halted by risk manager")
        elif report.status == "halted_review":
            # The cycle itself has emitted the complete review card.  Do not
            # append the old generic warning or call this a recovery: the
            # account is intentionally still halted and no order was sent.
            logger.bind(component="runner").info("halted live-account review completed")
        else:
            # Success — reset the consecutive error counter (and persisted file).
            if self._consecutive_errors > 0:
                self.alerts.info(f"✅ cycle recovered (after {self._consecutive_errors} errors)")
            self._consecutive_errors = 0
            self._save_error_counter()
            self._last_success_ts = datetime.now()
            with contextlib.suppress(Exception):
                from trading.memory.store import default_store

                default_store().journal(
                    "cycle",
                    {
                        "status": report.status,
                        "orders": report.orders_submitted,
                        "fills": report.fills_received,
                    },
                )

    # Tracked by _run_cycle_async to enable "auto-halt after N consecutive
    # failures" and the heartbeat watchdog. The runner is the only writer.
    AUTO_HALT_AFTER: int = 3
    HEARTBEAT_WATCHDOG_HOURS: float = 25.0  # 1h grace past 24h cron

    def _maybe_auto_halt(self, reason: str) -> None:
        """If we've crossed the consecutive-error threshold, drop a halt
        file ourselves and tell the operator loudly. They have to /resume
        to re-arm; we never auto-recover."""
        if self._consecutive_errors < self.AUTO_HALT_AFTER:
            return
        try:
            # Was a bare four-key overwrite with a NAIVE datetime.now().
            # Two bugs in three lines: it erased the kill-switch baselines
            # (so an auto-halt reset the drawdown peak — after a run of
            # failures, exactly when you want it intact), and it stamped a
            # timezone-less timestamp that nobody could interpret from a
            # UTC-scheduled container. set_halted does both correctly.
            from trading.risk.halt_file import set_halted

            settings.state_dir.mkdir(parents=True, exist_ok=True)
            set_halted(
                settings.state_dir,
                halted=True,
                reason=(
                    f"auto-halt after {self._consecutive_errors} consecutive failures: {reason}"
                ),
            )
        except Exception:
            logger.bind(component="runner").exception("auto-halt write failed")
        self.alerts.critical(
            f"🛑 *AUTO-HALT* — {self._consecutive_errors} cycle failures in a row\n"
            f"Last reason: `{reason[:200]}`\n\n"
            "*Next step:* investigate the failure, then `/resume` to re-arm.\n"
            "`/resume` also resets the failure counter, so a single fresh "
            "failure won't immediately re-halt."
        )

    async def _refresh_account_snapshot(self) -> None:
        """Pull a fresh account snapshot from the broker and persist it.

        Runs on a 60s interval so the Telegram bot's /balances and
        /positions always see near-live data without waiting for the
        next cycle. ib-async keeps the underlying account/position
        dicts push-updated by IBKR, so get_account is a cheap cache
        read. Failures are logged at debug level only — a transient
        broker hiccup shouldn't generate operator noise; the next
        cycle's hard-fail path will alert.

        Also folds in per-currency cash via get_balances() when the
        broker supports it, so /balances can render the CHF/USD/EUR
        split without doing a second live query from the bot process.

        Guards against overwriting a good snapshot with placeholder
        data when the broker is mid-wedge. Wedged sessions can return
        an empty accountSummary list, which our adapter renders as
        equity=0 / cash=0 / base_currency="USD" — strictly worse than
        the previous snapshot (which at least had real numbers from
        the last working call). Snapshot-refresh ALSO touches
        heartbeat.json so /status reflects "broker alive" between
        cycles, not just at end-of-cycle.

        Before reading that cached account state, an IBKR broker also gets a
        bounded, read-only ``reqCurrentTime`` probe. It is recorded in a
        separate artifact because a fresh account subscription and open TCP
        socket do not prove Gateway still has an authenticated IBKR session.
        The probe never reconnects or restarts Gateway; it makes a 2FA/login
        outage visible to the ops watchdog instead of hiding it.
        """
        liveness: dict[str, object] | None = None
        try:
            from trading.runtime.broker_liveness import record_broker_liveness

            observed = await asyncio.to_thread(
                record_broker_liveness,
                self.broker,
                settings.state_dir,
            )
            if isinstance(observed, dict):
                liveness = observed
        except Exception:
            logger.bind(component="broker_liveness").exception("broker API liveness record failed")

        try:
            snap = self.broker.get_account()
        except Exception as e:
            logger.bind(component="runner").debug(
                f"snapshot refresh skipped: {type(e).__name__}: {e!r}"
            )
            return

        # Defensive: drop obviously-empty snapshots. equity == 0 is the
        # giveaway — even a freshly-opened paper account has the funding
        # cash showing as both cash and equity. A zero here means the
        # broker returned no accountSummary rows, almost always because
        # of a wedged subscription state.
        if snap.equity == 0 and snap.cash == 0 and not snap.positions:
            logger.bind(component="runner").debug(
                "snapshot refresh produced empty data — not saving over previous snapshot"
            )
            return

        if hasattr(self.broker, "get_balances"):
            try:
                per_ccy = self.broker.get_balances() or {}  # type: ignore[attr-defined]
            except Exception as e:
                per_ccy = {}
                logger.bind(component="runner").debug(
                    f"get_balances failed during refresh: {type(e).__name__}: {e!r}"
                )
            if per_ccy:
                snap = snap.model_copy(update={"cash_by_currency": per_ccy})

        self._monitor_live_account_risk(snap, liveness=liveness)
        self.cycle.runner_store.save_snapshot(snap)

        # Touch heartbeat. A successful snapshot refresh proves the trader
        # is alive AND the broker is talking back — operationally a better
        # liveness signal than "last cycle completed", which between
        # weekly rebalances always reads 6+ days stale.
        try:
            hb_path = settings.state_dir / "heartbeat.json"
            hb_path.parent.mkdir(parents=True, exist_ok=True)
            hb_path.write_text(
                '{"ts": "'
                + datetime.now(tz=timezone.utc).isoformat()
                + '", "source": "snapshot_refresh"}'
            )
        except Exception as e:
            logger.bind(component="runner").debug(
                f"heartbeat touch failed: {type(e).__name__}: {e!r}"
            )

    def _monitor_live_account_risk(
        self,
        snapshot: AccountSnapshot,
        *,
        liveness: dict[str, object] | None,
        observed_at: datetime | None = None,
    ) -> None:
        """Continuously enforce live loss/drawdown safety from fresh snapshots.

        A cycle can be weekly and a manual /cycle can arrive hours after a
        loss threshold is crossed.  This observer runs every snapshot tick,
        but is deliberately conservative: it never invents a daily baseline
        after the five-minute verified NYSE-open window, and it skips a
        cached snapshot when the authenticated IBKR probe is red.
        """
        if not settings.is_live_armed():
            return
        if liveness is None:
            logger.bind(component="risk_monitor").warning(
                "skipping live risk observation because authenticated broker liveness is unavailable"
            )
            return
        if liveness is not None and liveness.get("ready") is not True:
            logger.bind(component="risk_monitor").warning(
                "skipping live risk observation while broker liveness is not ready"
            )
            return

        risk_manager = getattr(self.cycle, "risk_manager", None)
        if risk_manager is None:
            logger.bind(component="risk_monitor").error(
                "live snapshot has no risk manager; refusing to claim risk monitoring"
            )
            return

        try:
            from trading.runtime.nyse_session import (
                current_nyse_session,
                is_opening_capture_window,
            )

            observed = observed_at or datetime.now(tz=timezone.utc)
            session = current_nyse_session(observed)
            if session is not None and is_opening_capture_window(observed):
                note = risk_manager.capture_session_open(
                    snapshot,
                    session_date=session.label,
                    captured_at=observed,
                    source="snapshot_refresh",
                )
                if note:
                    self.alerts.critical(f"⚠️ Risk baseline repaired — {note}")

            risk_manager._reload_halt_state()
            was_halted = risk_manager.is_halted()
            decision = risk_manager.evaluate_session_risk(
                snapshot,
                session_label=session.label if session is not None else None,
            )
        except Exception:
            # An observer bug must never be mistaken for a healthy account;
            # the hard execution gate will independently repeat the strict
            # check inside Cycle.  Log loudly for ops without causing a
            # snapshot persistence outage.
            logger.bind(component="risk_monitor").exception("live risk observation failed")
            return

        if decision.action == "halt":
            self._last_risk_monitor_reject_reason = None
            if not was_halted:
                currency = str(snapshot.base_currency or "USD").upper()
                self.alerts.critical(
                    "🚨 *RISK HALT — continuous account monitor*\n"
                    f"{decision.reason}\n"
                    f"Snapshot: {currency} {snapshot.equity:,.2f} equity.\n"
                    "No new exposure can be submitted; verified /close or /flatten "
                    "remains available to reduce risk."
                )
            return

        if decision.action == "reject":
            if decision.reason != getattr(self, "_last_risk_monitor_reject_reason", None):
                self._last_risk_monitor_reject_reason = decision.reason
                self.alerts.warning(
                    "⚠️ *Live execution safety gate*\n"
                    f"{decision.reason}.\n"
                    "A real-account review remains available, but no executable "
                    "cycle may run until a trusted NYSE-open baseline is captured."
                )
            return

        self._last_risk_monitor_reject_reason = None

    async def _watchdog(self) -> None:
        """Daily: if we haven't completed a successful cycle in
        ``HEARTBEAT_WATCHDOG_HOURS``, alert the operator. Not a halt —
        just a nudge. The runner could be stuck without ever raising,
        which silent-mode would hide."""
        try:
            hb_path = settings.state_dir / "heartbeat.json"
            if not hb_path.exists():
                if self._last_success_ts is None:
                    # Bootstrapping — no heartbeat yet; ignore for now.
                    return
                age_s = (datetime.now() - self._last_success_ts).total_seconds()
            else:
                age_s = datetime.now().timestamp() - hb_path.stat().st_mtime
            if age_s > self.HEARTBEAT_WATCHDOG_HOURS * 3600.0:
                self.alerts.warning(
                    f"⏰ Watchdog: no successful cycle in {age_s / 3600:.1f}h. "
                    "Check `/health` and broker connection."
                )
        except Exception:
            logger.bind(component="runner").exception("watchdog poll failed")

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

    async def _run_options_monitor_async(self) -> None:
        """Twice-daily: poll SPY's option-chain structure (ATM IV, put
        skew, term slope, put/call OI) and alert on new stress triggers.
        Advisory only; any failure is logged and swallowed."""
        try:
            from trading.runtime.options_monitor import poll_and_alert

            await poll_and_alert()
        except Exception:
            logger.bind(component="options_monitor").exception("options monitor poll failed")

    async def _run_market_watch_async(self) -> None:
        """Daily macro instrument panel refresh. Failures are swallowed.

        A startup catch-up can race the regular cron job by a few seconds;
        skip the duplicate rather than launching a second network batch.
        """
        lock = getattr(self, "_market_watch_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._market_watch_lock = lock
        if lock.locked():
            logger.bind(component="market_watch").info("market watch already in flight; skipping")
            return
        async with lock:
            try:
                from trading.runtime.market_watch import collect

                await asyncio.to_thread(collect, settings.state_dir, settings.data_dir)
            except Exception:
                logger.bind(component="market_watch").exception("market watch failed")

    def _start_startup_market_watch_catchup(self) -> None:
        """Start a missed post-close collection without delaying trading."""
        from trading.runtime.market_watch import needs_startup_catchup

        if not needs_startup_catchup(settings.state_dir):
            return
        logger.bind(component="market_watch").info(
            "runner started after market-watch slot; collecting catch-up reading"
        )
        self._startup_market_watch_task = asyncio.create_task(
            self._run_market_watch_async(), name="market-watch-startup-catchup"
        )

    async def _run_news_watch_async(self) -> None:
        """Collect headlines + sector momentum for the scout. Advisory."""
        try:
            from trading.runtime.news_watch import collect

            await asyncio.to_thread(collect, settings.state_dir)
        except Exception:
            logger.bind(component="news_watch").exception("news watch failed")

    async def _run_econ_watch_async(self) -> None:
        """Collect FRED macro series for the Economy tab + agent context."""
        try:
            from trading.runtime.econ_watch import collect

            await asyncio.to_thread(collect, settings.state_dir)
        except Exception:
            logger.bind(component="econ_watch").exception("econ watch failed")

    async def _check_committee_flag(self) -> None:
        """Operator asked for a fresh debate via /committee: the bot drops
        state/committee_now.flag; we consume it and convene immediately."""
        flag = settings.state_dir / "committee_now.flag"
        if not flag.exists():
            return
        with contextlib.suppress(Exception):
            flag.unlink()
        logger.bind(component="agents").info("on-demand committee triggered")
        # On-demand runs deserve fresh gossip too — cheap, so just refresh.
        await self._run_news_watch_async()
        await self._run_committee_async()

    async def _check_agent_pm_flag(self) -> None:
        """Operator asked the PM to rebalance now via /pm run."""
        flag = settings.state_dir / "agent_pm_now.flag"
        if not flag.exists():
            return
        with contextlib.suppress(Exception):
            flag.unlink()
        logger.bind(component="agent_pm").info("on-demand agent PM triggered")
        await self._run_agent_pm_async()

    async def _run_committee_async(self) -> None:
        """Daily agent committee: gather context, run the debate, send
        the digest. Advisory only — writes to memory and Telegram, never
        to the order path. Failures are logged and swallowed."""
        try:
            from trading.agents.committee import format_digest_compact, run_committee
            from trading.agents.context import build_context
            from trading.memory.store import default_store

            mem = default_store()
            ctx = await asyncio.to_thread(build_context, settings.state_dir, settings.data_dir)
            digest = await asyncio.to_thread(run_committee, ctx, mem, calibration=mem.calibration())
            # Persist the full debate for /detail; send only the summary.
            import json as _json

            (settings.state_dir / "last_committee.json").write_text(
                _json.dumps(digest, default=str, indent=1)
            )
            self.alerts.info(format_digest_compact(digest))
        except Exception:
            logger.bind(component="agents").exception("committee run failed")

    async def _run_agent_pm_async(self) -> None:
        """Weekly agent PM — committee-driven SIMULATED sleeve. Reads a
        week of journaled rulings + calibration, makes one LLM call, and
        rebalances a virtual portfolio under state/agent_pm/. Never
        touches IBKR or the order path; failures logged and swallowed."""
        try:
            from trading.agents.context import build_context
            from trading.agents.pm import format_pm_digest, run_agent_pm
            from trading.memory.store import default_store

            mem = default_store()
            ctx = await asyncio.to_thread(build_context, settings.state_dir, settings.data_dir)
            result = await asyncio.to_thread(run_agent_pm, ctx, mem, settings.state_dir)
            # Pass the freshly-written book so the digest is self-contained:
            # one message showing what changed AND the resulting holdings
            # with units. Previously the book arrived separately, in share
            # counts, next to a digest quoting target weights.
            book = None
            with contextlib.suppress(Exception):
                import json as _json

                book = _json.loads((settings.state_dir / "agent_pm" / "portfolio.json").read_text())
            # Hand the digest the live account so its weights arrive with
            # an approximate CHF size attached. Without this the card is
            # still true, it just says less — which is why the lookup is
            # allowed to fail quietly.
            account: dict[str, Any] | None = None
            try:
                from trading.runner.state import RunnerStore

                snap = RunnerStore(settings.state_dir / "runner.db").latest_snapshot()
                if snap is not None and float(getattr(snap, "equity", 0.0) or 0.0) > 0:
                    account = {
                        "equity": float(snap.equity),
                        "currency": str(getattr(snap, "base_currency", "") or "USD"),
                        "sleeve_pct": float(settings.agent_pm_sleeve_pct or 0.0),
                    }
            except Exception:
                logger.bind(component="agent_pm").warning(
                    "no account snapshot for the PM digest translation"
                )
            self.alerts.info(format_pm_digest(result, book, account=account))
        except Exception:
            logger.bind(component="agent_pm").exception("agent PM run failed")

    async def _mark_agent_pm_async(self) -> None:
        """Daily equity mark for the simulated PM sleeve. Silent on success."""
        try:
            from trading.agents.pm import mark_to_market

            res = await asyncio.to_thread(mark_to_market, settings.state_dir)
            if not res.get("ok") and res.get("reason") != "no PM book yet":
                logger.bind(component="agent_pm").warning(f"mark failed: {res.get('reason')}")
        except Exception:
            logger.bind(component="agent_pm").exception("agent PM mark failed")

    async def _run_ops_watch_async(self) -> None:
        """Five-minute infra health check — silence means healthy."""
        try:
            from trading.runtime.ops_watch import run_ops_watch

            # log_dir passed explicitly: it opts the error-log scan in.
            # run_ops_watch does not reach for settings itself, so its
            # answer depends only on what it is handed.
            await asyncio.to_thread(
                lambda: run_ops_watch(settings.state_dir, log_dir=settings.log_dir)
            )
        except Exception:
            logger.bind(component="ops_watch").exception("ops watch failed")

    async def _run_guards_async(self) -> None:
        """Trailing-stop / ratchet pass. Exits go through the command
        pipeline exactly like an operator /close — halt + risk respected."""
        try:
            from trading.runner.holds import load_holds
            from trading.runner.state import RunnerStore
            from trading.runtime import commands as cmds
            from trading.runtime.guards import check_guards, last_prices

            def _pass() -> dict:
                snap = RunnerStore(settings.state_dir / "runner.db").latest_snapshot()
                if not snap or not snap.positions:
                    return {"exits": [], "alerts": []}

                def _mark(p: Any) -> float | None:
                    """The broker's own valuation per share.

                    ``avg_price + unrealized_pnl / quantity``. This is an
                    INDEPENDENT price source from the yfinance quote the
                    guards trail against, which is the whole point: a
                    trailing stop turns one number into a full-position
                    market sell, and that number comes from a free feed
                    that sometimes serves an unadjusted price across a
                    split. Two sources agreeing is cheap; being wrong is
                    not recoverable.
                    """
                    qty = float(p.quantity)
                    if abs(qty) < 1e-9:
                        return None
                    try:
                        return float(p.avg_price) + float(p.unrealized_pnl) / qty
                    except Exception:
                        return None

                positions = [
                    {
                        "symbol": p.instrument.symbol,
                        "qty": float(p.quantity),
                        "avg_price": float(p.avg_price),
                        "mark": _mark(p),
                    }
                    for p in snap.positions.values()
                ]
                px = last_prices([p["symbol"] for p in positions])
                return check_guards(
                    settings.state_dir,
                    settings.data_dir,
                    positions=positions,
                    prices=px,
                    equity=float(snap.equity),
                    holds=set(load_holds(settings.state_dir)),
                )

            result = await asyncio.to_thread(_pass)
            for exit_req in result["exits"]:
                cmd = cmds.Command.new(
                    cmds.CommandType.CLOSE,
                    args={"symbol": exit_req["symbol"]},
                    requested_by=f"guard:{exit_req['reason']}",
                )
                cmds.submit(cmd, settings.state_dir)
            # One roll-up replaces the per-name bubbles when several fire
            # together. Six separate messages arrive at the moment the
            # operator is least able to add them up by hand; the roll-up
            # leads with the numbers that decide the next move.
            rollup = result.get("rollup")
            exit_msgs = set(result.get("exit_alerts") or [])
            for msg in result["alerts"]:
                if rollup and msg in exit_msgs:
                    continue
                self.alerts.info(msg)
            if rollup:
                self.alerts.info(rollup)
        except Exception:
            logger.bind(component="guards").exception("guards run failed")

    async def _run_sentinel_async(self) -> None:
        """Intraday risk watch — INFORMATION ONLY. Sends a caution alert when a
        tripwire fires; it never convenes the committee (that path is the
        twice-weekly schedule, the late-day de-risk check, and /committee) and
        never touches the order path."""
        try:
            from trading.runtime.sentinel import format_sentinel_alert, run_sentinel

            result = await asyncio.to_thread(run_sentinel, settings.state_dir)
            if result.get("quiet"):
                return
            from trading.bot.keyboards import sentinel_keyboard

            # Read-only buttons only: "look closer" and "argue about it".
            # Trimming from a tap is deliberately not offered — see
            # bot/keyboards.py on what gets a button and what stays typed.
            self.alerts.info(format_sentinel_alert(result), buttons=sentinel_keyboard())
        except Exception:
            logger.bind(component="sentinel").exception("sentinel run failed")

    async def _run_lateday_derisk_async(self) -> None:
        """Late-day de-risk gate (~50 min before the close). If a holding has
        cratered on the day (>= DERISK_DROP_PCT), convene the committee once —
        the noisy open is excluded, and nothing fires after the close.
        Advisory only; never the order path."""
        try:
            from trading.runtime.sentinel import format_derisk_alert, run_late_day_derisk

            result = await asyncio.to_thread(run_late_day_derisk, settings.state_dir)
            if result.get("quiet"):
                return
            self.alerts.info(format_derisk_alert(result))
            if result.get("convene"):
                await self._run_committee_async()
        except Exception:
            logger.bind(component="sentinel").exception("late-day de-risk run failed")

    async def _run_historian_async(self) -> None:
        """Twice-weekly lesson distillation — see agents/historian.py."""
        try:
            from trading.agents.context import build_context
            from trading.agents.historian import format_historian_digest, run_historian
            from trading.memory.store import default_store, lesson_condition_fingerprint

            context = await asyncio.to_thread(build_context, settings.state_dir, settings.data_dir)
            conditions = lesson_condition_fingerprint(context)
            digest = await asyncio.to_thread(run_historian, default_store(), conditions=conditions)
            self.alerts.info(format_historian_digest(digest))
        except Exception:
            logger.bind(component="historian").exception("historian run failed")

    async def _run_memory_grader_async(self) -> None:
        """Nightly: grade due predictions using cached closes, and journal
        a daily heartbeat into permanent memory. Failures are swallowed —
        memory must never break trading."""
        try:
            from trading.memory.grading import grade_due_predictions
            from trading.memory.store import default_store

            mem = default_store()
            # The loop lives in memory/grading.py so it can be tested
            # without a Runner. It was untestable here, and that is where
            # it was broken for as long as it existed.
            counts = grade_due_predictions(mem, settings.data_dir)
            graded, skipped = counts["graded"], counts["skipped"]
            unpriced = list(counts.get("unpriced_subjects", []))
            awaiting_next_daily_bar = list(counts.get("awaiting_next_daily_bar_subjects", []))
            awaiting_next_daily_bar_ids = list(
                counts.get("awaiting_next_daily_bar_prediction_ids", [])
            )
            cache_behind = list(counts.get("cache_behind_subjects", []))
            grading_failed = list(counts.get("failed_subjects", []))
            shadow_legs = self._grade_shadow(mem)
            # Closed round-trips into the episodes table. Until 2026-08-06
            # nothing ever called add_episode, so lessons were promoted by
            # the historian voting on its own book rather than by contact
            # with realised P&L.
            episodes_written = 0
            try:
                from trading.memory.episodes import record_closed_episodes

                episodes_written = record_closed_episodes(
                    mem, settings.state_dir, settings.data_dir
                )
            except Exception:
                logger.bind(component="memory").exception("episode recording failed")
            snap = self.cycle.runner_store.latest_snapshot()
            mem.journal(
                "daily",
                {
                    "equity": getattr(snap, "equity", None) if snap else None,
                    "positions": len(getattr(snap, "positions", {}) or {}) if snap else 0,
                    "graded_today": graded,
                    "ungraded_today": skipped,
                    "unpriced_subjects": unpriced,
                    "awaiting_next_daily_bar_subjects": awaiting_next_daily_bar,
                    "awaiting_next_daily_bar_prediction_ids": awaiting_next_daily_bar_ids,
                    "cache_behind_subjects": cache_behind,
                    "grading_failed_subjects": grading_failed,
                    "shadow_legs_graded": shadow_legs,
                    "episodes_recorded": episodes_written,
                },
            )
        except Exception:
            logger.bind(component="memory").exception("memory grader failed")

    #: How far ahead of the cycle to check the broker. Long enough to open
    #: an app, approve a login and let the gateway finish handshaking;
    #: short enough that the answer is still true when the cycle runs.
    PRECYCLE_LEAD_MINUTES = 60

    async def _check_broker_ready_async(self) -> None:
        """Ask the broker for the account an hour before the cycle needs it.

        Alerts only on failure, and once on recovery after a failure — a
        green message every single week is a message that stops being
        read, which defeats the purpose.
        """
        try:
            from trading.runtime.broker_ready import (
                check_broker_ready,
                format_not_ready_alert,
                format_ready_note,
            )

            result = await asyncio.to_thread(check_broker_ready, self.cycle.broker)
            flag = settings.state_dir / "broker_ready_failed.flag"
            if not result["ready"]:
                flag.parent.mkdir(parents=True, exist_ok=True)
                flag.write_text(result.get("detail", ""))
                self.alerts.critical(
                    format_not_ready_alert(result, minutes_to_cycle=self.PRECYCLE_LEAD_MINUTES)
                )
                return

            # Funding check. A reachable broker is necessary but not
            # sufficient: a CHF-based account with no USD cash will have
            # every US equity order rejected by the margin limit, and the
            # cycle then completes having bought nothing — which reads, in
            # a position report, exactly like a strategy that found no
            # opportunities. Checked here because an hour is enough time
            # to convert; at cycle time it would only be an explanation.
            try:
                from trading.runtime.broker_ready import (
                    check_trade_currency_funding,
                    format_funding_alert,
                )

                funding = await asyncio.to_thread(
                    check_trade_currency_funding,
                    self.cycle.broker,
                    gross_exposure_pct=settings.max_gross_exposure,
                )
                if not funding["ok"]:
                    self.alerts.critical(
                        format_funding_alert(funding, minutes_to_cycle=self.PRECYCLE_LEAD_MINUTES)
                    )
                elif funding.get("reason", "ok") != "ok":
                    logger.bind(component="broker_ready").info(f"funding check {funding['reason']}")
            except Exception:
                logger.bind(component="broker_ready").exception("funding check failed")
            if flag.exists():
                flag.unlink(missing_ok=True)
                self.alerts.info(
                    format_ready_note(result, minutes_to_cycle=self.PRECYCLE_LEAD_MINUTES)
                )
            logger.bind(component="broker_ready").info(
                f"pre-cycle broker check ok (equity {result.get('equity')})"
            )
        except Exception:
            logger.bind(component="broker_ready").exception("pre-cycle broker check failed")

    async def _refresh_universes_async(self) -> None:
        """Weekly index-constituent refresh. Reports the delta, not just
        that it ran — a refresh that fetched an identical list every week
        because the source changed shape would otherwise look healthy."""
        try:
            import asyncio as _asyncio

            from trading.core.universes import available_universes, clear_cache, load_universe

            before = {}
            for name in ("sp500", "nasdaq100", "russell1000"):
                with contextlib.suppress(Exception):
                    before[name] = {i.symbol for i in load_universe(name)}

            # Import, do not subprocess. The first cut shelled out to
            # /app/scripts/refresh_universes.py — a path that does not
            # exist in the image, because the Dockerfile copies src/ and
            # config/ but not scripts/. It would have raised
            # FileNotFoundError every Sunday, been swallowed by the
            # except below, and left the universe frozen while looking
            # scheduled. Importing also avoids a second pandas in RAM.
            from trading.data.universe_refresh import refresh

            result = await _asyncio.to_thread(refresh)
            if not result["ok"]:
                logger.bind(component="data").warning(
                    f"universe refresh failed: {result['reason']}"
                )
                self.alerts.info(f"⚠️ universe refresh failed: {result['reason']}")
                return
            clear_cache()

            lines = []
            for name, old in before.items():
                if name not in available_universes():
                    continue
                new = {i.symbol for i in load_universe(name)}
                added, removed = new - old, old - new
                if added or removed:
                    lines.append(
                        f"{name}: +{len(added)} / -{len(removed)}"
                        + (f" (in: {', '.join(sorted(added)[:6])})" if added else "")
                        + (f" (out: {', '.join(sorted(removed)[:6])})" if removed else "")
                    )
            logger.bind(component="data").info(
                "universe refresh: " + ("; ".join(lines) if lines else "no membership changes")
            )
            if lines:
                self.alerts.info(
                    "🗂 *Index membership changed*\n" + "\n".join(f"• {ln}" for ln in lines)
                )
        except Exception:
            logger.bind(component="data").exception("universe refresh failed")

    async def _refresh_price_cache_async(self) -> None:
        """Top up the parquet cache for the configured universe.

        Read-through: only the missing suffix is fetched, so a normal day
        is one small request per symbol. Bounded concurrency keeps a 1-vCPU
        box usable; failures are per-symbol and never abort the pass.

        Reports staleness rather than assuming success — a refresh that
        quietly fetched nothing is the state this job exists to end.
        """
        try:
            import asyncio as _asyncio
            from datetime import timedelta as _td

            import pandas as _pd

            from trading.core.universes import load_universe

            universe = os.getenv("UNIVERSE", "sp500")
            symbols = {i.symbol.upper() for i in load_universe(universe)}
            cache = ParquetCache(settings.data_dir)
            end = datetime.now(tz=timezone.utc)
            default_start = end - _td(days=30)
            starts = {symbol: default_start for symbol in symbols}
            # Deliberately include subjects blocking the scorecard. The
            # configured universe can be all stocks while a committee made
            # calls on SPY, QQQ or a sector ETF; refreshing only the former
            # leaves those calls ungradable forever.
            try:
                from trading.memory.store import default_store

                _add_scorecard_backfill_targets(
                    symbols,
                    starts,
                    default_store().scorecard_backfill_targets(asof=end),
                    default_start=default_start,
                )
            except Exception:
                logger.bind(component="data").exception("scorecard refresh targets unavailable")
            sem = _asyncio.Semaphore(4)
            ok = failed = 0

            def _one(sym: str) -> bool:
                from trading.data.yfinance_source import YFinanceSource

                ins = Instrument(symbol=sym, asset_class=AssetClass.EQUITY)
                cache.get_bars(YFinanceSource(), ins, starts[sym], end, "1D")
                return True

            async def _guarded(sym: str) -> bool:
                async with sem:
                    try:
                        return await _asyncio.to_thread(_one, sym)
                    except Exception:
                        return False

            for res in await _asyncio.gather(*(_guarded(s) for s in sorted(symbols))):
                if res:
                    ok += 1
                else:
                    failed += 1

            from trading.runtime.portfolio_stats import _read_close

            bench = _read_close(settings.data_dir, "SPY")
            last = str(bench.index.max())[:10] if bench is not None and len(bench) else "unknown"
            age = None
            if bench is not None and len(bench):
                age = (_pd.Timestamp.now(tz="UTC").normalize() - bench.index.max().normalize()).days
            logger.bind(component="data").info(
                f"price cache refresh: {ok} ok, {failed} failed, last SPY bar {last}"
            )
            if age is not None and age > 4:
                self.alerts.info(
                    f"⚠️ price cache still stale after refresh — newest bar {last} "
                    f"({age}d old). Ladders and grading are reading old data."
                )
        except Exception:
            logger.bind(component="data").exception("price cache refresh failed")

    def _grade_shadow(self, mem: Any) -> int:
        """Fill matured forward-return legs on the counterfactual ledger.

        Each leg carries the benchmark over the identical window. A shadow
        return without its benchmark is not a result — in a rising market
        every passed name looks like a missed opportunity — so a row whose
        benchmark cannot be computed is left ungraded rather than graded
        against nothing, and picked up on a later pass.
        """
        from trading.runtime.portfolio_stats import (
            _read_close,
            completed_session_close,
            coverage_status,
        )

        bench_symbol = "SPY"
        bench = _read_close(settings.data_dir, bench_symbol)
        if bench is None or len(bench) < 5:
            logger.bind(component="memory").warning(
                f"shadow grading skipped: no cached closes for {bench_symbol}"
            )
            return 0

        def forward_return(series: Any, ts0: datetime, leg_days: int) -> float | None:
            """Return over ``leg_days`` calendar days from the decision date.

            The tz-naive/aware mix that silently disabled prediction
            grading disabled this too, and here it failed even quieter:
            the comparison raised, the local ``except`` turned it into
            ``None``, and ``None`` is indistinguishable from "not matured
            yet". So the counterfactual ledger looked patiently unfilled
            rather than broken, for every leg, since it was built.
            """
            try:
                due = ts0 + timedelta(days=leg_days)
                base = completed_session_close(series, ts0)
                end = completed_session_close(series, due)
                if not base or end is None:
                    return None
                # A midnight-labelled daily bar can otherwise appear before
                # its session has actually finished.  Require the next bar
                # after the end session before permanently scoring a leg.
                if coverage_status(series, due, asof=datetime.now(tz=timezone.utc)) != "covered":
                    return None
                return end / base - 1.0
            except Exception:
                logger.bind(component="memory").exception("shadow forward return failed")
                return None

        filled = 0
        closes: dict[str, Any] = {}
        for leg_days, _col, _bcol in mem.SHADOW_LEGS:
            for row in mem.ungraded_shadow(leg_days):
                sym = row["symbol"]
                if sym not in closes:
                    closes[sym] = _read_close(settings.data_dir, sym)
                series = closes[sym]
                if series is None or len(series) < 5:
                    continue
                ts0 = datetime.fromtimestamp(row["ts"], tz=timezone.utc)
                ret = forward_return(series, ts0, leg_days)
                bret = forward_return(bench, ts0, leg_days)
                if ret is None or bret is None:
                    continue
                mem.grade_shadow_leg(row["id"], leg_days, ret=ret, bench=bret)
                filled += 1
        if filled:
            logger.bind(component="memory").info(f"shadow ledger: filled {filled} leg(s)")
        return filled

    async def _run_daily_summary_async(self) -> None:
        """Daily after the US close: one-glance equity P&L note.

        Reads the day's first and last snapshot from runner.db (the
        60s refresh keeps those current) — no broker call, no market
        data, so the cost is one SQL read and one Telegram message.
        Silent when there isn't enough data to say something true.
        """
        try:
            now = datetime.now(tz=timezone.utc)
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            bounds = await asyncio.to_thread(self.cycle.runner_store.day_equity_bounds, day_start)
            if bounds is None:
                return
            first, last = bounds
            if first <= 0:
                return
            pct = last / first - 1.0
            snap = self.cycle.runner_store.latest_snapshot()
            ccy = getattr(snap, "base_currency", None) or "USD" if snap else "USD"
            arrow, verb = ("📈", "up") if pct >= 0 else ("📉", "down")
            # Name the book explicitly. This figure is the real IBKR
            # account in its base currency; the simulated PM sleeve posts
            # USD equity into the same chat, and unlabelled the two read
            # as one number that moved.
            env_label = "live" if settings.trading_env == "live" else "paper"
            lines = [
                f"{arrow} Trading account ({env_label}) {verb} {pct:+.2%} today",
                f"Total equity: {ccy} {last:,.0f}",
            ]
            # Portfolio beta vs SPY — cache reads only; skip silently if
            # the book is flat or the cache lacks the names.
            try:
                if snap and snap.positions:
                    from trading.runtime.portfolio_stats import _read_close, portfolio_beta

                    values: dict[str, float] = {}
                    for pos in snap.positions.values():
                        s = _read_close(settings.data_dir, pos.instrument.symbol)
                        if s is not None and len(s):
                            values[pos.instrument.symbol] = float(pos.quantity) * float(s.iloc[-1])
                    result = portfolio_beta(values, settings.data_dir)
                    if result is not None:
                        beta, used = result
                        lines.append(f"Portfolio beta vs SPY: {beta:.2f} ({used} names, 12m)")
            except Exception:
                logger.bind(component="daily_summary").debug("beta computation skipped")
            self.alerts.info("\n".join(lines))
        except Exception:
            logger.bind(component="daily_summary").exception("daily summary failed")

    async def _run_macro_monitor_async(self) -> None:
        """Daily: rates/dollar/energy/BTC financial-conditions dial.
        Advisory only; failures logged and swallowed."""
        try:
            from trading.runtime.macro_monitor import poll_and_alert

            await poll_and_alert()
        except Exception:
            logger.bind(component="macro_monitor").exception("macro monitor poll failed")

    async def _run_style_advisor_async(self) -> None:
        """Weekly: rank registered strategies on trailing 3/6/9-month
        performance from the local price cache and propose a switch when
        the leader changes. Advisory only — never applies anything."""
        try:
            from trading.core.universes import load_universe
            from trading.runtime.style_advisor import poll_and_alert

            instruments = load_universe(self.config.universe)
            prices = await asyncio.to_thread(
                self.cycle._load_prices, instruments, datetime.now(tz=timezone.utc)
            )
            if prices.empty:
                logger.bind(component="style_advisor").info("price cache empty; skipping")
                return
            current = self.config.strategies[0] if self.config.strategies else None
            await poll_and_alert(prices=prices, current_strategy=current)
        except Exception:
            logger.bind(component="style_advisor").exception("style advisor poll failed")

    async def _process_pending_commands(self) -> None:
        """Every 5s: pick up Telegram-queued commands, execute them.

        Runs in a worker thread so a slow broker call doesn't block the
        event loop. APScheduler's ``max_instances=1`` guarantees a single
        instance at a time, so we never have two parallel command
        processors competing for the broker.

        Threads the risk manager through so order-submitting commands
        are halt-gated; otherwise a /halt followed by /buy would still
        submit (audit May 2026).
        """
        try:
            from trading.runtime.command_processor import process_pending

            await asyncio.to_thread(
                process_pending,
                self.broker,
                settings.state_dir,
                self.alerts,
                risk_manager=self.cycle.risk_manager,
            )
        except Exception:
            logger.bind(component="command_processor").exception("command processing failed")

        # /refresh drops a flag rather than fetching inline — a full
        # universe pass takes ~2 minutes and this job holds
        # max_instances=1, so doing it here would block /halt and
        # /flatten behind a data refresh. Hand it to the scheduler as a
        # one-off instead.
        try:
            from trading.runtime.command_processor import REFRESH_FLAG

            flag = settings.state_dir / REFRESH_FLAG
            if flag.exists():
                flag.unlink(missing_ok=True)
                if self._scheduler is not None:
                    from apscheduler.triggers.date import DateTrigger

                    self._scheduler.add_job(
                        self._refresh_price_cache_async,
                        DateTrigger(run_date=datetime.now(tz=timezone.utc)),
                        id="price_cache_refresh_now",
                        replace_existing=True,
                        max_instances=1,
                    )
                    logger.bind(component="runner").info(
                        "operator requested a price-cache refresh; scheduled now"
                    )
        except Exception:
            logger.bind(component="runner").exception("could not schedule requested refresh")

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
            # New structured triggers are fail-closed: only an explicit
            # ``mode=execute`` may reach the order-capable cycle.  A review
            # producer can never become executable because another process
            # happened to read the file while it was being written.  We keep
            # a narrow compatibility path for the old *bare, aware ISO
            # timestamp* flags produced before modes existed.
            review_only = True
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    mode = str(parsed.get("mode", "")).strip().lower()
                    if mode == "execute":
                        review_only = False
                    elif mode != "review":
                        logger.bind(component="runner").warning(
                            f"missing or unknown off-cycle mode {mode!r}; using review-only"
                        )
                else:
                    logger.bind(component="runner").warning(
                        "non-object off-cycle trigger payload; using review-only"
                    )
            except json.JSONDecodeError:
                legacy = payload.strip()
                try:
                    legacy_ts = datetime.fromisoformat(legacy.replace("Z", "+00:00"))
                    if legacy_ts.tzinfo is None or legacy_ts.utcoffset() is None:
                        raise ValueError("legacy timestamp is naive")
                except ValueError:
                    logger.bind(component="runner").warning(
                        "malformed off-cycle trigger; using review-only"
                    )
                else:
                    review_only = False
            await self._run_cycle_async(review_only=review_only)
        except Exception:
            logger.bind(component="runner").exception("off-cycle trigger failed")

    async def _shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
        if self._startup_market_watch_task is not None:
            self._startup_market_watch_task.cancel()
        self.alerts.info("👋 Runner stopped — no further cycles until restart.")
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
