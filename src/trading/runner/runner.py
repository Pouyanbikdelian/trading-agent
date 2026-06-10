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
from datetime import datetime, timezone
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


_CRON_DOW_NAMES = {
    "MON": "Mondays",
    "TUE": "Tuesdays",
    "WED": "Wednesdays",
    "THU": "Thursdays",
    "FRI": "Fridays",
    "SAT": "Saturdays",
    "SUN": "Sundays",
}


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
            # HMM regime advisor: once daily after US close (≈ 22:00 UTC).
            # Slow signal — daily granularity is enough. Complements the
            # hourly SMA/VIX advisor without spamming.
            self._scheduler.add_job(
                self._run_hmm_advisor_async,
                CronTrigger(hour=22, minute=15, timezone="UTC"),
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

        self._scheduler.start()
        self.alerts.info(self._format_runner_started_message())
        logger.bind(component="runner").info(
            f"scheduler started — next run: {self._scheduler.get_job('cycle').next_run_time}"
        )

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
        try:
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
            age = (datetime.now() - snap.ts.replace(tzinfo=None)).total_seconds()
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

    async def _run_cycle_async(self) -> None:
        """Run one trading cycle with a hard timeout + Telegram-friendly
        error reporting.

        APScheduler executes coroutine jobs natively; we run the synchronous
        cycle in a worker thread so the event loop stays responsive (the
        trigger watcher, the HMM advisor, etc. continue to fire).

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
            report = await asyncio.wait_for(
                asyncio.to_thread(self.cycle.run_cycle),
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
        else:
            # Success — reset the consecutive error counter (and persisted file).
            if self._consecutive_errors > 0:
                self.alerts.info(f"✅ cycle recovered (after {self._consecutive_errors} errors)")
            self._consecutive_errors = 0
            self._save_error_counter()
            self._last_success_ts = datetime.now()

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
            import json as _json
            import os as _os
            import tempfile as _tmp

            halt_path = settings.state_dir / "halt.json"
            halt_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = _tmp.mkstemp(dir=halt_path.parent, prefix=f"{halt_path.name}.")
            with _os.fdopen(fd, "w") as f:
                _json.dump(
                    {
                        "halted": True,
                        "reason": f"auto-halt after {self._consecutive_errors} consecutive failures: {reason}",
                        "halted_at": datetime.now().isoformat(),
                        "flatten_on_next_cycle": False,  # don't flatten reflexively
                    },
                    f,
                    indent=2,
                )
            _os.replace(tmp, halt_path)
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
        """
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
