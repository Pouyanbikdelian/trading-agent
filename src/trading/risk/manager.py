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

import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import date, datetime, timezone
from pathlib import Path
from threading import RLock

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


def _baseline_is_newer(disk: HaltState, memory: HaltState) -> bool:
    """Whether ``disk`` carries a baseline captured after ``memory``'s.

    Deliberately strict. An equal timestamp is the same capture re-read,
    and a missing timestamp on disk is a legacy or unprovenanced state
    that must never displace a verified in-memory one.
    """
    fresh = disk.daily_baseline_captured_at
    if fresh is None or disk.equity_high_watermark <= 0:
        return False
    held = memory.daily_baseline_captured_at
    if held is None:
        return True
    if fresh.tzinfo is None or held.tzinfo is None:
        # A naive timestamp cannot be ordered against an aware one.
        # Refuse rather than guess; the next verified capture repairs it.
        return False
    return fresh > held


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
        # Snapshot refresh and an off-cycle/manual cycle can inspect the same
        # manager concurrently.  Serialise each read-modify-write transition
        # in-process; the file lock below additionally coordinates Telegram
        # and CLI writers in the other container.
        self._state_lock = RLock()
        self._state = self._load_state()
        #: Set when start_of_day repairs an implausible baseline; drained
        #: by take_baseline_note() so the runner alerts once.
        self._last_baseline_note: str | None = None

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
                return self._read_state_unlocked()
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

    def _read_state_unlocked(self) -> HaltState:
        """Parse the state while the caller already owns the file lock."""
        if not (self._halt_path and self._halt_path.exists()):
            return HaltState()
        return HaltState.model_validate_json(self._halt_path.read_text())

    def _save_state(self, *, halt_intent: bool | None = None) -> None:
        if self._halt_path is None:
            return
        # File lock + atomic rename: both the bot and the manager are
        # writers; serialising under flock ensures no torn writes or
        # silently-dropped intent (audit fix #9).
        from trading.core.file_lock import file_lock

        self._halt_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self._halt_path):
            # A Telegram /halt can land after our preceding reload but before
            # this save.  A fresh halt always wins over a stale in-memory
            # unhalted state; otherwise a harmless HWM update could undo the
            # operator's emergency stop.  The manager is the sole writer of
            # baseline/HWM fields, so retain its current counters.
            try:
                disk = self._read_state_unlocked()
            except Exception as e:
                logger.bind(component="risk").error(
                    f"halt.json could not be parsed during save ({type(e).__name__}: {e!r}); "
                    "FAILING CLOSED"
                )
                disk = HaltState(
                    halted=True,
                    reason=f"halt.json corrupt: {type(e).__name__}",
                    halted_at=datetime.now(timezone.utc),
                )
            if halt_intent is None and disk.halted and not self._state.halted:
                self._state = self._state.replace(
                    halted=True,
                    reason=disk.reason,
                    halted_at=disk.halted_at,
                )
            fd, tmp_name = tempfile.mkstemp(
                dir=self._halt_path.parent,
                prefix=f"{self._halt_path.name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w") as stream:
                    stream.write(self._state.model_dump_json(indent=2))
                os.replace(tmp_name, self._halt_path)
            except Exception:
                with suppress(FileNotFoundError):
                    os.unlink(tmp_name)
                raise

    # ------------------------------------------------------ halt control

    @property
    def state(self) -> HaltState:
        with self._state_lock:
            return self._state

    def is_halted(self) -> bool:
        with self._state_lock:
            return self._state.halted

    def halt(self, reason: str) -> None:
        with self._state_lock:
            self._state = self._state.replace(
                halted=True,
                reason=reason,
                halted_at=datetime.now(timezone.utc),
            )
            self._save_state(halt_intent=True)
        logger.bind(component="risk").warning(f"HALTED — {reason}")

    def unhalt(self) -> None:
        """Operator-only command. Phase 8's runner will gate this behind a
        manual confirmation; the risk manager itself never auto-unhalts."""
        with self._state_lock:
            if not self._state.halted:
                return
            prev = self._state.reason
            self._state = self._state.replace(halted=False, reason="", halted_at=None)
            self._save_state(halt_intent=False)
        logger.bind(component="risk").warning(f"UNHALTED (was: {prev})")

    def _reload_halt_state(self) -> None:
        """Refresh halt-related fields from disk, preserving in-memory
        daily P&L tracking — and adopt a NEWER baseline written elsewhere.

        We do NOT swap ``self._state`` wholesale: the runner populates
        ``daily_equity_open`` and ``equity_high_watermark`` in-memory and
        those don't always land in halt.json between processes. Reloading
        just the halt trio keeps the bot's /halt and /resume effective
        without losing live counters.

        That was complete only while this manager was the sole writer of
        the baseline fields. ``/baseline reset`` (2026-08-19) made the bot
        a second writer, and on 2026-08-20 the consequence showed up in
        production: the operator reset his high-water mark, this manager
        kept the stale 88,182.64 in memory, halted on a drawdown measured
        against it, and ``_save_state`` wrote the stale figure straight
        back over the reset. The reset appeared to do nothing, forever.

        So the baseline is adopted when the copy on disk was captured
        LATER than ours. ``daily_baseline_captured_at`` is the arbiter:
        both writers stamp it, an operator reset always stamps "now", and
        a same-instant re-read is a no-op. All seven fields move together
        because they describe one capture — taking the mark without its
        provenance is how a baseline stops being trustworthy.
        """
        with self._state_lock:
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
            if _baseline_is_newer(disk, self._state):
                logger.bind(component="risk").warning(
                    "adopting a newer baseline from disk: high-water "
                    f"{self._state.equity_high_watermark:,.2f} -> "
                    f"{disk.equity_high_watermark:,.2f}, daily open "
                    f"{self._state.daily_equity_open:,.2f} -> "
                    f"{disk.daily_equity_open:,.2f} "
                    f"(source {disk.daily_baseline_source!r})"
                )
                self._state = self._state.replace(
                    equity_high_watermark=disk.equity_high_watermark,
                    daily_equity_open=disk.daily_equity_open,
                    last_day=disk.last_day,
                    daily_baseline_session=disk.daily_baseline_session,
                    daily_baseline_captured_at=disk.daily_baseline_captured_at,
                    daily_baseline_source=disk.daily_baseline_source,
                    daily_baseline_currency=disk.daily_baseline_currency,
                )

    @contextmanager
    def submission_gate(self, *, allow_when_halted: bool = False) -> Iterator[bool]:
        """Hold the halt-state transaction through one broker submission.

        The cycle holds its broader execution lock for the batch.  This
        narrower gate additionally serialises each individual submit with
        Telegram/CLI ``/halt`` writes: after a halt is requested, at most the
        one submission already inside this gate may finish, and every later
        order in the batch is stopped.  Holding the file lock for a single
        broker call is intentional; holding it for the whole batch would
        delay an emergency halt behind unrelated orders.
        """
        with self._state_lock:
            if self._halt_path is None:
                yield not self._state.halted or allow_when_halted
                return
            from trading.core.file_lock import file_lock

            with file_lock(self._halt_path):
                try:
                    disk = self._read_state_unlocked()
                except Exception as e:
                    self._state = self._state.replace(
                        halted=True,
                        reason=f"halt.json corrupt: {type(e).__name__}",
                        halted_at=datetime.now(timezone.utc),
                    )
                    yield False
                    return
                self._state = self._state.replace(
                    halted=disk.halted,
                    reason=disk.reason,
                    halted_at=disk.halted_at,
                )
                yield not self._state.halted or allow_when_halted

    def baseline_divergence(self, equity: float) -> float | None:
        """|equity − stored daily open| as a fraction of the stored open.

        None when there is nothing to compare against. Pure — exposed so
        the runner and its tests can ask the same question the kill
        switch asks, without reaching into private state.
        """
        base = self._state.daily_equity_open
        if base <= 0 or equity <= 0:
            return None
        return abs(equity - base) / base

    def start_of_day(self, account: AccountSnapshot) -> str | None:
        """Legacy wall-clock baseline helper.

        Kept for backwards-compatible research/unit-test callers. Live
        execution must use ``capture_session_open`` plus
        ``evaluate_session_risk``: a wall-clock first observation is not a
        trustworthy market-open baseline after a restart.
        """
        with self._state_lock:
            today = account.ts.date()
            divergence = self.baseline_divergence(account.equity)
            implausible = (
                divergence is not None and divergence > self.limits.baseline_sanity_divergence_pct
            )

            if self._state.last_day == today and not implausible:
                return None

            if implausible:
                stale_open = self._state.daily_equity_open
                stale_hwm = self._state.equity_high_watermark
                note = (
                    f"baseline re-stamped: stored daily open {stale_open:,.0f} is "
                    f"{divergence:.0%} from actual equity {account.equity:,.0f} — "
                    f"beyond the {self.limits.baseline_sanity_divergence_pct:.0%} "
                    "sanity limit, so it describes a different account, currency or "
                    f"funding level rather than a loss. High-water mark {stale_hwm:,.0f} "
                    "reset with it."
                )
                # Reset, do not max(): a high-water mark carried over from a
                # baseline we have just declared bogus is equally bogus, and
                # keeping it would halt on drawdown one bar later.
                self._state = self._state.replace(
                    last_day=today,
                    daily_equity_open=account.equity,
                    equity_high_watermark=account.equity,
                    daily_baseline_session=None,
                    daily_baseline_captured_at=None,
                    daily_baseline_source=None,
                    daily_baseline_currency=None,
                )
                self._save_state()
                logger.bind(component="risk").warning(note)
                self._last_baseline_note = note
                return note

            new_hwm = max(self._state.equity_high_watermark, account.equity)
            self._state = self._state.replace(
                last_day=today,
                daily_equity_open=account.equity,
                equity_high_watermark=new_hwm,
                # This is explicitly *not* trusted by the live session
                # monitor.  It preserves legacy tests/research semantics
                # without making a noon restart look like an opening print.
                daily_baseline_session=None,
                daily_baseline_captured_at=None,
                daily_baseline_source=None,
                daily_baseline_currency=None,
            )
            self._save_state()
            return None

    def take_baseline_note(self) -> str | None:
        """Pop the last baseline-repair note, if any. Read-once so the
        runner alerts on the repair rather than on every cycle after it."""
        with self._state_lock:
            note = getattr(self, "_last_baseline_note", None)
            self._last_baseline_note = None
            return note

    # ------------------------------------------------ session-aware intraday

    def capture_session_open(
        self,
        account: AccountSnapshot,
        *,
        session_date: date,
        captured_at: datetime,
        source: str = "session_open",
    ) -> str | None:
        """Record a daily baseline only during the real NYSE opening window.

        The runner calls this from its 60-second snapshot job.  It validates
        both the exchange calendar date and the short opening window itself,
        so an accidental noon/restart caller cannot relabel a loss as a new
        day.  Recording a new baseline never clears an existing halt.
        """
        from trading.runtime.nyse_session import (
            current_nyse_session_label,
            is_opening_capture_window,
        )

        try:
            actual_session = current_nyse_session_label(captured_at)
            is_open = is_opening_capture_window(captured_at)
        except ValueError as e:
            logger.bind(component="risk").warning(
                f"refusing daily baseline capture with invalid timestamp: {e}"
            )
            return None
        if actual_session != session_date or not is_open:
            logger.bind(component="risk").warning(
                "refusing daily baseline outside the verified NYSE opening window"
            )
            return None
        if account.equity <= 0:
            logger.bind(component="risk").warning(
                "refusing daily baseline from non-positive account equity"
            )
            return None

        currency = str(account.base_currency or "").upper()
        if not currency:
            logger.bind(component="risk").warning(
                "refusing daily baseline with missing account currency"
            )
            return None

        with self._state_lock:
            self._reload_halt_state()
            same_trusted_session = (
                self._state.daily_baseline_session == session_date
                and self._state.daily_baseline_captured_at is not None
                and self._state.daily_baseline_source is not None
                and self._state.daily_baseline_currency == currency
                and self._state.daily_equity_open > 0
            )
            if same_trusted_session:
                return None

            divergence = self.baseline_divergence(account.equity)
            reset_hwm = self._state.daily_baseline_currency not in {None, currency} or (
                divergence is not None and divergence > self.limits.baseline_sanity_divergence_pct
            )
            new_hwm = (
                account.equity
                if reset_hwm
                else max(self._state.equity_high_watermark, account.equity)
            )
            self._state = self._state.replace(
                last_day=session_date,
                daily_equity_open=account.equity,
                equity_high_watermark=new_hwm,
                daily_baseline_session=session_date,
                daily_baseline_captured_at=captured_at,
                daily_baseline_source=source,
                daily_baseline_currency=currency,
            )
            self._save_state()
            if not reset_hwm:
                return None
            note = (
                "session baseline re-stamped with high-water reset: prior state "
                "described a different account, currency or funding level"
            )
            self._last_baseline_note = note
            logger.bind(component="risk").warning(note)
            return note

    def _trusted_daily_baseline_reason(
        self,
        *,
        session_label: date | None,
        base_currency: str,
    ) -> str | None:
        """Explain why the current state cannot support a daily-loss check."""
        if session_label is None:
            return "NYSE session unavailable; daily-loss baseline is not authorised"
        if self._state.daily_baseline_session != session_label:
            return "daily-loss baseline unavailable for the current NYSE session"
        if (
            self._state.daily_baseline_captured_at is None
            or not self._state.daily_baseline_source
            or self._state.daily_equity_open <= 0
        ):
            return "daily-loss baseline lacks verified NYSE-open provenance"
        stored_currency = str(self._state.daily_baseline_currency or "").upper()
        if stored_currency != base_currency:
            return (
                "daily-loss baseline currency does not match the live account "
                f"({stored_currency or 'missing'} vs {base_currency})"
            )
        return None

    def evaluate_session_risk(
        self,
        account: AccountSnapshot,
        *,
        session_label: date | None,
        require_daily_baseline: bool = True,
    ) -> RiskDecision:
        """Evaluate live risk without ever inventing a new daily baseline.

        This is the only intraday evaluator used by live runner paths.  A
        missing, legacy, wrong-session, or wrong-currency baseline blocks new
        exposure with ``reject`` rather than persisting a misleading halt.
        A genuine loss/drawdown still persists a hard halt immediately.
        """
        base_currency = str(account.base_currency or "").upper()
        if account.equity <= 0 or not base_currency:
            return RiskDecision(
                action="reject",
                reason="live account snapshot is not usable for risk evaluation",
            )

        with self._state_lock:
            self._reload_halt_state()
            if self._state.halted:
                return RiskDecision(action="halt", reason=f"already halted: {self._state.reason}")

            baseline_reason = self._trusted_daily_baseline_reason(
                session_label=session_label,
                base_currency=base_currency,
            )
            has_trusted_provenance = (
                self._state.daily_baseline_captured_at is not None
                and bool(self._state.daily_baseline_source)
                and bool(self._state.daily_baseline_currency)
            )
            # ``baseline_reason`` is non-None on both branches below (a
            # currency mismatch or missing provenance always produces one),
            # but default it explicitly rather than trust that reasoning at
            # a safety boundary: RiskDecision.reason is a required str.
            untrusted_reason = baseline_reason or "daily-loss baseline is not authorised"
            if (
                self._state.daily_baseline_currency
                and self._state.daily_baseline_currency.upper() != base_currency
            ):
                return RiskDecision(action="reject", reason=untrusted_reason)

            # A pre-provenance state may contain a paper-account high-water
            # mark. Do not compare it to a live snapshot and manufacture a
            # false drawdown halt; wait for the next verified session open.
            # Once the account/currency provenance is sound, keep checking
            # drawdown even if today's daily-open capture was missed.
            if not has_trusted_provenance:
                return RiskDecision(action="reject", reason=untrusted_reason)

            if account.equity > self._state.equity_high_watermark:
                self._state = self._state.replace(equity_high_watermark=account.equity)
                self._save_state()
                if self._state.halted:
                    return RiskDecision(
                        action="halt", reason=f"already halted: {self._state.reason}"
                    )

            if baseline_reason is None:
                day_pnl = (
                    account.equity - self._state.daily_equity_open
                ) / self._state.daily_equity_open
                if day_pnl <= -self.limits.max_daily_loss_pct:
                    self.halt(
                        f"daily loss {day_pnl:.2%} breaches limit "
                        f"-{self.limits.max_daily_loss_pct:.2%}"
                    )
                    return RiskDecision(action="halt", reason=self._state.reason)

            if self._state.equity_high_watermark > 0:
                drawdown = (
                    account.equity - self._state.equity_high_watermark
                ) / self._state.equity_high_watermark
                if drawdown <= -self.limits.max_drawdown_pct:
                    self.halt(
                        f"drawdown {drawdown:.2%} breaches limit "
                        f"-{self.limits.max_drawdown_pct:.2%}"
                    )
                    return RiskDecision(action="halt", reason=self._state.reason)

            if baseline_reason is not None and require_daily_baseline:
                return RiskDecision(action="reject", reason=baseline_reason)
            return RiskDecision(action="allow", reason="session risk checks passed")

    # ------------------------------------------------------ intraday

    def evaluate_intraday(self, account: AccountSnapshot) -> RiskDecision:
        """Legacy wall-clock intraday check retained for non-live callers.

        Live runner paths use ``evaluate_session_risk`` so a restart cannot
        define an opening balance at an arbitrary time of day.
        """
        with self._state_lock:
            return self._evaluate_intraday_legacy(account)

    def _evaluate_intraday_legacy(self, account: AccountSnapshot) -> RiskDecision:
        """Original research/paper behaviour, called under ``_state_lock``."""
        # Re-read halt.json before checking. The Telegram bot (a separate
        # process) writes this file from /halt and /resume; without a
        # reload here, the trader process would only see halt changes on
        # its next restart. This caused a real incident: after /resume
        # cleared the halt on disk, the still-running cycle saw the stale
        # in-memory state and refused with "already halted" (2026-05-22).
        self._reload_halt_state()
        if self._state.halted:
            return RiskDecision(action="halt", reason=f"already halted: {self._state.reason}")

        # Roll the daily baseline. start_of_day() no-ops when last_day is
        # already today AND the stored baseline is plausible, so call it
        # unconditionally.
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
        """Convert a target signal into executable orders through the hard gate.

        This is deliberately the only normal execution-facing sizing entry
        point.  A persisted halt always wins here, even if a caller has an
        otherwise valid signal.
        """
        if self._state.halted:
            return [], [RiskDecision(action="halt", reason=f"halted: {self._state.reason}")]
        return self._plan_signal_to_orders(
            signal,
            account=account,
            last_prices=last_prices,
            instruments=instruments,
            sector_map=sector_map,
            order_id_factory=order_id_factory,
            fx_rates=fx_rates,
            pending_orders=pending_orders,
        )

    def preview_signal_to_orders(
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
        """Build a non-executable, risk-sized review plan while halted.

        This method exists solely for ``Cycle``'s explicit review-only path.
        It applies the identical static sizing, concentration, long-only,
        pending-order and no-margin protections as ``signal_to_orders`` but
        does not read, clear, or weaken the persistent execution halt.  The
        orders it returns are ephemeral display objects: callers must never
        pass them to a broker or an approval queue.
        """
        return self._plan_signal_to_orders(
            signal,
            account=account,
            last_prices=last_prices,
            instruments=instruments,
            sector_map=sector_map,
            order_id_factory=order_id_factory,
            fx_rates=fx_rates,
            pending_orders=pending_orders,
        )

    def _plan_signal_to_orders(
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
        if account.equity <= 0:
            return [], [RiskDecision(action="reject", reason="non-positive equity")]

        # Signed share deltas of orders already working at the broker,
        # keyed by NORMALIZED instrument key — netted into current
        # position below. Normalization: IBKR reports ETFs as plain
        # stocks (secType STK), so a broker-side ``equity:SPY`` must
        # match a universe-side ``etf:SPY``. ``pending_sell`` tracks the
        # sell side alone: only sells shrink what we're allowed to sell
        # (long-only clamp) — counting an UNFILLED buy as owned could
        # let a sell fire before the buy fills and end short (external
        # review, 2026-07-15).
        def _norm(key: str) -> str:
            return f"equity:{key.split(':', 1)[1]}" if key.startswith("etf:") else key

        pending_delta: dict[str, float] = {}
        pending_sell: dict[str, float] = {}
        for po in pending_orders or []:
            k = _norm(po.instrument.key)
            signed = po.quantity if po.side == Side.BUY else -po.quantity
            pending_delta[k] = pending_delta.get(k, 0.0) + signed
            if signed < 0:
                pending_sell[k] = pending_sell.get(k, 0.0) + signed
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
            # Position lookup through the same normalization as pending
            # orders (etf:X ≡ equity:X — IBKR reports ETFs as STK).
            nkey = _norm(key)
            settled_qty = 0.0
            for pkey, pos in account.positions.items():
                if _norm(pkey) == nkey:
                    settled_qty += pos.quantity
            # Where the book WILL be once working orders fill (sizing).
            current_qty = settled_qty + pending_delta.get(nkey, 0.0)
            delta = target_qty - current_qty
            if whole_shares_only:
                delta = float(int(delta))
            # Long-only invariant, quantity half: a sell may flatten the
            # effective position but never cross zero. The sellable base
            # is settled shares net of pending SELLS only — an unfilled
            # buy is not ours to sell yet.
            if not self.limits.allow_short and delta < 0:
                max_sell = max(settled_qty + pending_sell.get(nkey, 0.0), 0.0)
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
        working_orders: list[Order] | None = None,
    ) -> list[Order]:
        """Build market orders that close every non-zero position.

        Bypasses every exposure limit — this is the emergency exit — and
        deliberately ignores ``/hold``: a panic button that silently skips
        positions is a trap.

        It does NOT bypass the long-only invariant. ``working_orders`` are
        the orders already live at the broker; their signed quantity is
        netted out, because a full-size close on top of a full-size close
        already working crosses zero and leaves a SHORT in the name you
        were trying to exit. That is the 2026-07-15 stacking bug, and this
        was the third path still missing the fix (the cycle got it then,
        the command handlers on 2026-08-07, this one an hour later).

        Netting can only ever make a flatten sell LESS, never leave it
        less complete: the shares are already on their way out.

        Omitting ``working_orders`` keeps the old unnetted behaviour, so a
        caller with no way to ask the broker still flattens.
        """
        ts = ts or datetime.now(timezone.utc)

        def _norm(key: str) -> str:
            return f"equity:{key.split(':', 1)[1]}" if key.startswith("etf:") else key

        # Signed remaining quantity per instrument, from every source.
        pending: dict[str, float] = {}
        for wo in working_orders or []:
            k = _norm(wo.instrument.key)
            pending[k] = pending.get(k, 0.0) + (
                wo.quantity if wo.side == Side.BUY else -wo.quantity
            )

        orders: list[Order] = []
        for pos in positions:
            if abs(pos.quantity) < _EPS_QTY:
                continue
            # What still needs closing once the working orders land.
            remaining = pos.quantity + pending.get(_norm(pos.instrument.key), 0.0)
            if abs(remaining) < _EPS_QTY:
                logger.bind(component="risk").info(
                    f"force-flatten: {pos.instrument.symbol} already fully covered "
                    "by working orders — not duplicating"
                )
                continue
            # Never flip the sign: if working orders would OVERSHOOT, close
            # only what we actually hold rather than opening the opposite side.
            if remaining * pos.quantity < 0:
                remaining = 0.0
                continue
            orders.append(
                Order(
                    client_order_id=order_id_factory(),
                    instrument=pos.instrument,
                    side=Side.SELL if pos.quantity > 0 else Side.BUY,
                    quantity=abs(remaining),
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.DAY,
                    created_at=ts,
                )
            )
        return orders
