"""The one way to write ``halt.json``.

Four places used to write this file — ``/halt`` and ``/resume`` in the
bot, ``trading halt`` and ``trading resume`` in the CLI, and the
runner's auto-halt — and all four wrote a fresh four-key dict:
``halted``, ``reason``, ``halted_at``, ``flatten_on_next_cycle``.

``HaltState`` carries three more fields: ``equity_high_watermark``,
``daily_equity_open`` and ``last_day``. It neither forbids extra keys
nor requires the missing ones, so every one of those writes silently
erased the kill-switch baselines and the next load quietly defaulted
them to ``0.0 / 0.0 / None``.

The daily-loss switch skips a zero baseline entirely
(``if daily_equity_open > 0``), and the drawdown switch restarts its
peak from wherever equity happens to be. So halting and resuming gave
both kill switches amnesia — and the drawdown switch could never fire on
a decline that spanned a halt, which is the exact situation in which
somebody halted.

It is also, by accident, what cleared the poisoned CHF 1,068,862
watermark on 2026-08-07. Being saved by a bug is not the same as being
safe.

So: read, modify, write. The halt trio is the only thing any caller may
change; everything else on disk survives untouched. Unrecognised keys
are dropped deliberately — see ``flatten_on_next_cycle`` below.

**On ``flatten_on_next_cycle``**: it was written by ``/halt`` and by the
CLI, and read by nothing, anywhere. The only force-flatten in the system
comes from a playbook regime rule. ``/halt`` replied "Next cycle will
force-flatten positions" and no cycle ever did. The flag is gone rather
than implemented: halting is reversible and liquidating is not, and one
word should not market-sell a book. ``/flatten`` remains the explicit,
separate way to exit — and it, unlike a halt, is meant to be irreversible.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from trading.core.file_lock import file_lock
from trading.core.logging import logger
from trading.risk.limits import HaltState

FILENAME = "halt.json"

#: Fields a halt/resume may touch. Everything else on disk is preserved.
_HALT_TRIO = ("halted", "reason", "halted_at")


def halt_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / FILENAME


def _read_halt_state_unlocked(path: Path) -> HaltState:
    """Best-effort parse while the caller owns ``file_lock(path)``."""
    if not path.exists():
        return HaltState()
    try:
        return HaltState.model_validate_json(path.read_text())
    except Exception as e:
        logger.bind(component="halt_file").warning(
            f"{path} unreadable ({type(e).__name__}); treating as fresh state"
        )
        return HaltState()


def _write_halt_state_unlocked(path: Path, state: HaltState) -> None:
    """Atomically publish a full state while the caller owns its file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(state.model_dump_json(indent=2))
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_halt_state(state_dir: Path | str) -> HaltState:
    """Load the persisted state, defaulting on anything unreadable.

    Unlike ``RiskManager._load_state`` this does NOT fail closed — it is
    used by writers that are about to overwrite the file anyway, and by
    read-only callers who want a best-effort view.
    """
    path = halt_path(state_dir)
    with file_lock(path):
        return _read_halt_state_unlocked(path)


def write_halt_state(state_dir: Path | str, state: HaltState) -> Path:
    """Atomically persist a full ``HaltState``, under the cross-process lock."""
    path = halt_path(state_dir)
    with file_lock(path):
        _write_halt_state_unlocked(path, state)
    return path


def set_halted(
    state_dir: Path | str,
    *,
    halted: bool,
    reason: str = "",
    halted_at: datetime | None = None,
) -> HaltState:
    """Flip the halt flag WITHOUT disturbing the kill-switch baselines.

    ``halted_at`` defaults to now when halting and to None when clearing.
    Always timezone-aware: the runner's auto-halt used a naive
    ``datetime.now()``, which on a UTC-scheduled system in a non-UTC
    container is a timestamp nobody can interpret later.
    """
    path = halt_path(state_dir)
    with file_lock(path):
        # This is one transaction, not a read followed by a separately
        # locked write.  The 60-second risk monitor concurrently updates
        # the high-water mark, and an operator /halt or /resume must never
        # clobber those counters (or be clobbered by them).
        current = _read_halt_state_unlocked(path)
        if halted and halted_at is None:
            halted_at = datetime.now(tz=timezone.utc)
        if not halted:
            halted_at = None
            reason = ""
        new = current.replace(halted=halted, reason=reason, halted_at=halted_at)
        _write_halt_state_unlocked(path, new)
    logger.bind(component="halt_file").warning(
        f"halt.json: halted={halted} reason={reason!r} "
        f"(baselines preserved: open={new.daily_equity_open:,.0f} "
        f"hwm={new.equity_high_watermark:,.0f})"
    )
    return new


def describe_exposure(state_dir: Path | str) -> str:
    """One line on what a halt does NOT do, for the operator's reply.

    A halt stops trading; it does not reduce exposure. Neither do the
    daily-loss and drawdown kill switches — they halt too. The position
    guards are the only thing that exits on their own, and they are off
    by default. Saying so at halt time is the difference between an
    operator who has decided to sit exposed and one who thinks they are
    covered.
    """
    from trading.runtime.guards import enabled as guards_enabled

    if guards_enabled():
        return (
            "Positions are unchanged and still fully exposed. Trailing-stop "
            "guards remain active and may still exit a position; nothing else "
            "will trade. `/flatten` to exit everything."
        )
    return (
        "Positions are unchanged and still fully exposed. Guards are OFF, so "
        "*nothing* will reduce this book while halted. `/flatten` to exit."
    )


class BaselineResetError(Exception):
    """The operator's reset could not be proven safe, so nothing was written."""


def reset_equity_baseline(
    state_dir: Path | str,
    *,
    equity: float,
    currency: str,
    observed_at: datetime,
    reason: str,
    actor: str,
    max_snapshot_age_s: float = 900.0,
    now: datetime | None = None,
) -> tuple[HaltState, HaltState]:
    """Re-stamp the kill-switch baselines to the account as it is today.

    **Why this has to exist.** ``equity_high_watermark`` only ever ratchets
    up; nothing in the system lowers it. With a tight
    ``MAX_DRAWDOWN_PCT`` that makes the drawdown switch a one-way
    trapdoor: once the account is far enough below its peak, every
    evaluation halts, ``/resume`` re-halts on the next tick, and the only
    documented escape — a new equity high — is unreachable because a
    halted book can only be reduced. A limit with no acknowledgement path
    is not a risk control, it is a lockout.

    So the escape is explicit, operator-typed, loud and recorded. It is
    deliberately NOT automatic and deliberately NOT part of ``/resume``:
    accepting a drawdown is a decision, and a decision needs a name on it.

    This never clears a halt. After resetting you still have to
    ``/resume`` — two separate acts, because "I accept this loss" and
    "start trading again" are two separate judgements.

    Refuses (raising :class:`BaselineResetError`, writing nothing) on a
    non-positive equity, a missing currency, a currency that disagrees
    with the stored baseline, or a snapshot too old to describe the
    account now. Returns ``(before, after)`` so the caller can show the
    operator exactly what changed.
    """
    if equity <= 0:
        raise BaselineResetError("account equity is not positive; nothing was reset")
    normalized_currency = str(currency or "").strip().upper()
    if not normalized_currency:
        raise BaselineResetError("account currency is unknown; nothing was reset")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise BaselineResetError("account snapshot has no timezone; nothing was reset")

    moment = now or datetime.now(tz=timezone.utc)
    age_s = (moment - observed_at).total_seconds()
    if age_s > max_snapshot_age_s:
        raise BaselineResetError(
            f"the newest account snapshot is {age_s / 60:.0f} min old — too stale to "
            "re-stamp a baseline against. Check the runner and the gateway first."
        )

    # Anchor to the NYSE session the operator is actually looking at. Off
    # a session day there is nothing to anchor to and the daily-loss switch
    # legitimately has no reference; the high-water reset below still
    # lands, which is the half that matters for a drawdown lockout.
    from trading.runtime.nyse_session import current_nyse_session_label

    session_label = current_nyse_session_label(observed_at)

    path = halt_path(state_dir)
    with file_lock(path):
        before = _read_halt_state_unlocked(path)
        stored_currency = str(before.daily_baseline_currency or "").strip().upper()
        if stored_currency and stored_currency != normalized_currency:
            raise BaselineResetError(
                f"stored baseline is in {stored_currency} but the account reports "
                f"{normalized_currency}; resolve that before resetting"
            )
        # The provenance fields exist to stop an *accidental* mid-session
        # baseline — a restart at noon relabelling a loss as a new day.
        # This is the opposite case: the operator looked at the loss and
        # said "measure from here". So the baseline is stamped as
        # trusted, with ``source`` recording that a human did it rather
        # than the opening bell. Refusing to stamp it would be the safer-
        # looking choice and the wrong one: it would leave every cycle
        # reduce-only until the next open, which is the lockout this
        # command exists to end.
        after = before.replace(
            equity_high_watermark=equity,
            daily_equity_open=equity,
            last_day=session_label,
            daily_baseline_session=session_label,
            # Stamped at the moment of the RESET, not at the snapshot it
            # was computed from. Two reasons. It is what actually happened
            # — the operator established this baseline now. And the live
            # runner adopts a disk baseline only when it is strictly newer
            # than its own; an operator resetting minutes after the
            # opening capture, off a snapshot taken before it, would
            # otherwise write a baseline the runner then ignores.
            daily_baseline_captured_at=moment,
            daily_baseline_source=f"operator_reset:{actor}",
            daily_baseline_currency=normalized_currency,
        )
        _write_halt_state_unlocked(path, after)

    record = {
        "ts": moment.isoformat(),
        "actor": actor,
        "reason": reason,
        "currency": normalized_currency,
        "snapshot_ts": observed_at.isoformat(),
        "snapshot_age_s": round(age_s, 1),
        "equity": equity,
        "previous_high_watermark": before.equity_high_watermark,
        "previous_daily_open": before.daily_equity_open,
        "still_halted": after.halted,
        "session_label": session_label.isoformat() if session_label else None,
    }
    try:
        audit = Path(state_dir) / "baseline_resets.jsonl"
        audit.parent.mkdir(parents=True, exist_ok=True)
        with audit.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as e:
        # The write already happened; losing the audit line must not be
        # silent, but re-raising here would misreport the state on disk.
        logger.bind(component="halt_file").error(f"baseline reset audit write failed: {e!r}")

    logger.bind(component="halt_file").warning(
        f"BASELINE RESET by {actor}: high-water {before.equity_high_watermark:,.2f} → "
        f"{equity:,.2f} {normalized_currency}, daily open "
        f"{before.daily_equity_open:,.2f} → {equity:,.2f} (reason: {reason!r}); "
        f"halt state untouched (halted={after.halted})"
    )
    return before, after
