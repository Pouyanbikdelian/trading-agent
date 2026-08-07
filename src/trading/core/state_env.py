"""State directories remember which environment wrote them.

On 2026-08-07 the live account halted at "daily loss -91.82%" seven
minutes into its first session. Nothing was wrong with the market or the
book. ``STATE_DIR`` had been pointed at ``state/live`` an hour earlier —
to dry-run the live code path — while the gateway was still in PAPER
mode. That cycle wrote CHF 1,068,862 of paper equity into
``daily_equity_open`` and ``equity_high_watermark``. ``start_of_day`` is
idempotent per date, so when the real session started it never
re-stamped, and the kill switch compared 87,413 of live equity against a
paper baseline.

The mistake was one environment variable, held for one hour, and it
produced a number that looked like a catastrophic loss. Nothing in the
system could tell that the two figures came from different accounts,
because a state directory is just files: it carries no record of who
wrote it.

So it carries one now. The first process to open a state directory
stamps it with its ``TRADING_ENV``; every process after that must agree.
A paper process pointed at a live directory fails at startup, by name,
before it can write a single figure — instead of succeeding quietly and
poisoning a baseline that only detonates an hour later in the one
session where being wrong is expensive.

Deliberately strict about the failure and lenient about adoption: an
unstamped directory is adopted (existing deployments have several), but
a *mismatch* is fatal and never auto-resolved. The recovery is an
explicit operator act — ``trading state stamp --env live``, or deleting
``env.json`` — because every way of making this self-healing amounts to
guessing which of two accounts the numbers on disk belong to.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from trading.core.logging import logger

#: Lives inside the state directory, alongside halt.json and runner.db.
STAMP_FILENAME = "env.json"

#: Files whose presence means the directory already holds real trading
#: state, as opposed to being a fresh mkdir. Only used to decide how
#: loudly to log an adoption — never to change the outcome.
_SUBSTANTIVE = ("halt.json", "runner.db", "orders.db")


class StateEnvMismatchError(RuntimeError):
    """A process opened a state directory belonging to another env."""


class StateEnvCheck(NamedTuple):
    """Outcome of verifying a state directory against a running env."""

    ok: bool
    running_env: str
    stamped_env: str | None
    adopted: bool
    detail: str


def stamp_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / STAMP_FILENAME


def read_stamp(state_dir: Path | str) -> str | None:
    """The env this directory belongs to, or None if never stamped.

    A corrupt stamp reads as unstamped rather than raising: the value of
    this file is the mismatch it catches, and a JSON error should not be
    able to wedge every process that touches the directory.
    """
    path = stamp_path(state_dir)
    if not path.exists():
        return None
    try:
        env = json.loads(path.read_text()).get("trading_env")
    except Exception:
        logger.bind(component="state_env").warning(f"{path} unreadable — treating as unstamped")
        return None
    return str(env) if env else None


def write_stamp(state_dir: Path | str, env: str) -> Path:
    """Atomically (re)stamp a state directory. Overwrites without asking."""
    path = stamp_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trading_env": env,
        "stamped_at": datetime.now(tz=timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def _looks_substantive(state_dir: Path) -> bool:
    return any((state_dir / name).exists() for name in _SUBSTANTIVE)


def verify_state_dir(
    state_dir: Path | str,
    env: str,
    *,
    adopt: bool = True,
) -> StateEnvCheck:
    """Check a state directory against the env about to write to it.

    Never raises — see :func:`assert_state_dir_env` for the enforcing
    wrapper. Returns ``ok=False`` only on a genuine mismatch.
    """
    state_dir = Path(state_dir)
    stamped = read_stamp(state_dir)

    if stamped is None:
        if not adopt:
            return StateEnvCheck(True, env, None, False, "unstamped (not adopted)")
        state_dir.mkdir(parents=True, exist_ok=True)
        substantive = _looks_substantive(state_dir)
        write_stamp(state_dir, env)
        detail = f"adopted {state_dir} for env={env!r}"
        if substantive:
            # Worth a warning: we are claiming a directory that already
            # holds a halt state and an order ledger. If that state came
            # from somewhere else, this is the moment to notice.
            detail += " — directory already held trading state"
            logger.bind(component="state_env").warning(detail)
        else:
            logger.bind(component="state_env").info(detail)
        return StateEnvCheck(True, env, None, True, detail)

    if stamped == env:
        return StateEnvCheck(True, env, stamped, False, "ok")

    detail = (
        f"state dir {state_dir} was written by TRADING_ENV={stamped!r} but this process is {env!r}"
    )
    logger.bind(component="state_env").error(detail)
    return StateEnvCheck(False, env, stamped, False, detail)


def explain_mismatch(check: StateEnvCheck, state_dir: Path | str) -> str:
    """The message an operator reads at 3pm with the market open.

    Leads with what to do, because by the time this fires the process is
    already refusing to start and the diagnosis is the easy part.
    """
    return (
        f"REFUSING TO START — {check.detail}.\n\n"
        "Equity, the daily-loss baseline and the drawdown high-water mark on "
        "disk describe a DIFFERENT account. Starting here is how 2026-08-07 "
        "halted at -91.82% on its first live day.\n\n"
        "Do one of:\n"
        f"  • point STATE_DIR somewhere else (most likely: you meant "
        f"state/{check.running_env})\n"
        f"  • if this directory really is {check.running_env!r} now, wipe the "
        f"stale baseline first — remove halt.json, runner.db*, orders.db* — "
        f"then `trading state stamp --env {check.running_env}`\n"
        f"  • or delete {stamp_path(state_dir)} to re-adopt (only safe if the "
        "directory is empty of trading state)"
    )


def assert_state_dir_env(
    state_dir: Path | str,
    env: str,
    *,
    adopt: bool = True,
) -> StateEnvCheck:
    """Verify, and raise :class:`StateEnvMismatchError` if the envs disagree."""
    check = verify_state_dir(state_dir, env, adopt=adopt)
    if not check.ok:
        raise StateEnvMismatchError(explain_mismatch(check, state_dir))
    return check
