"""Read-only mirror of the LIVE account for the dashboard.

Yan wants to SEE the real account on the Live tab before the system
trades it. This loop connects to a dedicated live-mode IB Gateway
(profile ``mirror`` in docker-compose, logged in with the live username,
``READ_ONLY_API=yes``), snapshots the account every few minutes into its
own ``state_live/runner.db``, and nothing else. The dashboard picks the
directory up via ``DASHBOARD_LIVE_STATE_DIR`` and renders it as its own
sleeve — never merged with the paper book (GO_LIVE.md §1).

Isolation, stated plainly:

* This module NEVER submits, modifies or cancels an order — it calls
  ``get_account()`` and writes a snapshot row, full stop.
* The gateway it talks to is booted with IBKR's Read-Only API flag, so
  even a compromised process on that socket cannot trade.
* It writes to its own state dir. The paper runner's state, orders and
  halt files are untouched.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from trading.runner.state import RunnerStore

from trading.core.logging import logger

_LOG = logger.bind(component="live_mirror")


def snapshot_once(
    state_dir: Path,
    host: str,
    port: int,
    client_id: int,
    *,
    store: RunnerStore | None = None,
) -> bool:
    """One connect → snapshot → persist pass. Returns True on success.

    Fresh broker connection each pass (cheap at this cadence, dodges
    stale-session failure modes) but the connection is ALWAYS torn down
    in ``finally``, and the SQLite store is reused when the loop passes
    one in. The original version leaked one sqlite handle per pass —
    ~96/day — until the process hit EMFILE after 5 days and couldn't
    even open its log file (VPS incident 2026-07-14)."""
    from trading.execution.ibkr import IbkrBroker
    from trading.runner.state import RunnerStore

    broker = IbkrBroker(host=host, port=port, client_id=client_id)
    try:
        broker.connect()
        snap = broker.get_account()
    except Exception as e:
        _LOG.warning(f"snapshot failed ({type(e).__name__}): {e}")
        return False
    finally:
        with contextlib.suppress(Exception):
            broker.disconnect()
    own_store = store is None
    st = store if store is not None else RunnerStore(Path(state_dir) / "runner.db")
    try:
        st.save_snapshot(snap)
    finally:
        if own_store:
            with contextlib.suppress(Exception):
                st.close()
    _LOG.info(
        f"live mirror: equity {snap.equity:,.0f} {snap.base_currency}, "
        f"{len(snap.positions)} positions"
    )
    return True


def run_loop(
    state_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 27,
    interval_s: int = 900,
) -> None:
    """Snapshot forever. Failures are logged and retried next tick — the
    mirror is a viewing convenience, never worth crashing over. One
    store for the process lifetime (see snapshot_once on the FD leak)."""
    from trading.runner.state import RunnerStore

    _LOG.info(f"live mirror starting: {host}:{port} every {interval_s}s → {state_dir}")
    store = RunnerStore(Path(state_dir) / "runner.db")
    while True:
        snapshot_once(Path(state_dir), host, port, client_id, store=store)
        time.sleep(interval_s)
