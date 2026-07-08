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

from trading.core.logging import logger

_LOG = logger.bind(component="live_mirror")


def snapshot_once(state_dir: Path, host: str, port: int, client_id: int) -> bool:
    """One connect → snapshot → persist pass. Returns True on success.
    Fresh connection each pass: at a 5-15 min cadence the reconnect cost
    is trivial and it sidesteps every stale-session failure mode."""
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
    RunnerStore(Path(state_dir) / "runner.db").save_snapshot(snap)
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
    mirror is a viewing convenience, never worth crashing over."""
    _LOG.info(f"live mirror starting: {host}:{port} every {interval_s}s → {state_dir}")
    while True:
        snapshot_once(Path(state_dir), host, port, client_id)
        time.sleep(interval_s)
