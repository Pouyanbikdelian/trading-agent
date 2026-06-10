r"""Operator holds — positions pinned outside the strategy's reach.

``/hold NVDA`` writes the symbol into ``state/holds.json``; from then on
the cycle drops ANY order (buy or sell) touching that symbol, so the
position is frozen exactly as it stands until ``/unhold NVDA``. This is
how the operator keeps long-term conviction positions in the same
account without the weekly rebalance cycling them away.

Semantics (deliberately conservative):

* A held symbol's existing position is never sold by the cycle, and the
  cycle never adds to it either — "frozen", not "protected from buys".
* Held positions still count toward equity, so the strategy's sizing
  treats their value as part of the book (the strategy may therefore
  deploy slightly less cash than its weights imply — acceptable, and
  far safer than excluding them from gross/margin checks).
* Manual ``/buy`` / ``/sell`` / ``/close`` commands BYPASS holds — the
  operator saying "sell" explicitly always wins. Only the automated
  cycle is blocked. ``/flatten`` also bypasses: a panic button that
  silently skips positions would be a trap.

State file format: ``{"symbols": ["NVDA", "AAPL"], "updated_at": iso}``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading.core.types import Order

FILENAME = "holds.json"


def load_holds(state_dir: Path) -> set[str]:
    """Read the held-symbol set. Missing/corrupt file = no holds."""
    path = Path(state_dir) / FILENAME
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
        return {str(s).upper() for s in payload.get("symbols", [])}
    except Exception:
        return set()


def save_holds(state_dir: Path, symbols: set[str]) -> None:
    """Atomically persist the held-symbol set (with cross-process lock)."""
    import os
    import tempfile

    from trading.core.file_lock import file_lock

    path = Path(state_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "symbols": sorted(symbols),
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        os.replace(tmp, path)


def filter_held_orders(orders: list[Order], held: set[str]) -> tuple[list[Order], list[Order]]:
    """Split orders into (kept, dropped) by held symbols. Pure function."""
    if not held:
        return list(orders), []
    kept: list[Order] = []
    dropped: list[Order] = []
    for o in orders:
        (dropped if o.instrument.symbol.upper() in held else kept).append(o)
    return kept, dropped
