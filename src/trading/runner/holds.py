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


OVERRIDES_FILENAME = "strategy_overrides.json"


def load_k_override(state_dir: Path) -> int | None:
    """Operator's runtime ``k`` override (set via ``/k N``), or None."""
    path = Path(state_dir) / OVERRIDES_FILENAME
    if not path.exists():
        return None
    try:
        k = json.loads(path.read_text()).get("k")
        return int(k) if k else None
    except Exception:
        return None


def save_k_override(state_dir: Path, k: int | None) -> None:
    import os
    import tempfile

    from trading.core.file_lock import file_lock

    path = Path(state_dir) / OVERRIDES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
        with os.fdopen(fd, "w") as f:
            json.dump(
                {"k": k, "updated_at": datetime.now(tz=timezone.utc).isoformat()},
                f,
                indent=2,
            )
        os.replace(tmp, path)


def apply_runtime_overrides(
    params: object,
    state_dir: Path,
    *,
    position_symbols: set[str] | None = None,
) -> tuple[object, list[str]]:
    """Adjust a strategy's params for operator runtime state.

    Two adjustments, both only when the params object has a ``k`` field
    (top-K style strategies; everything else passes through untouched):

    1. ``/k N`` override replaces the configured k.
    2. Each held symbol (``/hold``) WITH AN OPEN POSITION reserves one
       basket slot: effective_k = max(1, k - n_holds). The strategy then
       picks fewer names because the operator's pinned positions occupy
       the rest of the book — "if I hold 2, cycle 6 more".

       ``position_symbols`` is the set of symbols with live positions.
       A pin without a position protects nothing and must not shrink the
       basket (found 2026-07-14: a stale pin on an unheld symbol silently
       ate a slot). When the caller can't supply positions (previews),
       pass ``None`` — every pin then counts, the old conservative
       behavior.

    Returns (params, notes) where notes are human-readable lines for
    the basket message.
    """
    if not hasattr(params, "k"):
        return params, []
    notes: list[str] = []
    k = int(params.k)  # type: ignore[attr-defined]
    override = load_k_override(state_dir)
    if override is not None and override != k:
        notes.append(f"k overridden via /k: {k} → {override}")
        k = override
    held = load_holds(state_dir)
    if position_symbols is not None:
        ghost = held - {s.upper() for s in position_symbols}
        held = held - ghost
        if ghost:
            notes.append(
                f"{len(ghost)} pin(s) without a position ignored for slot math: "
                f"{', '.join(sorted(ghost))} (consider /unhold)"
            )
    if held:
        reserved = min(len(held), k - 1)
        if reserved > 0:
            notes.append(
                f"{len(held)} pinned position(s) reserve basket slots: k {k} → {k - reserved}"
            )
            k = k - reserved
    new_params = params.model_copy(update={"k": k})  # type: ignore[attr-defined]
    return new_params, notes


def filter_held_orders(orders: list[Order], held: set[str]) -> tuple[list[Order], list[Order]]:
    """Split orders into (kept, dropped) by held symbols. Pure function."""
    if not held:
        return list(orders), []
    kept: list[Order] = []
    dropped: list[Order] = []
    for o in orders:
        (dropped if o.instrument.symbol.upper() in held else kept).append(o)
    return kept, dropped
