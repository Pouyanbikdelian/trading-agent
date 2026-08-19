r"""Operator exclusions — names the system may never buy again.

``/exclude PM`` writes the symbol into ``state/exclusions.json``. From
then on nothing automatic may open or add to that position: the Agent
PM's target weights drop it like an off-universe ticker, the cycle drops
any buy touching it, and it is struck from the candidate scoreboard so
``/pick`` cannot reintroduce it. Selling is untouched — you must always
be able to get out of a name you have banned.

**Why this exists.** On 2026-08-19 an audit traced what happened to the
operator's "I don't like PM (Philip Morris), don't buy it in the future".
It had been captured as an *operator mandate*: free text, injected into
an LLM prompt, governed by a strength grader that matches ``\bbuy\b`` and
so scores "don't buy PM" identically to "buy PM" — and expiring silently
after 14 days. Nothing in code enforced it. The PM agent kept holding PM
at 7%.

The codebase already knows the distinction. ``agents/pm.py`` says it:
*"Prose the code does not back is a suggestion, and this system's whole
design premise is that limits bind."* A standing "never buy this" is a
limit. It belongs here, next to the whitelist and the caps, not in a
paragraph a model may weigh against its own conviction.

Semantics (deliberately narrow):

* Blocks every AUTOMATIC path that could open or increase the position —
  PM weights, cycle buys, candidate ranking, ``/pick``.
* Never blocks a sell, a ``/close`` or a ``/flatten``. An exclusion is
  "do not own more of this", never "you are stuck with it".
* Manual ``/buy`` BYPASSES the exclusion, exactly as it bypasses
  ``/hold``: the operator typing a buy right now outranks a preference
  they expressed in the past. The reply says the symbol is excluded so
  the contradiction is visible rather than silent.
* Excluding a symbol does NOT sell what you already hold. The bot's
  reply says so and offers ``/close``, because a one-word command that
  market-sells a position is the same mistake ``/halt``'s phantom
  flatten was.

Contrast with ``/hold``, which freezes a name in BOTH directions and is
for conviction positions you want kept. ``/exclude`` is the opposite
intent: get out when you like, never get back in automatically.

State file format::

    {"symbols": {"PM": {"reason": "...", "added_at": iso}}, "updated_at": iso}
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.file_lock import file_lock
from trading.core.types import Order, Side

FILENAME = "exclusions.json"

#: A ticker, loosely. Deliberately permissive about dots and dashes
#: (BRK.B, RDS-A) but never about whitespace or wildcards: an exclusion
#: list that can match more than one name by accident is worse than none.
_MAX_SYMBOL_LEN = 12


def normalize_symbol(raw: Any) -> str:
    """Canonicalise a ticker, raising on anything that is not one."""
    symbol = str(raw or "").strip().upper()
    if not symbol or len(symbol) > _MAX_SYMBOL_LEN:
        raise ValueError(f"not a usable ticker: {raw!r}")
    if not all(ch.isalnum() or ch in {".", "-"} for ch in symbol):
        raise ValueError(f"not a usable ticker: {raw!r}")
    return symbol


def load_exclusions(state_dir: Path | str) -> set[str]:
    """The excluded-symbol set. Missing or corrupt file = no exclusions.

    Fail-OPEN is correct here, unlike the halt file. An unreadable
    exclusion list must not stop the desk trading entirely; it degrades
    to "no preference recorded", which is the state the system was in
    before this file existed. The reader below surfaces the corruption.
    """
    try:
        return set(load_exclusion_details(state_dir))
    except Exception:
        return set()


def load_exclusion_details(state_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Excluded symbols with their reason and timestamp, for display."""
    path = Path(state_dir) / FILENAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    raw = payload.get("symbols") or {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for symbol, meta in raw.items():
            try:
                key = normalize_symbol(symbol)
            except ValueError:
                continue
            out[key] = meta if isinstance(meta, dict) else {}
    elif isinstance(raw, list):
        # Tolerate a bare list so a hand-edited file still works.
        for symbol in raw:
            try:
                out[normalize_symbol(symbol)] = {}
            except ValueError:
                continue
    return out


def save_exclusions(state_dir: Path | str, entries: dict[str, dict[str, Any]]) -> None:
    """Atomically persist the exclusion list under the cross-process lock."""
    path = Path(state_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    {
                        "symbols": {k: entries[k] for k in sorted(entries)},
                        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def add_exclusion(state_dir: Path | str, symbol: str, *, reason: str = "") -> bool:
    """Ban a symbol. Returns False if it was already excluded.

    Read-modify-write happens inside one lock so two operators (or the
    bot and the CLI) cannot lose each other's entry.
    """
    key = normalize_symbol(symbol)
    path = Path(state_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        try:
            entries = load_exclusion_details(state_dir)
        except Exception:
            entries = {}
        if key in entries:
            return False
        entries[key] = {
            "reason": reason.strip(),
            "added_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        _write_unlocked(path, entries)
    return True


def remove_exclusion(state_dir: Path | str, symbol: str) -> bool:
    """Un-ban a symbol. Returns False if it was not excluded."""
    key = normalize_symbol(symbol)
    path = Path(state_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        try:
            entries = load_exclusion_details(state_dir)
        except Exception:
            entries = {}
        if key not in entries:
            return False
        entries.pop(key)
        _write_unlocked(path, entries)
    return True


def _write_unlocked(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Publish the list while the caller owns ``file_lock(path)``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "symbols": {k: entries[k] for k in sorted(entries)},
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def filter_excluded_orders(
    orders: list[Order], excluded: set[str]
) -> tuple[list[Order], list[Order]]:
    """Split orders into (kept, dropped), dropping only exposure-INCREASING
    ones on excluded symbols.

    A sell of an excluded name is exactly what the operator wants to keep
    working — banning a stock must never trap the position. Only the buy
    side is refused. Pure function; no I/O.
    """
    if not excluded:
        return list(orders), []
    kept: list[Order] = []
    dropped: list[Order] = []
    for order in orders:
        symbol = str(getattr(order.instrument, "symbol", "")).upper()
        side = getattr(order.side, "value", order.side)
        if symbol in excluded and str(side).lower() == Side.BUY.value:
            dropped.append(order)
        else:
            kept.append(order)
    return kept, dropped
