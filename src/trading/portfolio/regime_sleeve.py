"""The regime-switch ETF sleeve's merge into the cycle's target book.

Third independent sleeve knob, alongside STRATEGY_SLEEVE_PCT and
AGENT_PM_SLEEVE_PCT: ``REGIME_SLEEVE_PCT`` is the fraction of the managed
account the regime-switch allocator directs (strategies/regime_switch.py).
0.0 (the default) is off: no keys are added, so deploying this code changes
nothing until the operator sets the knob.

Merging follows the PM bridge's rules (cycle._merge_pm_signal):
* weights are fractions of the sleeve's own book, scaled by the sleeve;
* overlapping names ADD (two books wanting SPY hold the sum);
* every symbol the sleeve can hold gets a key, zero included, so a leg the
  allocator drops is actually sold — a missing key would leave it held
  forever. Pinned symbols (/hold) are protected downstream: the cycle drops
  orders on them at step 8a.

It still reaches the broker only through the risk manager and the
operator's basket approval, like every other book.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_FILENAME = "regime_sleeve.json"
MAX_STALENESS_DAYS = 5


def merge_regime_targets(
    target_weights: dict[str, float],
    sleeve_weights: dict[str, float],
    sleeve_pct: float,
    *,
    universe: tuple[str, ...],
    key_for: Callable[[str], str],
) -> dict[str, float]:
    """Add the sleeve (scaled) into the book. Never mutates the input."""
    merged = dict(target_weights or {})
    if sleeve_pct <= 0:
        return merged
    for sym in universe:
        w = float(sleeve_weights.get(sym, 0.0)) * sleeve_pct
        k = key_for(sym)
        merged[k] = float(merged.get(k, 0.0)) + w
    return merged


def key_resolver(instruments_by_key: dict[str, Any]) -> Callable[[str], str]:
    """Map a ticker to the cycle's instrument key; ETFs default to ``etf:SYM``."""
    by_sym: dict[str, str] = {}
    for k in instruments_by_key or {}:
        by_sym.setdefault(k.split(":")[-1].upper(), k)
    return lambda sym: by_sym.get(sym.upper(), f"etf:{sym.upper()}")


def check_fresh(asof: str, now: datetime) -> None:
    """Refuse an allocation computed on a stale price cache."""
    age = (now.date() - datetime.fromisoformat(asof).date()).days
    if age > MAX_STALENESS_DAYS:
        raise ValueError(f"regime sleeve prices are {age} days old (as of {asof})")


def write_state(state_dir: Path, payload: dict[str, Any]) -> None:
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, path)


def read_state(state_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((Path(state_dir) / STATE_FILENAME).read_text())
    except Exception:
        return {}


_REGIME_LABEL = {
    "bull/calm": "🟢 bull · calm",
    "bull/stressed": "🟡 bull · volatile",
    "bear": "🔴 bear",
}
_LEG_LABEL = {
    "qqq_trend": "QQQ trend",
    "risk_parity": "risk parity",
    "gtaa5": "GTAA-5",
    "donchian": "Donchian",
    "min_variance": "min-variance",
}


def format_note(state: dict[str, Any]) -> str:
    """One compact block for the cycle card and /sleeve."""
    if not state:
        return "🧭 Regime sleeve: no allocation yet — it is computed at the next cycle."
    pct = float(state.get("sleeve_pct", 0.0))
    eff = float(state.get("effective_pct", pct))
    legs = " + ".join(
        f"{_LEG_LABEL.get(k, k)} {v:.0%}" for k, v in (state.get("legs") or {}).items()
    )
    ws = sorted((state.get("weights") or {}).items(), key=lambda kv: -kv[1])
    book = ", ".join(f"{s} {w * eff:.1%}" for s, w in ws) or "all cash"
    mode = state.get("mode")
    mode_txt = f" (mode {mode}: scaled to {eff:.0%})" if mode and eff != pct else ""
    return (
        f"🧭 Regime sleeve {pct:.0%}{mode_txt} — {_REGIME_LABEL.get(state.get('regime'), '?')}\n"
        f"   legs: {legs}\n"
        f"   of account: {book}\n"
        f"   prices as of {state.get('asof')}"
    )
