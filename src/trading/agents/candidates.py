"""Ranked candidate ladder — the agents' only source of NEW names.

Why this module exists (found 2026-08-05): the agent context carried the
book, the macro dial, headlines and lessons, but no ranked universe. The
only tickers named anywhere in the prompt were the ones already held,
plus whatever exemplars the charters happened to spell out. An LLM asked
to allocate under those conditions does the only thing it can — it
free-associates names, and free association is stable week over week. The
simulated PM held the same handful of names for a month while the charter
told it, truthfully but uselessly, that it could buy any of ~1,600 index
constituents.

So: hand the agents the same scoreboard the live strategy computes for
itself. ``/signal`` already renders it for the operator; this is the same
computation, reachable from ``build_context``.

Network-free by construction — reads the Parquet cache only, same as
every other part of the context builder. A missing cache degrades to
``None`` (agents then reason without a ladder, as they did before), never
to an exception: an idea feed is a strong nice-to-have, never a reason to
lose the whole cycle.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from trading.core.logging import logger

# 25 is a deliberate compromise. Fewer than ~15 and the ladder is just the
# book's neighbours; more than ~30 and it crowds out market context in the
# prompt budget while adding names no sane allocator reaches anyway.
DEFAULT_TOP_N = 25

# Below this many bars a momentum score is noise dressed as a ranking.
MIN_BARS = 120

# Calendar days of staleness before the ladder says so in the prompt.
# Four covers a normal weekend plus a public holiday; beyond that, bars
# are genuinely missing rather than merely un-traded.
STALE_LADDER_DAYS = 4


def _pctile_52w(series: Any) -> float | None:
    """Where price sits in its own 52-week range, 0=low 1=high.

    Carried alongside the score because the quant charter's first hard
    rule is that a high percentile tells you WHERE price is, not whether
    reward-to-risk is good. A ladder of scores without it invites exactly
    the top-ticking the charter warns against.
    """
    try:
        yr = series.iloc[-252:]
        lo, hi = float(yr.min()), float(yr.max())
        if hi <= lo:
            return None
        return round((float(series.iloc[-1]) - lo) / (hi - lo), 2)
    except Exception:
        return None


def build_candidate_ladder(
    data_dir: Path,
    *,
    top_n: int = DEFAULT_TOP_N,
    universe: str | None = None,
    strategy: str | None = None,
) -> dict[str, Any] | None:
    """The live strategy's ranked scoreboard, as of the latest cached bar.

    ``data_dir`` is the PARQUET ROOT — ``settings.data_dir``, which already
    points at ``data/parquet``, not at ``data``. Passing the repo's ``data``
    directory finds nothing and returns ``None``.

    Mirrors the runner's own configuration (``UNIVERSE`` / ``STRATEGY`` /
    ``REBALANCE`` env) so the agents rank what the system ranks — a ladder
    computed from a different strategy than the one trading would be a
    second opinion masquerading as the house view.

    Returns ``None`` when there is no usable cache, no such universe, or
    the strategy has no natural ranking (risk-parity, pairs). Callers
    treat ``None`` as "no ladder this cycle", not as an error.
    """
    import pandas as pd

    universe = universe or os.getenv("UNIVERSE", "sp500")
    strategy = strategy or os.getenv("STRATEGY", "top_k_momentum")

    try:
        from trading.core.universes import load_universe

        symbols = [i.symbol for i in load_universe(universe)]
    except Exception as e:
        logger.bind(component="agents").warning(f"candidate ladder: universe {universe!r}: {e}")
        return None

    try:
        # Deliberately NOT ParquetCache.read(): that needs an AssetClass to
        # build the path, and this universe mixes equities with ETFs, which
        # the cache files under a different directory. ``_read_close`` is
        # the house helper for exactly this — it tries both asset dirs AND
        # both "1D"/"1d" spellings, because macOS hides the case difference
        # and the Linux VPS does not. Guessing EQUITY/"1D" here silently
        # returned an empty ladder, which is the one failure mode this
        # module exists to prevent.
        from trading.runtime.portfolio_stats import _read_close

        series: dict[str, Any] = {}
        for sym in symbols:
            closes = _read_close(data_dir, sym)
            if closes is not None and len(closes) >= MIN_BARS:
                series[sym] = closes
        if not series:
            logger.bind(component="agents").warning(
                f"candidate ladder: no cached prices for universe {universe!r}"
            )
            return None
        prices = pd.concat(series, axis=1).ffill().dropna(how="all")
    except Exception as e:
        logger.bind(component="agents").warning(f"candidate ladder: price load failed: {e}")
        return None

    try:
        from trading.strategies.base import get_strategy

        cls = get_strategy(strategy)
        kwargs: dict[str, Any] = {}
        rebal = os.getenv("REBALANCE")
        if rebal:
            with contextlib.suppress(ValueError):
                kwargs["rebalance"] = int(rebal)
        ranked = cls(cls.Params(**kwargs)).top_candidates(prices, top_n=top_n)
    except Exception as e:
        logger.bind(component="agents").warning(f"candidate ladder: {strategy!r} rank failed: {e}")
        return None

    if not ranked:
        return None

    rows: list[dict[str, Any]] = []
    for rank, (sym, score) in enumerate(ranked, start=1):
        row: dict[str, Any] = {"rank": rank, "symbol": sym, "score": round(float(score), 4)}
        pct = _pctile_52w(series[sym]) if sym in series else None
        if pct is not None:
            row["pctile_52w"] = pct
        rows.append(row)

    # Staleness, stated rather than implied. Nothing on the box refreshes
    # the parquet cache on a schedule: it updates as a side effect of the
    # trading cycle, whose refresh loop logs a warning and falls back to
    # disk whenever a fetch times out. So a ladder can rank week-old
    # momentum and look exactly like a fresh one. Say the age out loud and
    # let the PM discount it.
    last_bar = prices.index[-1]
    age_days: int | None = None
    with contextlib.suppress(Exception):
        age_days = (pd.Timestamp.now(tz="UTC").normalize() - last_bar.normalize()).days

    out: dict[str, Any] = {
        "strategy": strategy,
        "universe": universe,
        "as_of": str(getattr(last_bar, "date", lambda: last_bar)()),
        "bars": int(prices.shape[0]),
        "score_units": (
            "strategy-specific (formation return for momentum); comparable "
            "across rows, not across strategies"
        ),
        "ranked": rows,
    }
    if age_days is not None:
        out["age_days"] = age_days
        if age_days > STALE_LADDER_DAYS:
            out["staleness_warning"] = (
                f"last bar is {age_days} days old — these ranks may not "
                "reflect the current tape; weight them accordingly"
            )
            logger.bind(component="agents").warning(
                f"candidate ladder is {age_days}d stale (last bar {out['as_of']})"
            )
    return out
