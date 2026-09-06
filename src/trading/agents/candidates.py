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
import json
import os
import tempfile
from datetime import datetime, timezone
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

# The live runner writes this after it has applied the active playbook,
# screens, runtime parameter overrides and current price refresh.  It is the
# one authoritative scoreboard for an agent cycle; rebuilding a second one
# from environment defaults is how the desk ended up debating a ladder that
# did not necessarily match the executable strategy.
CANDIDATE_SNAPSHOT_FILE = "candidate_snapshot.json"
SNAPSHOT_SCHEMA_VERSION = 1


def candidate_snapshot_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / CANDIDATE_SNAPSHOT_FILE


def _iso_date(value: Any) -> str:
    try:
        return str(value.date())
    except Exception:
        return str(value)[:10]


def _age_days(last_bar: Any) -> int | None:
    try:
        import pandas as pd

        return int((pd.Timestamp.now(tz="UTC").normalize() - last_bar.normalize()).days)
    except Exception:
        return None


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Replace a snapshot as one operation: agents see old or new, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, default=str, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


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


def _correlation_review(
    prices: Any, *, selected: set[str], ranked: list[dict[str, Any]], window: int = 63
) -> dict[str, Any] | None:
    """Compact, advisory correlation facts for the PM and committee.

    This does not decide a portfolio.  It identifies concentration that a
    sector label cannot see and makes accepting a correlated exception a
    deliberate, explainable act rather than a surprise in the next drawdown.
    """
    try:
        import numpy as np

        cols = [str(c) for c in prices.columns]
        selected_cols = [s for s in sorted(selected) if s in cols]
        rets = prices.pct_change().iloc[-window:].dropna(how="all")
        if len(rets) < 20:
            return None
        corr = rets.corr()
        out: dict[str, Any] = {"window_bars": len(rets)}
        if len(selected_cols) >= 2:
            sub = corr.loc[selected_cols, selected_cols]
            values = sub.to_numpy()
            off = values[~np.eye(len(selected_cols), dtype=bool)]
            eig = np.linalg.eigvalsh(values)
            eig = eig[eig > 1e-9]
            out.update(
                {
                    "selected_names": selected_cols,
                    "avg_pairwise_corr": round(float(off.mean()), 2) if off.size else 0.0,
                    "effective_bets": round(float(eig.sum() ** 2 / np.square(eig).sum()), 1)
                    if eig.size
                    else float(len(selected_cols)),
                }
            )
            pairs: list[dict[str, Any]] = []
            for i, left in enumerate(selected_cols):
                for right in selected_cols[i + 1 :]:
                    value = float(sub.loc[left, right])
                    if abs(value) >= 0.8:
                        pairs.append({"left": left, "right": right, "corr": round(value, 2)})
            if pairs:
                out["high_corr_pairs"] = sorted(
                    pairs, key=lambda row: abs(float(row["corr"])), reverse=True
                )[:8]
        for row in ranked:
            symbol = str(row["symbol"])
            peers = [s for s in selected_cols if s != symbol]
            if symbol in corr.index and peers:
                max_corr = max(abs(float(corr.loc[symbol, peer])) for peer in peers)
                row["max_abs_corr_to_selected"] = round(max_corr, 2)
                if max_corr >= 0.8:
                    row["correlation_warning"] = "highly correlated with the selected basket"
        return out
    except Exception:
        return None


def build_runner_candidate_snapshot(
    prices: Any,
    *,
    ranked: list[tuple[str, float]],
    selected: set[str],
    config: Any,
    generated_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Build the runner's authoritative candidate snapshot from its own inputs.

    It deliberately accepts the resolved ``RunnerConfig`` rather than reading
    environment variables.  A playbook can change universe, screens or
    strategy parameters at runtime; the exact configuration below is the
    immutable provenance needed to later judge whether repeated names were
    deserved.
    """
    if prices is None or getattr(prices, "empty", True) or not ranked:
        return None
    last_bar = prices.index[-1]
    rows: list[dict[str, Any]] = []
    for rank, (raw_symbol, score) in enumerate(ranked, start=1):
        symbol = str(raw_symbol).split(":")[-1].upper()
        row: dict[str, Any] = {"rank": rank, "symbol": symbol, "score": round(float(score), 4)}
        if symbol in prices.columns:
            percentile = _pctile_52w(prices[symbol])
            if percentile is not None:
                row["pctile_52w"] = percentile
        rows.append(row)
    age = _age_days(last_bar)
    out: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": "runner_snapshot",
        "generated_at": (generated_at or datetime.now(tz=timezone.utc)).isoformat(),
        "as_of": _iso_date(last_bar),
        "age_days": age,
        "bars": int(prices.shape[0]),
        "universe_size": int(prices.shape[1]),
        "strategy": "+".join(getattr(config, "strategies", []) or []),
        "universe": str(getattr(config, "universe", "")),
        "frequency": str(getattr(config, "freq", "")),
        "strategy_params": dict(getattr(config, "strategy_params", {}) or {}),
        "screens_active": bool(getattr(config, "screens", None)),
        "score_units": (
            "strategy-specific (formation return for momentum); comparable "
            "within this snapshot, not across strategies"
        ),
        "ranked": rows,
        "selected_mechanical": sorted(selected),
    }
    review = _correlation_review(prices, selected=selected, ranked=rows)
    if review:
        out["correlation_review"] = review
    if age is not None and age > STALE_LADDER_DAYS:
        out["staleness_warning"] = (
            f"last bar is {age} days old — no new agent/PM entries may rely on this snapshot"
        )
    return out


def write_runner_candidate_snapshot(state_dir: Path | str, snapshot: dict[str, Any]) -> Path:
    """Persist an authoritative, atomic read-only input for the agents."""
    path = candidate_snapshot_path(state_dir)
    _atomic_json_write(path, snapshot)
    return path


def load_runner_candidate_snapshot(state_dir: Path | str) -> dict[str, Any] | None:
    """Load a validated runner snapshot, never an arbitrary JSON blob."""
    path = candidate_snapshot_path(state_dir)
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        return None
    ranked = payload.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        return None
    if any(not isinstance(row, dict) or not row.get("symbol") for row in ranked):
        return None
    return payload


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

    universe = str(universe or os.getenv("UNIVERSE", "sp500") or "sp500")
    strategy = str(strategy or os.getenv("STRATEGY", "top_k_momentum") or "top_k_momentum")

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
    age_days = _age_days(last_bar)

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
