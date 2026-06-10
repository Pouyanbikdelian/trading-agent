r"""Portfolio analytics for Telegram — beta and holdings correlation.

Everything reads the local Parquet price cache only (no network, no
broker calls), so both the daily beta line and ``/correlation`` cost a
handful of file reads. Symbols missing from the cache are skipped
rather than failing the whole report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ASSET_DIRS = ("equity", "etf")


def _read_close(data_dir: Path, symbol: str) -> pd.Series | None:
    # The cache names files after the Frequency literal "1D", but older
    # CLI fetches wrote "1d". macOS hides the difference (case-insensitive
    # filesystem); Linux does not — so try both spellings explicitly.
    for sub in _ASSET_DIRS:
        for fname in ("1D.parquet", "1d.parquet"):
            p = Path(data_dir) / sub / symbol.upper() / fname
            if p.exists():
                try:
                    s = pd.read_parquet(p)["close"].dropna()
                    s.index = pd.to_datetime(s.index)
                    return s.sort_index()
                except Exception:
                    return None
    return None


def portfolio_beta(
    position_values: dict[str, float],
    data_dir: Path,
    *,
    market: str = "SPY",
    lookback: int = 252,
) -> tuple[float, int] | None:
    """Value-weighted portfolio beta vs ``market`` over ``lookback`` days.

    ``position_values``: symbol -> current market value (sign carries
    direction for shorts). Returns (beta, names_used) or None when the
    market series or every holding is missing from the cache.
    """
    mkt = _read_close(data_dir, market)
    if mkt is None or len(mkt) < 60:
        return None
    mkt_ret = mkt.pct_change().iloc[-lookback:].dropna()
    if mkt_ret.std() == 0:
        return None

    total = sum(abs(v) for v in position_values.values())
    if total <= 0:
        return None

    beta_acc = 0.0
    weight_acc = 0.0
    used = 0
    for sym, value in position_values.items():
        s = _read_close(data_dir, sym)
        if s is None:
            continue
        ret = s.pct_change().iloc[-lookback:].dropna()
        joined = pd.concat([ret, mkt_ret], axis=1, keys=["a", "m"]).dropna()
        if len(joined) < 60:
            continue
        beta_i = float(np.cov(joined["a"], joined["m"])[0, 1] / joined["m"].var())
        w = value / total  # signed weight, normalized by gross
        beta_acc += w * beta_i
        weight_acc += abs(w)
        used += 1
    if used == 0 or weight_acc == 0:
        return None
    return beta_acc, used


def holdings_correlation(
    symbols: list[str], data_dir: Path, *, lookback: int = 252
) -> pd.DataFrame | None:
    """Pairwise return correlation of ``symbols`` over ``lookback`` days."""
    series = {}
    for sym in symbols:
        s = _read_close(data_dir, sym)
        if s is not None and len(s) > 60:
            series[sym.upper()] = s.pct_change()
    if len(series) < 2:
        return None
    rets = pd.DataFrame(series).iloc[-lookback:].dropna(how="all")
    return rets.corr()


def format_correlation(corr: pd.DataFrame, *, max_matrix: int = 10) -> str:
    """Telegram-friendly rendering: compact monospace matrix for small
    books, ranked pair list for big ones. Values as integer percent."""
    syms = list(corr.columns)
    # Average pairwise correlation (off-diagonal).
    n = len(syms)
    off = corr.values[np.triu_indices(n, k=1)]
    avg = float(np.nanmean(off)) if len(off) else 0.0

    lines = [f"🔗 *Holdings correlation* — trailing 12m, {n} names", ""]
    if n <= max_matrix:
        tag = {s: s[:4] for s in syms}
        lines.append("```")
        lines.append("     " + " ".join(f"{tag[s]:>4}" for s in syms))
        for s in syms:
            cells = []
            for t in syms:
                v = corr.loc[s, t]
                cells.append("   ." if s == t else f"{v * 100:>4.0f}")
            lines.append(f"{tag[s]:<5}" + " ".join(cells))
        lines.append("```")
    pairs = [
        (syms[i], syms[j], float(corr.iloc[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if not np.isnan(corr.iloc[i, j])
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    if pairs:
        hi = pairs[0]
        lo = pairs[-1]
        lines.append(f"Average pairwise: `{avg * 100:+.0f}%`")
        lines.append(f"Most correlated:  `{hi[0]}–{hi[1]}  {hi[2] * 100:+.0f}%`")
        lines.append(f"Least correlated: `{lo[0]}–{lo[1]}  {lo[2] * 100:+.0f}%`")
    if avg > 0.7:
        lines.append("_⚠️ High average correlation — this book moves as one trade._")
    return "\n".join(lines)
