"""Market watch — the dashboard's macro instrument panel.

Daily collector for the indicators a swing trader actually watches
(selection grounded in docs/research_macro_leadlag.md + desk practice):

* ``curve_10y3m``   — 10y minus 3m Treasury yield (Fed-preferred
                      recession spread; the re-steepening after an
                      inversion is the historically dangerous phase).
* ``vix`` / ``vix3m`` / ``vix_ratio`` — VIX term structure. Ratio > 1
                      (backwardation) = genuine stress, not noise.
* ``pct_above_50`` / ``pct_above_200`` — breadth of the S&P universe
                      from OUR OWN price cache (no network): index up
                      while breadth falls is the classic top warning.
* ``hyg_ief``       — credit vs treasuries (credit cracks first).
* ``spy_tlt`` / ``qqq_spy`` / ``gld_dbc`` — risk-appetite ratios.

History is appended to ``state/market_watch.json`` (bounded), so the
dashboard can draw trend lines, and a year from now the committee can
ask "what did breadth look like last time the curve did this".
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

STATE_FILENAME = "market_watch.json"
HISTORY_KEEP = 500

_YF_TICKERS = {"^IRX": "y_3m", "^TNX": "y_10y", "^VIX": "vix", "^VIX3M": "vix3m"}


def compute_breadth(data_dir: Path, max_names: int = 600) -> dict[str, float | None]:
    """% of cached equity names above their 50/200-day SMAs. Local only."""
    import pandas as pd

    above50 = above200 = total = 0
    eq_dir = Path(data_dir) / "equity"
    if not eq_dir.exists():
        return {"pct_above_50": None, "pct_above_200": None}
    for i, p in enumerate(sorted(eq_dir.glob("*/1[dD].parquet"))):
        if i >= max_names:
            break
        try:
            s = pd.read_parquet(p, columns=["close"])["close"].dropna()
            if len(s) < 200:
                continue
            last = float(s.iloc[-1])
            total += 1
            if last > float(s.iloc[-50:].mean()):
                above50 += 1
            if last > float(s.iloc[-200:].mean()):
                above200 += 1
        except Exception:
            continue
    if total < 50:
        return {"pct_above_50": None, "pct_above_200": None}
    return {"pct_above_50": above50 / total, "pct_above_200": above200 / total}


def compute_ratios(data_dir: Path) -> dict[str, float | None]:
    """Risk-appetite ratios from the ETF cache, normalized to a 1y base
    of 100 so the dashboard lines are comparable."""
    from trading.runtime.portfolio_stats import _read_close

    out: dict[str, float | None] = {}
    pairs = {
        "spy_tlt": ("SPY", "TLT"),
        "qqq_spy": ("QQQ", "SPY"),
        "gld_dbc": ("GLD", "DBC"),
        "hyg_ief": ("HYG", "IEF"),
    }
    for name, (a, b) in pairs.items():
        sa, sb = _read_close(data_dir, a), _read_close(data_dir, b)
        if sa is None or sb is None:
            out[name] = None
            continue
        ratio = (sa / sb).dropna()
        if len(ratio) < 252:
            out[name] = None
            continue
        base = float(ratio.iloc[-252:].iloc[0])
        out[name] = round(float(ratio.iloc[-1]) / base * 100.0, 2) if base else None
    return out


def fetch_rates_vix() -> dict[str, float | None]:
    """Yields + VIX term structure via yfinance. None-tolerant."""
    out: dict[str, float | None] = dict.fromkeys(_YF_TICKERS.values())
    try:
        import yfinance as yf

        raw = yf.download(
            " ".join(_YF_TICKERS),
            period="10d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        for tkr, name in _YF_TICKERS.items():
            try:
                out[name] = float(raw[tkr]["Close"].dropna().iloc[-1])
            except Exception:
                continue
    except Exception as e:
        logger.bind(component="market_watch").info(f"rates/vix fetch failed: {e}")
    return out


def assemble_reading(data_dir: Path) -> dict[str, Any]:
    r = fetch_rates_vix()
    reading: dict[str, Any] = {
        "t": datetime.now(tz=timezone.utc).isoformat(),
        **r,
        **compute_breadth(data_dir),
        **compute_ratios(data_dir),
    }
    if r.get("y_10y") is not None and r.get("y_3m") is not None:
        # ^TNX and ^IRX both quote in percent already (e.g. 4.32).
        reading["curve_10y3m"] = round(r["y_10y"] - r["y_3m"], 3)
    else:
        reading["curve_10y3m"] = None
    if r.get("vix") and r.get("vix3m"):
        reading["vix_ratio"] = round(r["vix"] / r["vix3m"], 3)
    else:
        reading["vix_ratio"] = None
    return reading


def collect(state_dir: Path, data_dir: Path) -> dict[str, Any]:
    """One collection pass: append today's reading to bounded history."""
    path = Path(state_dir) / STATE_FILENAME
    history: list[dict[str, Any]] = []
    try:
        if path.exists():
            history = json.loads(path.read_text()).get("history", [])
    except Exception:
        history = []
    reading = assemble_reading(data_dir)
    # Replace same-day reading instead of duplicating (idempotent re-runs).
    today = reading["t"][:10]
    history = [h for h in history if str(h.get("t", ""))[:10] != today]
    history.append(reading)
    history = history[-HISTORY_KEEP:]

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump({"history": history, "latest": reading}, f)
    os.replace(tmp, path)
    logger.bind(component="market_watch").info(f"market watch updated ({len(history)} readings)")
    return reading
