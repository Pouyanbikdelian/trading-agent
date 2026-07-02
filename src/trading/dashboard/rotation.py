"""Sector-rotation analytics for the dashboard's Rotation tab.

Three lenses on "where is the money going":

* **RRG** — a Relative-Rotation-Graph approximation (JdK's exact math is
  proprietary): x = relative strength of each sector ETF vs SPY,
  normalized around 100; y = the momentum of that relative strength.
  Sectors orbit the quadrants clockwise (Improving → Leading →
  Weakening → Lagging), so weekly trails literally show money rotating.
* **Regime ribbon** — every month since 2000 classified into a simple
  investment-clock quadrant (inflation hot/cool × policy tightening/
  easing) from the FRED series econ_watch already collects. This is the
  "what led when conditions looked like this" context, not a backtest.
* **Rotation radar** — mechanical early signals: quadrant crossings in
  the last few weeks plus the fastest improver. The scout reads news;
  this reads prices. Both feed the same instinct.

``compute_rotation`` and ``classify_regimes`` are pure (frames/dicts in,
dict out) so tests stay hermetic; ``build_rotation`` does the I/O and
holds a small TTL cache so the 5-minute page refresh doesn't hammer
yfinance.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading.core.logging import logger
from trading.runtime.news_watch import SECTOR_ETFS

# RRG parameters, in weeks. 12w mean ≈ one quarter of relative strength;
# 4w momentum ≈ one month of change in that trend — the classic pairing.
_RS_WINDOW = 12
_MOM_WINDOW = 4
_TRAIL_POINTS = 13  # one quarter of weekly trail on screen
_CACHE_TTL_S = 2 * 3600.0

# Investment-clock playbook: which sectors classically lead each regime.
# Deliberately static and labeled as folklore on the page — the honest
# upgrade (regime-conditioned realized leadership) is the Phase C analog
# engine.
REGIME_PLAYBOOK: dict[str, dict[str, Any]] = {
    "hot_hiking": {
        "label": "High inflation · tightening",
        "color": "#f0556d",
        "favors": ["energy", "materials", "cons_staples", "healthcare"],
    },
    "hot_easing": {
        "label": "High inflation · easing",
        "color": "#e8a54b",
        "favors": ["energy", "materials", "industrials", "real_estate"],
    },
    "cool_hiking": {
        "label": "Cool inflation · tightening",
        "color": "#58a6ff",
        "favors": ["financials", "industrials", "tech"],
    },
    "cool_easing": {
        "label": "Cool inflation · easing",
        "color": "#3fcf8e",
        "favors": ["tech", "cons_discretionary", "communications", "real_estate"],
    },
}

_INFLATION_HOT = 3.0  # CPI YoY above this = "hot"; below = "cool"
_RATE_FLAT_PP = 0.10  # |6m fed-funds change| under this counts as "hold"


def _quadrant(x: float, y: float) -> str:
    if x >= 100 and y >= 100:
        return "leading"
    if x >= 100:
        return "weakening"
    if y >= 100:
        return "improving"
    return "lagging"


def compute_rotation(
    closes: pd.DataFrame,
    dollar_vol: dict[str, float] | None = None,
) -> dict[str, Any]:
    """RRG trails + relative momentum from a daily close matrix.

    ``closes``: columns are ETF symbols and must include ``SPY``; index
    is a DatetimeIndex. Needs roughly 15 months of history for the 12w
    mean to have room; degrades to fewer trail points otherwise.
    """
    if "SPY" not in closes.columns or len(closes) < 60:
        return {}
    weekly = closes.sort_index().resample("W-FRI").last().dropna(how="all")
    spy = weekly["SPY"]

    sectors: list[dict[str, Any]] = []
    alerts: list[dict[str, str]] = []
    for sym, name in SECTOR_ETFS.items():
        if sym not in weekly.columns:
            continue
        px = weekly[sym].dropna()
        rs = (px / spy).dropna()
        if len(rs) < _RS_WINDOW + _MOM_WINDOW + 2:
            continue
        ratio = 100.0 * rs / rs.rolling(_RS_WINDOW).mean()
        mom = 100.0 * ratio / ratio.shift(_MOM_WINDOW)
        frame = pd.DataFrame({"x": ratio, "y": mom}).dropna().iloc[-_TRAIL_POINTS:]
        if frame.empty:
            continue
        pts: list[tuple[str, float, float]] = [
            (str(ts)[:10], round(float(r.x), 3), round(float(r.y), 3)) for ts, r in frame.iterrows()
        ]
        trail = [{"t": t, "x": x, "y": y} for t, x, y in pts]
        daily = closes[sym].dropna()
        rel = (daily / closes["SPY"]).dropna()

        def _rel_ret(s: pd.Series, days: int) -> float | None:
            if len(s) <= days:
                return None
            return round(100.0 * (float(s.iloc[-1] / s.iloc[-days]) - 1.0), 2)

        quad_now = _quadrant(pts[-1][1], pts[-1][2])
        quad_then = _quadrant(pts[0][1], pts[0][2]) if len(pts) > 3 else quad_now
        # Crossing detection over the last ~3 weeks: the earliest useful
        # tell is Improving→Leading (money confirming a new leader) and
        # Leading→Weakening (money starting to leave).
        if len(pts) >= 4:
            q3 = _quadrant(pts[-4][1], pts[-4][2])
            if q3 != quad_now:
                icon = {
                    "leading": "🚀",
                    "improving": "🌱",
                    "weakening": "⚠️",
                    "lagging": "🔻",
                }[quad_now]
                alerts.append(
                    {
                        "sym": sym,
                        "name": name,
                        "kind": quad_now,
                        "msg": f"{icon} {name.replace('_', ' ')} ({sym}) crossed {q3} → {quad_now}",
                    }
                )
        sectors.append(
            {
                "sym": sym,
                "name": name,
                "trail": trail,
                "quadrant": quad_now,
                "quadrant_start": quad_then,
                "rel_1m": _rel_ret(rel, 21),
                "rel_3m": _rel_ret(rel, 63),
                "dollar_vol": (dollar_vol or {}).get(sym),
            }
        )

    # Fastest improver: steepest 4-week climb in RS-momentum among
    # not-yet-leading sectors — the "early sign" the radar exists for.
    climbers = [
        (s, s["trail"][-1]["y"] - s["trail"][-4]["y"])
        for s in sectors
        if len(s["trail"]) >= 4 and s["quadrant"] in ("improving", "lagging")
    ]
    if climbers:
        best, slope = max(climbers, key=lambda t: t[1])
        if slope > 0.5:
            alerts.append(
                {
                    "sym": str(best["sym"]),
                    "name": str(best["name"]),
                    "kind": "climber",
                    "msg": f"📈 fastest improver: {str(best['name']).replace('_', ' ')} "
                    f"({best['sym']}), RS-momentum +{slope:.1f} over 4w",
                }
            )
    return {"sectors": sectors, "alerts": alerts}


def classify_regimes(econ: dict[str, Any]) -> dict[str, Any]:
    """Monthly investment-clock regimes from econ_watch's FRED history.

    Inflation axis: CPI YoY vs 3%. Policy axis: 6-month change in the
    fed-funds rate (rising / easing; small moves count as holds and
    inherit the previous direction so the ribbon doesn't flicker).
    """
    series = econ.get("series") or {}
    cpi = {str(p["t"])[:7]: float(p["v"]) for p in (series.get("cpi_yoy") or {}).get("points", [])}
    ff = {str(p["t"])[:7]: float(p["v"]) for p in (series.get("fed_funds") or {}).get("points", [])}
    months = sorted(set(cpi) & set(ff))
    if len(months) < 8:
        return {}

    history: list[dict[str, str]] = []
    direction = "easing"  # neutral-ish seed; corrected within 6 months
    for i, m in enumerate(months):
        if i >= 6:
            chg = ff[m] - ff[months[i - 6]]
            if abs(chg) >= _RATE_FLAT_PP:
                direction = "hiking" if chg > 0 else "easing"
        infl = "hot" if cpi[m] >= _INFLATION_HOT else "cool"
        history.append({"t": m, "r": f"{infl}_{direction}"})

    cur = history[-1]
    chg6 = ff[months[-1]] - ff[months[-7]] if len(months) >= 7 else 0.0
    return {
        "history": history,
        "current": {
            "r": cur["r"],
            "label": REGIME_PLAYBOOK[cur["r"]]["label"],
            "cpi_yoy": cpi[months[-1]],
            "ff_chg_6m": round(chg6, 2),
            "since": next(
                (
                    history[j + 1]["t"]
                    for j in range(len(history) - 2, -1, -1)
                    if history[j]["r"] != cur["r"]
                ),
                history[0]["t"],
            ),
        },
        "playbook": REGIME_PLAYBOOK,
    }


# ---------------------------------------------------------------- I/O layer

_cache: dict[str, Any] = {"t": 0.0, "payload": None}


def _load_history(data_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Daily closes for SECTOR_ETFS + SPY: parquet cache first, one
    yfinance batch for whatever's missing. Also returns a 90-day average
    dollar-volume per symbol (treemap tile size = where money trades)."""
    from trading.runtime.portfolio_stats import _read_close

    want = [*SECTOR_ETFS, "SPY"]
    closes: dict[str, pd.Series] = {}
    dollar_vol: dict[str, float] = {}
    missing: list[str] = []
    for sym in want:
        s = _read_close(data_dir, sym)
        if s is not None and len(s) > 260:
            closes[sym] = s.iloc[-320:]
        else:
            missing.append(sym)
    if missing:
        import yfinance as yf

        raw = yf.download(
            " ".join(missing),
            period="15mo",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        for sym in missing:
            try:
                px = raw[sym]["Close"].dropna()
                if len(px) > 60:
                    closes[sym] = px
                    vol = raw[sym]["Volume"].dropna().iloc[-90:]
                    dollar_vol[sym] = float((px.reindex(vol.index) * vol).mean())
            except Exception:
                continue
    return pd.DataFrame(closes), dollar_vol


def build_rotation(state_dir: Path, data_dir: Path) -> dict[str, Any]:
    """Assemble the rotation payload, TTL-cached in memory: the numbers
    move daily, the page refreshes every five minutes."""
    now = time.time()
    if _cache["payload"] is not None and now - float(_cache["t"]) < _CACHE_TTL_S:
        return dict(_cache["payload"])

    out: dict[str, Any] = {"t": datetime.now(tz=timezone.utc).isoformat()}
    try:
        closes, dollar_vol = _load_history(data_dir)
        out.update(compute_rotation(closes, dollar_vol))
    except Exception as e:  # degraded tab beats a dead dashboard
        logger.bind(component="rotation").warning(f"rotation compute failed: {e}")
    try:
        ec = state_dir / "econ_watch.json"
        econ = json.loads(ec.read_text()) if ec.exists() else {}
        out["regimes"] = classify_regimes(econ)
    except Exception as e:
        logger.bind(component="rotation").warning(f"regime classify failed: {e}")
        out["regimes"] = {}
    if out.get("sectors"):
        _cache.update(t=now, payload=dict(out))
    return out
