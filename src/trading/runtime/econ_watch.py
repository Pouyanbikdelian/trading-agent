"""Economy watch — slow macro series from FRED, no API key required.

FRED's ``fredgraph.csv`` endpoint serves any public series as plain CSV
(``DATE,VALUE``) with no auth — the free tier that never expires. We pull
a small fixed set covering inflation, housing, labor, credit/liquidity
and the consumer, compute YoY transforms where the level is meaningless
(CPI at 320 says nothing; +3.1% YoY does), and persist a trimmed history
to ``state/econ_watch.json`` for the dashboard's Economy tab plus a
compact latest-readings block for the agent context.

These series move monthly/weekly; one collection per weekday is already
generous. Failures degrade per-series, never break the pass.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

STATE_FILENAME = "econ_watch.json"
TIMEOUT_S = 30.0
_START = "2019-01-01"  # ~6y of history is plenty for the charts
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# FRED/Cloudflare deprioritizes anonymous clients; identify ourselves.
_HEADERS = {"User-Agent": "trading-agent/0.1 (econ watch; contact: podibiki@gmail.com)"}

# key -> (FRED series id, label, transform, unit)
# transform: "level" as-is | "yoy" percent change vs 12 months prior
#            | "k" thousands | "tn" trillions
SERIES: dict[str, tuple[str, str, str, str]] = {
    "cpi_yoy": ("CPIAUCSL", "CPI", "yoy", "%"),
    "core_cpi_yoy": ("CPILFESL", "Core CPI", "yoy", "%"),
    "breakeven_10y": ("T10YIE", "10y breakeven", "level", "%"),
    "mortgage_30y": ("MORTGAGE30US", "30y mortgage", "level", "%"),
    "housing_starts": ("HOUST", "Housing starts", "k", "k"),
    "case_shiller_yoy": ("CSUSHPINSA", "Case-Shiller", "yoy", "%"),
    "unemployment": ("UNRATE", "Unemployment", "level", "%"),
    "claims": ("ICSA", "Initial claims", "div1k", "k"),
    "hy_oas": ("BAMLH0A0HYM2", "HY OAS", "level", "%"),
    "curve_2s10s": ("T10Y2Y", "2s10s curve", "level", "pp"),
    "fed_bs": ("WALCL", "Fed balance sheet", "tn", "$tn"),
    "retail_yoy": ("RSAFS", "Retail sales", "yoy", "%"),
    "sentiment": ("UMCSENT", "UMich sentiment", "level", "idx"),
}


def _parse_csv(text: str) -> list[tuple[str, float]]:
    """``DATE,VALUE`` rows; FRED writes '.' for missing observations."""
    out: list[tuple[str, float]] = []
    for ln in text.strip().splitlines()[1:]:
        parts = ln.split(",")
        if len(parts) != 2 or parts[1] in (".", ""):
            continue
        try:
            out.append((parts[0][:10], float(parts[1])))
        except ValueError:
            continue
    return out


def _transform(rows: list[tuple[str, float]], how: str) -> list[dict[str, float | str]]:
    if how == "yoy":
        by_date = dict(rows)
        out = []
        for t, v in rows:
            prior = f"{int(t[:4]) - 1}{t[4:]}"
            if by_date.get(prior):
                out.append({"t": t, "v": round((v / by_date[prior] - 1.0) * 100, 2)})
        return out
    # HOUST is already thousands; ICSA is raw counts; WALCL is $ millions.
    scale = {"k": 1.0, "div1k": 1e-3, "tn": 1e-6, "level": 1.0}[how]
    return [{"t": t, "v": round(v * scale, 3)} for t, v in rows]


def _monthly(pts: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Resample to month-end (last observation wins). Mixing daily, weekly
    and monthly series on one chart axis is unreadable; a uniform monthly
    grid is what every pro macro dashboard actually plots."""
    by_month: dict[str, float] = {}
    for p in pts:
        by_month[str(p["t"])[:7]] = float(p["v"])
    return [{"t": m, "v": v} for m, v in sorted(by_month.items())]


def fetch_series(series_id: str, how: str, client: Any = None) -> list[dict[str, float | str]]:
    import httpx

    # YoY needs 12 months of runway before _START — one wider fetch beats two.
    start = "2018-01-01" if how == "yoy" else _START
    getter = client or httpx
    last_err: Exception | None = None
    for _attempt in range(2):  # FRED is occasionally slow; one retry
        try:
            resp = getter.get(
                _FRED_URL,
                params={"id": series_id, "cosd": start},
                headers=_HEADERS,
                timeout=TIMEOUT_S,
                follow_redirects=True,
            )
            resp.raise_for_status()
            pts = _transform(_parse_csv(resp.text), how)
            return _monthly([p for p in pts if str(p["t"]) >= _START])[-90:]
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def collect(state_dir: Path) -> dict[str, Any]:
    """One pass over all series; atomic write; per-series degradation.
    One keep-alive connection for the whole pass — 13 cold TLS handshakes
    to a slow CDN is how the first version timed out."""
    import httpx

    series: dict[str, Any] = {}
    with httpx.Client(follow_redirects=True, timeout=TIMEOUT_S) as client:
        for key, (sid, label, how, unit) in SERIES.items():
            try:
                pts = fetch_series(sid, how, client=client)
                if pts:
                    series[key] = {
                        "label": label,
                        "unit": unit,
                        "latest": pts[-1]["v"],
                        "points": pts,
                    }
            except Exception as e:
                logger.bind(component="econ_watch").info(f"{key} ({sid}) failed: {e}")
    reading = {"t": datetime.now(tz=timezone.utc).isoformat(), "series": series}
    path = Path(state_dir) / STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(reading, f)
    os.replace(tmp, path)
    logger.bind(component="econ_watch").info(f"econ watch updated ({len(series)} series)")
    return reading


def latest_block(state_dir: Path, *, max_age_hours: float = 80.0) -> dict[str, Any]:
    """Compact {key: latest} for the agent context. {} when stale/absent —
    agents must never reason over a dead economy snapshot."""
    path = Path(state_dir) / STATE_FILENAME
    try:
        reading = json.loads(path.read_text())
        ts = datetime.fromisoformat(reading["t"])
        if (datetime.now(tz=timezone.utc) - ts).total_seconds() / 3600 > max_age_hours:
            return {}
        return {
            k: {"v": s["latest"], "unit": s["unit"], "label": s["label"]}
            for k, s in reading.get("series", {}).items()
        }
    except Exception:
        return {}
