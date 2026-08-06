"""Index constituents from Wikipedia — the importable half.

Lives in ``src/`` rather than ``scripts/`` for one blunt reason: the
Docker image copies ``src``, ``config`` and ``docker``, and **not**
``scripts``. A first cut had the runner's weekly refresh job shell out to
``/app/scripts/refresh_universes.py``, which does not exist inside the
container — so the job would have raised ``FileNotFoundError`` every
Sunday, been swallowed by the surrounding ``except``, logged one line,
and left the universe quietly frozen. That is precisely the failure mode
the 2026-08-06 sweep existed to eliminate, reintroduced by the fix for it.

``scripts/refresh_universes.py`` is now a thin CLI wrapper over this
module, and the runner imports it directly — no subprocess, which also
avoids loading a second pandas into a 2 GB box.

Wikipedia is the de-facto reference for these indices: crowd-maintained
and usually updated within hours of a rebalance. The free alternative to
a paid index feed.
"""

from __future__ import annotations

import datetime as _dt
import io
import os
import urllib.request
from pathlib import Path
from typing import Any

from trading.core.logging import logger

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
R1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

# Wikipedia blocks requests without a UA and asks scripts to identify
# themselves with a project name and a contact route.
USER_AGENT = (
    "trading-agent/0.1 (universe-refresh; https://github.com/Pouyanbikdelian/trading-agent)"
)

# Sanity floors. Better to refuse than to ship a five-symbol "S&P 500"
# because Wikipedia renamed a table column.
MIN_SP500 = 400
MIN_NDX = 80
MIN_R1000 = 700


def out_path() -> Path:
    """Where the generated file goes: ``state/``, a writable volume.

    NOT ``config/`` — that is bind-mounted read-only so the running
    system cannot mutate operator YAML. Right call, wrong home for
    machine-written data: it left the constituents two months stale
    because the only place they could be written could not be written to.
    Override with ``UNIVERSES_GENERATED_PATH``.
    """
    override = os.getenv("UNIVERSES_GENERATED_PATH")
    if override:
        return Path(override)
    from trading.core.config import settings

    return Path(settings.state_dir) / "universes.generated.yaml"


def _fetch_html(url: str, *, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _normalize(symbols: list[str]) -> list[str]:
    """Vendor share-class separators → yfinance dashes (BRK.B → BRK-B),
    de-duplicated. The IBKR adapter handles the dash form for US equities."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        s = str(raw).strip().upper().replace(".", "-")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def fetch_sp500() -> list[str]:
    import pandas as pd

    df = pd.read_html(io.StringIO(_fetch_html(SP500_URL)))[0]
    # The column has been "Symbol" or "Ticker" depending on when you
    # scrape it; match by prefix to survive the next rename.
    col = next(c for c in df.columns if str(c).lower().startswith(("sym", "tick")))
    return _normalize(df[col].astype(str).tolist())


def fetch_nasdaq100() -> list[str]:
    import pandas as pd

    for df in pd.read_html(io.StringIO(_fetch_html(NDX_URL))):
        lowered = {str(c).lower() for c in df.columns}
        if lowered & {"ticker", "symbol"}:
            col = next(c for c in df.columns if str(c).lower() in ("ticker", "symbol"))
            return _normalize(df[col].astype(str).tolist())
    raise RuntimeError("could not find the NASDAQ-100 constituent table")


def fetch_russell1000() -> list[str]:
    import pandas as pd

    best: list[str] = []
    for df in pd.read_html(io.StringIO(_fetch_html(R1000_URL))):
        for c in df.columns:
            if str(c).lower().startswith(("sym", "tick")):
                syms = _normalize(df[c].astype(str).tolist())
                if len(syms) > len(best):
                    best = syms
    if not best:
        raise RuntimeError("could not find the Russell 1000 component table")
    return best


def refresh(path: Path | None = None) -> dict[str, Any]:
    """Fetch, sanity-check and write. Returns a result dict; never raises.

    ``{"ok": bool, "counts": {...}, "path": str, "reason": str|None}``.
    Russell 1000 is best-effort — its Wikipedia page is less rigorously
    maintained, and its absence must not block the primary indices.
    """
    import yaml

    try:
        sp500 = fetch_sp500()
        ndx = fetch_nasdaq100()
    except Exception as e:
        logger.bind(component="data").warning(f"universe refresh: primary fetch failed: {e!r}")
        return {"ok": False, "reason": f"fetch failed: {e!r}", "counts": {}}

    if len(sp500) < MIN_SP500 or len(ndx) < MIN_NDX:
        reason = f"refusing to write: sp500={len(sp500)}, nasdaq100={len(ndx)}"
        logger.bind(component="data").warning(f"universe refresh: {reason}")
        return {
            "ok": False,
            "reason": reason,
            "counts": {"sp500": len(sp500), "nasdaq100": len(ndx)},
        }

    try:
        r1000 = fetch_russell1000()
        if len(r1000) < MIN_R1000:
            r1000 = []
    except Exception as e:
        logger.bind(component="data").info(f"universe refresh: russell1000 skipped ({e!r})")
        r1000 = []

    universes: dict[str, Any] = {
        "sp500": {
            "asset_class": "equity",
            "description": f"S&P 500 from Wikipedia ({len(sp500)} names).",
            "symbols": sp500,
        },
        "nasdaq100": {
            "asset_class": "equity",
            "description": f"NASDAQ-100 from Wikipedia ({len(ndx)} names).",
            "symbols": ndx,
        },
    }
    if r1000:
        universes["russell1000"] = {
            "asset_class": "equity",
            "description": f"Russell 1000 from Wikipedia ({len(r1000)} names).",
            "symbols": r1000,
        }

    target = Path(path) if path else out_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# AUTO-GENERATED — do not hand-edit. Written by "
        "trading.data.universe_refresh (weekly job / scripts/refresh_universes.py).\n"
        "# Hand-curated entries in config/universes.yaml WIN over anything here.\n"
        + yaml.safe_dump(
            {
                "_generated_by": "trading.data.universe_refresh",
                "_generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                "universes": universes,
            },
            sort_keys=False,
            default_flow_style=False,
        )
    )
    counts = {"sp500": len(sp500), "nasdaq100": len(ndx), "russell1000": len(r1000)}
    logger.bind(component="data").info(f"universe refresh wrote {target}: {counts}")
    return {"ok": True, "counts": counts, "path": str(target), "reason": None}
