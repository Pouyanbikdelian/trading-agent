"""Refresh S&P 500 + NASDAQ-100 universe constituents from Wikipedia.

Wikipedia is the de-facto reference for these indices: crowd-maintained and
typically updated within hours of an actual rebalance. The free alternative
to a paid index-data feed.

Output: writes ``config/universes.generated.yaml`` with two universes:

* ``sp500``      — current S&P 500 constituents.
* ``nasdaq100``  — current NASDAQ-100 constituents.

The runner's universe loader merges this file with ``universes.yaml``; the
hand-curated file wins on any name collision, so adding either name to
``universes.yaml`` lets you override the auto-managed list.

Run cadence
-----------
* Weekly is plenty. Indices rebalance quarterly, but ad-hoc adds/drops
  (acquisitions, bankruptcies, spin-offs) happen between rebalances.
* Recommended: a cron / GitHub Actions schedule that runs this and opens
  a PR with the changes — so any constituent shift is reviewable, not
  silently applied to a live runner.

Symbol formatting
-----------------
Wikipedia uses dotted class shares (``BRK.B``, ``BF.B``). yfinance wants
dashes; IBKR wants no separator. We standardize on **dashes** because the
default data source is yfinance. The IBKR adapter handles the dash form for
US equities transparently.

Usage::

    uv run python scripts/refresh_universes.py
    git diff config/universes.generated.yaml      # review what changed
    git add config/universes.generated.yaml && git commit
"""

from __future__ import annotations

import datetime as _dt
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Wikipedia blocks requests without a UA. They ask scripts to identify
# themselves with a project name and a way to contact you if it misbehaves.
_USER_AGENT = (
    "trading-agent/0.1 (universe-refresh script; "
    "https://github.com/your-user/trading-agent)"
)

_OUT = Path(__file__).resolve().parents[1] / "config" / "universes.generated.yaml"


def _fetch_html(url: str, *, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _normalize(symbols: list[str]) -> list[str]:
    """Convert vendor-specific share-class separators to yfinance's dashes
    and de-duplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        s = str(raw).strip().upper().replace(".", "-")
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return sorted(out)


def fetch_sp500() -> list[str]:
    html = _fetch_html(_SP500_URL)
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    # The constituents table's symbol column has been called "Symbol" or
    # "Ticker" depending on when you scrape it. Match by prefix to ride out
    # the next Wikipedia rename without code changes.
    sym_col = next(c for c in df.columns if str(c).lower().startswith(("sym", "tick")))
    return _normalize(df[sym_col].astype(str).tolist())


def fetch_nasdaq100() -> list[str]:
    html = _fetch_html(_NDX_URL)
    tables = pd.read_html(io.StringIO(html))
    for df in tables:
        cols_lower = {str(c).lower() for c in df.columns}
        if cols_lower & {"ticker", "symbol"}:
            sym_col = next(c for c in df.columns if str(c).lower() in ("ticker", "symbol"))
            return _normalize(df[sym_col].astype(str).tolist())
    raise RuntimeError("could not find NASDAQ-100 constituent table on the page")


def main() -> int:
    try:
        sp500 = fetch_sp500()
        ndx = fetch_nasdaq100()
    except Exception as e:   # noqa: BLE001 — surface, don't write a half-baked file
        print(f"refresh failed: {e!r}", file=sys.stderr)
        return 1

    # Sanity checks before writing — better to refuse than to ship a 5-symbol
    # "S&P 500" because Wikipedia changed a table id.
    if len(sp500) < 400:
        print(f"refusing to write: sp500 has only {len(sp500)} symbols", file=sys.stderr)
        return 1
    if len(ndx) < 80:
        print(f"refusing to write: nasdaq100 has only {len(ndx)} symbols", file=sys.stderr)
        return 1

    payload = {
        "_generated_by": "scripts/refresh_universes.py",
        "_generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "universes": {
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
        },
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        "# AUTO-GENERATED — do not hand-edit. Re-run scripts/refresh_universes.py to refresh.\n"
        "# Hand-curated entries in universes.yaml WIN over anything here.\n"
        + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    )
    print(f"wrote {_OUT}")
    print(f"  sp500     : {len(sp500)} symbols")
    print(f"  nasdaq100 : {len(ndx)} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
