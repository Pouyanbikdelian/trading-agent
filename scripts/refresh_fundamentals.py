"""Refresh the fundamentals Parquet from yfinance.

Reads the universe(s) passed on the command line, pulls each symbol's
``Ticker.info`` from yfinance, normalizes the fields we care about, and
writes ``<data_dir>/fundamentals.parquet``. The runner's quality and
sector-momentum screens read that file.

Usage
-----
::

    uv run python scripts/refresh_fundamentals.py sp500 nasdaq100
    # → writes data/parquet/fundamentals.parquet

Run cadence: **weekly is plenty.** Fundamentals don't move fast enough
to justify more often, and yfinance rate-limits big batches. A weekly
GitHub Actions / cron job is the right call for production.

Note
----
yfinance fundamentals coverage is patchy on small/non-US names. If you
hit big gaps you'll either need a paid feed (Tiingo $10/mo, IEX $30/mo,
Polygon $30/mo) or to swap in a different ``fetch_*`` function — the
quality screen takes a dict, so the data plumbing is replaceable.
"""

from __future__ import annotations

import sys

from trading.core.config import settings
from trading.core.universes import load_universe
from trading.data.fundamentals_source import (
    fetch_fundamentals_yf,
    read_fundamentals_cache,
    write_fundamentals_cache,
)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: refresh_fundamentals.py <universe> [<universe> ...]", file=sys.stderr)
        return 1

    settings.ensure_dirs()
    out_path = settings.data_dir / "fundamentals.parquet"

    symbols: set[str] = set()
    for name in argv:
        try:
            for ins in load_universe(name):
                symbols.add(ins.symbol)
        except KeyError as e:
            print(f"unknown universe: {e}", file=sys.stderr)
            return 1

    print(f"refreshing fundamentals for {len(symbols)} symbols across {len(argv)} universes")
    # Merge with whatever's already cached so symbols that drop out of one
    # universe but remain in another aren't forgotten.
    existing = read_fundamentals_cache(out_path)
    fresh = fetch_fundamentals_yf(sorted(symbols))
    existing.update(fresh)

    write_fundamentals_cache(out_path, existing)
    print(f"wrote {len(existing)} rows to {out_path}")
    print(f"  new/refreshed: {len(fresh)}; total cached: {len(existing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
