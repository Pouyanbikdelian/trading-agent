"""Refresh index constituents — thin CLI wrapper.

All the logic lives in ``trading.data.universe_refresh`` because the
Docker image copies ``src/`` and NOT ``scripts/``: anything the runner
needs at runtime must be importable, not a file on a path that only
exists in the git checkout.

Writes ``state/universes.generated.yaml`` (override with
``UNIVERSES_GENERATED_PATH``). The runner also does this on its own
schedule — Sundays 03:00 UTC, job ``universe_refresh`` — and alerts with
the membership delta, so a constituent change is visible rather than
silent. This script is for running it by hand.

Usage::

    uv run python scripts/refresh_universes.py
    docker compose exec trader python -m trading.data.universe_refresh
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading.data.universe_refresh import refresh


def main() -> int:
    result = refresh()
    if not result["ok"]:
        print(f"refresh failed: {result['reason']}", file=sys.stderr)
        return 1
    print(f"wrote {result['path']}")
    for name, n in result["counts"].items():
        print(f"  {name:<12} {n} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
