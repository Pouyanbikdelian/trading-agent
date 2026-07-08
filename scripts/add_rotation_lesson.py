"""One-off: record the risk-on/risk-off rotation lesson in the memory spine.

Adds a single `candidate` lesson (see docs/rotation_cluster_analysis.md) to the
production store at ``settings.state_dir / memory``. Idempotent: if a lesson
with the same marker already exists it does nothing, so it's safe to re-run.

Run inside the trader container after deploy:
    docker compose exec -T trader python scripts/add_rotation_lesson.py
"""
from __future__ import annotations

from trading.memory.store import default_store

MARKER = "risk-on/risk-off seesaw"

STATEMENT = (
    "US sector ETFs split into a persistent risk-on/risk-off seesaw: "
    "Tech/Semis/Uranium (XLK/SMH/URA) vs Healthcare/REIT/Biotech (XLV/XLRE/IBB). "
    "Their returns RELATIVE to SPY are ~-0.5 correlated and negative in 100% of "
    "rolling 12-month windows over 2021-2026 (strongest recently, ~-0.67); the "
    "tightest pair is XLV<->XLK, while URA is the loosest member (own supply cycle). "
    "It is a COINCIDENT rotation (peak corr at lag 0, no exploitable 1-2 month lead) "
    "and does NOT survive as a naive momentum or mean-reversion long/short between the "
    "clusters (both whipsaw to ~0 or negative Sharpe on 2021-2026 monthly data; simply "
    "holding the risk cluster won that window). Use it as a CROWDING/HEDGE and "
    "regime-context signal: when the book is crowded in one cluster the other is a real "
    "hedge, and a flip in the 3-month relative-momentum spread marks the rotation turning. "
    "Evidence: scripts/rotation_cluster_analysis.py; docs/rotation_cluster_analysis.md."
)

TAGS = "rotation,sector-etf,risk-on-risk-off,hedging,rrg,regime"


def main() -> None:
    store = default_store()
    for row in store.lessons():
        if MARKER in (row["statement"] or ""):
            print(f"lesson already present ({row['id']}, status={row['status']}); skipping")
            return
    lid = store.add_lesson(STATEMENT, tags=TAGS)
    print(f"added lesson {lid} (status=candidate)")
    print(f"card: {store.lessons_dir / (lid + '.md')}")


if __name__ == "__main__":
    main()
