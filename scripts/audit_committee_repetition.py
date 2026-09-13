#!/usr/bin/env python3
"""Is the committee actually saying anything new?

Reads the memory journal and the prediction ledger and reports, per agent:
how often it names the same subject cycle after cycle, how concentrated its
ideas are, and whether the desk's "disagreement index" is measuring
anything. Stdlib only — runs on the VPS with no install.

    python3 scripts/audit_committee_repetition.py [--db state/memory/memory.db] [--days 120]

Written 2026-09-04 while chasing "the agents keep recommending the same
names". The repetition numbers this prints are the ground truth that the
structural findings in the audit were inferred from; run it on the box
that actually trades.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path


def _entropy(counts) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in ps)
    hmax = math.log(len(ps)) if len(ps) > 1 else 1.0
    return h / hmax if hmax else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="state/memory/memory.db")
    ap.add_argument("--days", type=float, default=120.0)
    a = ap.parse_args()

    p = Path(a.db)
    if not p.exists():
        print(f"no memory db at {p} — run this on the box that trades")
        return 1
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    since = time.time() - a.days * 86400

    # ---- takes: one row per agent per committee run -------------------
    rows = conn.execute(
        "SELECT ts, actor, payload FROM journal WHERE kind='take' AND ts>=? ORDER BY ts",
        (since,),
    ).fetchall()
    if not rows:
        print(f"no 'take' journal rows in the last {a.days:.0f} days")
        return 1

    runs = defaultdict(dict)  # run_bucket -> {agent: payload}
    for r in rows:
        bucket = int(r["ts"] // 3600)  # same hour == same committee run
        try:
            runs[bucket][r["actor"]] = json.loads(r["payload"])
        except Exception:
            continue
    order = sorted(runs)
    print(
        f"committee runs in window: {len(order)}  "
        f"({time.strftime('%Y-%m-%d', time.localtime(order[0] * 3600))} -> "
        f"{time.strftime('%Y-%m-%d', time.localtime(order[-1] * 3600))})\n"
    )

    def subject(pay):
        s = ((pay or {}).get("prediction") or {}).get("subject")
        return str(s).upper().strip() if s else None

    agents = sorted({ag for b in order for ag in runs[b]})
    print(f"{'agent':<16} {'runs':>5} {'repeat':>8} {'uniq':>6} {'divers':>7}  top subjects")
    print("-" * 92)
    for ag in agents:
        seq = [subject(runs[b].get(ag)) for b in order if ag in runs[b]]
        seq = [s for s in seq if s]
        if not seq:
            continue
        repeats = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
        rep = repeats / max(len(seq) - 1, 1)
        c = Counter(seq)
        top = ", ".join(f"{s}×{n}" for s, n in c.most_common(4))
        print(
            f"{ag:<16} {len(seq):>5} {rep * 100:>7.0f}% {len(c):>6} {_entropy(c.values()):>6.2f}  {top}"
        )

    # ---- is the whole committee converging on one name? ----------------
    print()
    per_run_unique = [len({subject(v) for v in runs[b].values() if subject(v)}) for b in order]
    if per_run_unique:
        print(
            f"distinct subjects named per run: median "
            f"{sorted(per_run_unique)[len(per_run_unique) // 2]} of ~{len(agents)} agents"
        )

    # ---- stance spread: range (what the code reports) vs entropy -------
    ranges, entropies = [], []
    score = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}
    for b in order:
        st = [score.get((v or {}).get("stance", "neutral"), 0.0) for v in runs[b].values()]
        if not st:
            continue
        ranges.append((max(st) - min(st)) / 2.0)
        entropies.append(_entropy(Counter(st).values()))
    if ranges:
        sat = sum(1 for r in ranges if r >= 1.0) / len(ranges)
        print(
            f"disagreement_index (as reported): mean {sum(ranges) / len(ranges):.2f}, "
            f"pinned at 1.00 in {sat * 100:.0f}% of runs"
        )
        print(
            f"stance entropy (what it should be): mean {sum(entropies) / len(entropies):.2f} "
            f"(0 = everyone agrees, 1 = evenly split)"
        )

    # ---- calibration: is anyone actually right? ------------------------
    pr = conn.execute(
        "SELECT agent, outcome, COUNT(*) n FROM predictions "
        "WHERE ts>=? AND outcome IS NOT NULL GROUP BY agent, outcome",
        (since,),
    ).fetchall()
    if pr:
        tally = defaultdict(Counter)
        for r in pr:
            tally[r["agent"]][r["outcome"]] = r["n"]
        print(f"\n{'agent':<16} {'graded':>7} {'hit rate':>9}")
        print("-" * 36)
        for ag in sorted(tally):
            t = tally[ag]
            tot = sum(t.values())
            print(f"{ag:<16} {tot:>7} {t.get('hit', 0) / tot * 100:>8.0f}%")
    else:
        print("\nno graded predictions in window — the scorecard is not closing the loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
