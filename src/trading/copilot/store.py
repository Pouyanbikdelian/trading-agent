"""Copilot decision store: structured records + FTS5 retrieval.

``state/copilot.db`` is DERIVED data — rebuilt any time from the memory
journal, which the committee already writes (rulings via
``mem.journal("committee", ...)``, per-agent takes via
``mem.journal("take", ...)``, PM runs via ``mem.journal("agent_pm", ...)``).
Deriving instead of double-writing keeps the committee pipeline
untouched (Phase 1 scope: no behavior changes) and makes ingest
idempotent: we track the last journal id seen and only fold in new rows.

Schema:

* ``decisions`` — one row per committee ruling / PM run: timestamp,
  symbols mentioned, thesis, votes (per-agent stances), dissent score,
  invalidation conditions ("watch"), raw payload.
* ``transcript`` — one row per agent take: who said what, linked to the
  decision that followed (the next ruling at-or-after the take).
* ``decisions_fts`` / ``transcript_fts`` — SQLite FTS5 mirrors for
  keyword retrieval. No vector DB: at this corpus size (a few rulings a
  week) FTS + symbol/date filters answers everything; embeddings would
  be complexity without evidence of need.

Citation IDs: decisions are cited as ``D<journal_id>``, transcript rows
as ``T<journal_id>`` — stable, and traceable straight back to the
memory journal row that produced them.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

_LOG = logger.bind(component="copilot")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id          TEXT PRIMARY KEY,      -- 'D<journal_id>'
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,         -- committee | agent_pm
    symbols     TEXT NOT NULL,         -- JSON list
    thesis      TEXT NOT NULL,
    votes       TEXT NOT NULL,         -- JSON {agent: stance}
    dissent     REAL,
    confidence  REAL,
    invalidation TEXT NOT NULL,        -- the ruling's 'watch' conditions
    final       TEXT NOT NULL,         -- posture / target weights
    payload     TEXT NOT NULL          -- raw JSON for full fidelity
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE TABLE IF NOT EXISTS transcript (
    id          TEXT PRIMARY KEY,      -- 'T<journal_id>'
    ts          REAL NOT NULL,
    decision_id TEXT,                  -- ruling this take fed into (nullable)
    agent       TEXT NOT NULL,
    stance      TEXT,
    confidence  REAL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcript_ts ON transcript(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
    id UNINDEXED, thesis, invalidation, final, symbols
);
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    id UNINDEXED, agent, text
);
"""

# Symbols are extracted from free text: uppercase 1-5 letter tokens that
# aren't common English words. Deliberately conservative — a missed
# symbol only weakens retrieval; a false one pollutes it.
_SYM_RE = re.compile(r"\b[A-Z]{1,5}\b")
_NOT_SYMBOLS = {
    "A",
    "I",
    "AND",
    "OR",
    "THE",
    "TO",
    "OF",
    "IN",
    "ON",
    "AT",
    "IS",
    "IT",
    "BE",
    "AS",
    "BY",
    "IF",
    "NO",
    "NOT",
    "BUT",
    "FOR",
    "WITH",
    "VIX",
    "USD",
    "CHF",
    "EUR",
    "FED",
    "ECB",
    "BOJ",
    "CPI",
    "PPI",
    "PCE",
    "GDP",
    "ETF",
    "ETFS",
    "LLM",
    "AI",
    "US",
    "EU",
    "UK",
    "IPO",
    "YOY",
    "QOQ",
    "ATH",
    "PM",
    "AM",
    "OK",
    "CEO",
    "CFO",
    "EPS",
    "RSI",
    "EMA",
    "SMA",
    "HWM",
    "IV",
    "OI",
}


def _extract_symbols(*texts: str, known: set[str] | None = None) -> list[str]:
    out: list[str] = []
    for t in texts:
        for m in _SYM_RE.findall(t or ""):
            if m in _NOT_SYMBOLS:
                continue
            if known is not None and m not in known:
                continue
            if m not in out:
                out.append(m)
    return out


class CopilotStore:
    """Owns copilot.db. Writable ONLY for ingest of derived data —
    nothing here touches the source-of-truth stores."""

    def __init__(self, state_dir: Path) -> None:
        self.path = Path(state_dir) / "copilot.db"
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------ ingest

    def ingest(self, memory_dir: Path, *, known_symbols: set[str] | None = None) -> int:
        """Fold new journal rows into the copilot store. Idempotent —
        remembers the highest journal id processed. Returns rows added."""
        src = Path(memory_dir) / "memory.db"
        if not src.exists():
            # MemoryStore's actual filename may differ; try the dir itself.
            candidates = list(Path(memory_dir).glob("*.db"))
            if not candidates:
                return 0
            src = candidates[0]
        ro = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        ro.row_factory = sqlite3.Row
        try:
            last = int(
                (
                    self.conn.execute(
                        "SELECT value FROM meta WHERE key='last_journal_id'"
                    ).fetchone()
                    or {"value": 0}
                )["value"]
            )
            rows = ro.execute(
                "SELECT id, ts, kind, actor, payload FROM journal "
                "WHERE id > ? AND kind IN ('committee','take','agent_pm') ORDER BY id",
                (last,),
            ).fetchall()
        finally:
            ro.close()

        added = 0
        pending_takes: list[sqlite3.Row] = []
        for r in rows:
            payload = json.loads(r["payload"])
            if r["kind"] == "take":
                pending_takes.append(r)
                self._add_take(r, payload, decision_id=None)
                added += 1
            elif r["kind"] == "committee":
                did = self._add_committee(r, payload, known_symbols)
                # Takes since the previous ruling fed THIS ruling.
                for t in pending_takes:
                    self.conn.execute(
                        "UPDATE transcript SET decision_id=? WHERE id=?",
                        (did, f"T{t['id']}"),
                    )
                pending_takes = []
                added += 1
            elif r["kind"] == "agent_pm":
                self._add_pm(r, payload, known_symbols)
                added += 1
        if rows:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES ('last_journal_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(rows[-1]["id"]),),
            )
        return added

    def _add_committee(
        self, r: sqlite3.Row, payload: dict[str, Any], known: set[str] | None
    ) -> str:
        did = f"D{r['id']}"
        ruling = payload.get("ruling") or {}
        thesis = str(ruling.get("proposal", ""))
        watch = str(ruling.get("watch", ""))
        posture = str(ruling.get("posture", ""))
        votes = {
            a: t.get("stance", "?")
            for a, t in (payload.get("takes") or {}).items()
            if isinstance(t, dict)
        }
        symbols = _extract_symbols(thesis, watch, known=known)
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                did,
                float(r["ts"]),
                "committee",
                json.dumps(symbols),
                thesis,
                json.dumps(votes),
                float(payload.get("disagreement") or 0.0),
                None,
                watch,
                posture,
                json.dumps(payload),
            ),
        )
        self.conn.execute(
            "INSERT INTO decisions_fts VALUES (?,?,?,?,?)",
            (did, thesis, watch, posture, " ".join(symbols)),
        )
        return did

    def _add_pm(self, r: sqlite3.Row, payload: dict[str, Any], known: set[str] | None) -> None:
        did = f"D{r['id']}"
        weights = payload.get("weights") or {}
        rationale = str(payload.get("rationale", ""))
        symbols = sorted(weights) or _extract_symbols(rationale, known=known)
        final = json.dumps(weights)
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                did,
                float(r["ts"]),
                "agent_pm",
                json.dumps(symbols),
                rationale,
                json.dumps({}),
                None,
                None,
                str(payload.get("watch", "")),
                final,
                json.dumps(payload),
            ),
        )
        self.conn.execute(
            "INSERT INTO decisions_fts VALUES (?,?,?,?,?)",
            (did, rationale, str(payload.get("watch", "")), final, " ".join(symbols)),
        )

    def _add_take(self, r: sqlite3.Row, payload: dict[str, Any], decision_id: str | None) -> None:
        tid = f"T{r['id']}"
        text = " ".join(
            str(payload.get(k, ""))
            for k in ("take", "reason", "rationale", "text")
            if payload.get(k)
        ) or json.dumps({k: v for k, v in payload.items() if k not in ("agent", "prediction_id")})
        self.conn.execute(
            "INSERT OR REPLACE INTO transcript VALUES (?,?,?,?,?,?,?)",
            (
                tid,
                float(r["ts"]),
                decision_id,
                str(payload.get("agent", r["actor"])),
                payload.get("stance"),
                payload.get("confidence"),
                text[:2000],
            ),
        )
        self.conn.execute(
            "INSERT INTO transcript_fts VALUES (?,?,?)",
            (tid, str(payload.get("agent", r["actor"])), text[:2000]),
        )

    # ---------------------------------------------------------- retrieval

    @staticmethod
    def _fts_query(terms: list[str]) -> str:
        safe = [re.sub(r"[^A-Za-z0-9]", "", t) for t in terms]
        safe = [t for t in safe if len(t) >= 2]
        return " OR ".join(f'"{t}"' for t in safe[:12])

    def search_decisions(
        self, terms: list[str], *, symbol: str | None = None, limit: int = 6, strict: bool = False
    ) -> list[dict[str, Any]]:
        """FTS + symbol filter, newest first.

        ``strict=True``: only genuine FTS/symbol matches — no match means
        an EMPTY list, never "here's something recent instead". The
        engine depends on that: padding off-topic questions with
        unrelated rulings made the copilot narrate noise (2026-07-16).
        ``strict=False`` keeps the newest-overall fallback for browsing
        callers that want "latest decisions" semantics."""
        rows: list[sqlite3.Row] = []
        q = self._fts_query(terms)
        if q:
            rows = self.conn.execute(
                "SELECT d.* FROM decisions_fts f JOIN decisions d ON d.id=f.id "
                "WHERE decisions_fts MATCH ? ORDER BY d.ts DESC LIMIT ?",
                (q, limit * 3),
            ).fetchall()
        if not rows and not strict:
            rows = self.conn.execute(
                "SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit * 3,)
            ).fetchall()
        out = []
        for r in rows:
            symbols = json.loads(r["symbols"])
            if symbol and symbol.upper() not in symbols:
                continue
            out.append(self._decision_dict(r))
            if len(out) >= limit:
                break
        # Symbol asked but nothing matched its filter: fall back to a
        # STRICT FTS match on the symbol itself (it may appear in thesis
        # text without making the extracted-symbols list). Strict means:
        # no match → empty, never "here's something recent instead" —
        # the engine's honesty path depends on empty meaning empty.
        if symbol and not out:
            q = self._fts_query([symbol])
            if q:
                rows = self.conn.execute(
                    "SELECT d.* FROM decisions_fts f JOIN decisions d ON d.id=f.id "
                    "WHERE decisions_fts MATCH ? ORDER BY d.ts DESC LIMIT ?",
                    (q, limit),
                ).fetchall()
                return [self._decision_dict(r) for r in rows]
        return out

    def search_transcript(self, terms: list[str], *, limit: int = 6) -> list[dict[str, Any]]:
        q = self._fts_query(terms)
        if not q:
            return []
        rows = self.conn.execute(
            "SELECT t.* FROM transcript_fts f JOIN transcript t ON t.id=f.id "
            "WHERE transcript_fts MATCH ? ORDER BY t.ts DESC LIMIT ?",
            (q, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc).isoformat(),
                "decision_id": r["decision_id"],
                "agent": r["agent"],
                "stance": r["stance"],
                "text": r["text"],
            }
            for r in rows
        ]

    def transcript_for_decision(self, decision_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM transcript WHERE decision_id=? ORDER BY ts", (decision_id,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc).isoformat(),
                "agent": r["agent"],
                "stance": r["stance"],
                "text": r["text"][:500],
            }
            for r in rows
        ]

    @staticmethod
    def _decision_dict(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"],
            "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc).isoformat(),
            "kind": r["kind"],
            "symbols": json.loads(r["symbols"]),
            "thesis": r["thesis"],
            "votes": json.loads(r["votes"]),
            "dissent": r["dissent"],
            "invalidation": r["invalidation"],
            "final": r["final"],
        }
