"""MemoryStore — single facade over the permanent memory spine.

Design rules (docs/concept_multiagent_memory.md):

* **Append-only.** No DELETE statements exist in this module. Lessons
  and dossiers are superseded or retired, never erased; the journal is
  immutable history.
* **Text is canonical.** Lessons and World State dossiers are markdown
  files under ``state/memory/`` (an Obsidian-compatible vault); SQLite
  carries the indexes, counters and relational links. Embeddings, when
  they arrive, are derived artifacts — recomputable, never authoritative.
* **Everything gradeable.** Predictions carry an explicit horizon and
  are auto-graded by ``grade_due_predictions`` once prices exist for
  the due date. Skill is a number attached to memory, not a vibe.
* **Trust is earned.** Sources start at a neutral Beta(1,1) prior and
  move only on graded evidence. Gossip is labeled, never dropped.

Concurrency mirrors RunnerStore: WAL, ``check_same_thread=False``,
writes serialized by the runner. Markdown writes are atomic
(tempfile + os.replace).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,           -- cycle|fill|halt|take|debate|news|note|...
    actor       TEXT NOT NULL DEFAULT 'system',
    payload     TEXT NOT NULL            -- JSON
);
CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal(ts);
CREATE INDEX IF NOT EXISTS idx_journal_kind ON journal(kind);

CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,        -- ep-<uuid8>
    ts_open     REAL NOT NULL,
    ts_close    REAL NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL DEFAULT 'long',
    entry_px    REAL,
    exit_px     REAL,
    pnl_pct     REAL,
    entry_pctile_52w REAL,               -- 0=52w low, 1=52w high (top vs dip)
    context     TEXT NOT NULL DEFAULT '{}',  -- JSON: regime, vix, macro dial, agents' views
    tags        TEXT NOT NULL DEFAULT ''     -- space-separated
);
CREATE INDEX IF NOT EXISTS idx_episodes_symbol ON episodes(symbol);
CREATE INDEX IF NOT EXISTS idx_episodes_close ON episodes(ts_close);

CREATE TABLE IF NOT EXISTS lessons (
    id          TEXT PRIMARY KEY,        -- ls-<uuid8>
    created_ts  REAL NOT NULL,
    statement   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'candidate',  -- candidate|established|retired
    support     INTEGER NOT NULL DEFAULT 0,
    contradict  INTEGER NOT NULL DEFAULT 0,
    retired_ts  REAL,
    retired_why TEXT,
    tags        TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lesson_evidence (
    lesson_id   TEXT NOT NULL,
    episode_id  TEXT NOT NULL,
    relation    TEXT NOT NULL,           -- supports|contradicts|origin
    ts          REAL NOT NULL,
    PRIMARY KEY (lesson_id, episode_id, relation)
);

CREATE TABLE IF NOT EXISTS predictions (
    id          TEXT PRIMARY KEY,        -- pr-<uuid8>
    ts          REAL NOT NULL,
    agent       TEXT NOT NULL,
    subject     TEXT NOT NULL,           -- e.g. 'NDX', 'AAPL', 'portfolio'
    direction   TEXT NOT NULL,           -- up|down|flat
    horizon_days INTEGER NOT NULL,
    confidence  REAL NOT NULL,           -- 0..1
    statement   TEXT NOT NULL,
    sources     TEXT NOT NULL DEFAULT '',-- space-separated source keys
    due_ts      REAL NOT NULL,
    graded_ts   REAL,
    outcome     TEXT,                    -- hit|miss|flat
    realized_move REAL,
    brier       REAL
);
CREATE INDEX IF NOT EXISTS idx_pred_due ON predictions(due_ts);
CREATE INDEX IF NOT EXISTS idx_pred_agent ON predictions(agent);

CREATE TABLE IF NOT EXISTS source_trust (
    source      TEXT PRIMARY KEY,
    hits        INTEGER NOT NULL DEFAULT 0,
    misses      INTEGER NOT NULL DEFAULT 0,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'unknown'   -- wire|outlet|social|gossip|...
);
"""


def _now() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def _short(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class MemoryStore:
    """Facade over the five memory stores. One instance per process."""

    def __init__(self, root: str | Path) -> None:
        """``root`` is the memory directory, e.g. ``state/memory``."""
        self.root = Path(root)
        self.lessons_dir = self.root / "lessons"
        self.world_dir = self.root / "world"
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.root / "memory.db"), isolation_level=None, check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------ journal

    def journal(self, kind: str, payload: dict[str, Any], *, actor: str = "system") -> int:
        cur = self.conn.execute(
            "INSERT INTO journal (ts, kind, actor, payload) VALUES (?, ?, ?, ?)",
            (_now(), kind, actor, json.dumps(payload, default=str)),
        )
        return int(cur.lastrowid)

    def journal_tail(self, n: int = 20, kind: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM journal"
        args: tuple[Any, ...] = ()
        if kind:
            q += " WHERE kind = ?"
            args = (kind,)
        q += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(q, (*args, n)).fetchall()
        return [
            {
                "id": r["id"],
                "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc),
                "kind": r["kind"],
                "actor": r["actor"],
                "payload": json.loads(r["payload"]),
            }
            for r in rows
        ]

    # ----------------------------------------------------------- episodes

    def add_episode(
        self,
        *,
        symbol: str,
        ts_open: datetime,
        ts_close: datetime,
        entry_px: float | None,
        exit_px: float | None,
        pnl_pct: float | None,
        entry_pctile_52w: float | None,
        context: dict[str, Any] | None = None,
        tags: str = "",
        side: str = "long",
    ) -> str:
        eid = _short("ep")
        self.conn.execute(
            """INSERT INTO episodes
               (id, ts_open, ts_close, symbol, side, entry_px, exit_px, pnl_pct,
                entry_pctile_52w, context, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                eid,
                ts_open.timestamp(),
                ts_close.timestamp(),
                symbol.upper(),
                side,
                entry_px,
                exit_px,
                pnl_pct,
                entry_pctile_52w,
                json.dumps(context or {}, default=str),
                tags,
            ),
        )
        self.journal("episode", {"id": eid, "symbol": symbol, "pnl_pct": pnl_pct})
        return eid

    def episodes_for(self, symbol: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if symbol:
            return self.conn.execute(
                "SELECT * FROM episodes WHERE symbol = ? ORDER BY ts_close DESC LIMIT ?",
                (symbol.upper(), limit),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM episodes ORDER BY ts_close DESC LIMIT ?", (limit,)
        ).fetchall()

    # ------------------------------------------------------------ lessons

    def add_lesson(
        self, statement: str, *, origin_episodes: list[str] | None = None, tags: str = ""
    ) -> str:
        lid = _short("ls")
        ts = _now()
        self.conn.execute(
            "INSERT INTO lessons (id, created_ts, statement, tags) VALUES (?, ?, ?, ?)",
            (lid, ts, statement, tags),
        )
        for eid in origin_episodes or []:
            self.conn.execute(
                "INSERT OR IGNORE INTO lesson_evidence VALUES (?, ?, 'origin', ?)",
                (lid, eid, ts),
            )
        self._write_lesson_card(lid)
        self.journal("lesson_created", {"id": lid, "statement": statement})
        return lid

    def add_evidence(self, lesson_id: str, episode_id: str, *, supports: bool) -> None:
        rel = "supports" if supports else "contradicts"
        self.conn.execute(
            "INSERT OR IGNORE INTO lesson_evidence VALUES (?, ?, ?, ?)",
            (lesson_id, episode_id, rel, _now()),
        )
        col = "support" if supports else "contradict"
        self.conn.execute(f"UPDATE lessons SET {col} = {col} + 1 WHERE id = ?", (lesson_id,))
        # Promotion: 3+ net supporting episodes establishes a candidate.
        row = self.conn.execute(
            "SELECT status, support, contradict FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        if row and row["status"] == "candidate" and row["support"] - row["contradict"] >= 3:
            self.conn.execute(
                "UPDATE lessons SET status = 'established' WHERE id = ?", (lesson_id,)
            )
            self.journal("lesson_established", {"id": lesson_id})
        self._write_lesson_card(lesson_id)

    def retire_lesson(self, lesson_id: str, why: str) -> None:
        """Retired, never deleted — the card keeps its full history."""
        self.conn.execute(
            "UPDATE lessons SET status='retired', retired_ts=?, retired_why=? WHERE id=?",
            (_now(), why, lesson_id),
        )
        self._write_lesson_card(lesson_id)
        self.journal("lesson_retired", {"id": lesson_id, "why": why})

    def lessons(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute(
                "SELECT * FROM lessons WHERE status = ? ORDER BY support - contradict DESC",
                (status,),
            ).fetchall()
        return self.conn.execute("SELECT * FROM lessons ORDER BY created_ts DESC").fetchall()

    def _write_lesson_card(self, lesson_id: str) -> None:
        """Render the lesson as an Obsidian-compatible markdown card."""
        row = self.conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        if row is None:
            return
        ev = self.conn.execute(
            "SELECT * FROM lesson_evidence WHERE lesson_id = ? ORDER BY ts", (lesson_id,)
        ).fetchall()
        created = datetime.fromtimestamp(row["created_ts"], tz=timezone.utc)
        lines = [
            "---",
            f"id: {row['id']}",
            f"status: {row['status']}",
            f"created: {created.date().isoformat()}",
            f"support: {row['support']}",
            f"contradict: {row['contradict']}",
            f"tags: [{row['tags']}]",
            "---",
            "",
            f"# {row['statement']}",
            "",
            "## Evidence",
        ]
        for e in ev:
            ts = datetime.fromtimestamp(e["ts"], tz=timezone.utc).date().isoformat()
            lines.append(f"- {ts} **{e['relation']}** [[{e['episode_id']}]]")
        if row["status"] == "retired":
            died = datetime.fromtimestamp(row["retired_ts"], tz=timezone.utc).date().isoformat()
            lines += ["", f"## Retired {died}", "", row["retired_why"] or ""]
        _atomic_write(self.lessons_dir / f"{lesson_id}.md", "\n".join(lines) + "\n")

    # --------------------------------------------------------- world state

    def update_dossier(self, slug: str, update: str, *, expects: str | None = None) -> Path:
        """Append a timestamped update to a narrative dossier. Creates the
        dossier on first touch. History is never rewritten."""
        path = self.world_dir / f"{slug}.md"
        stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if not path.exists():
            head = f"# {slug.replace('_', ' ').title()}\n\n## Timeline\n"
            _atomic_write(path, head)
        body = path.read_text()
        entry = f"\n### {stamp}\n{update.strip()}\n"
        if expects:
            entry += f"\n*Crowd expects:* {expects.strip()}\n"
        _atomic_write(path, body + entry)
        self.journal("dossier_update", {"slug": slug, "update": update[:200]})
        return path

    def dossiers(self) -> list[str]:
        if not self.world_dir.exists():
            return []
        return sorted(p.stem for p in self.world_dir.glob("*.md"))

    # ---------------------------------------------------------- scorecard

    def add_prediction(
        self,
        *,
        agent: str,
        subject: str,
        direction: str,
        horizon_days: int,
        confidence: float,
        statement: str,
        sources: list[str] | None = None,
    ) -> str:
        pid = _short("pr")
        ts = _now()
        self.conn.execute(
            """INSERT INTO predictions
               (id, ts, agent, subject, direction, horizon_days, confidence,
                statement, sources, due_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid,
                ts,
                agent,
                subject.upper(),
                direction,
                horizon_days,
                confidence,
                statement,
                " ".join(sources or []),
                ts + horizon_days * 86400.0,
            ),
        )
        return pid

    def due_predictions(self, asof: datetime | None = None) -> list[sqlite3.Row]:
        cutoff = (asof or datetime.now(tz=timezone.utc)).timestamp()
        return self.conn.execute(
            "SELECT * FROM predictions WHERE graded_ts IS NULL AND due_ts <= ?", (cutoff,)
        ).fetchall()

    def grade_prediction(
        self, prediction_id: str, realized_move: float, *, flat_band: float = 0.005
    ) -> str:
        """Grade against the realized move over the horizon. Also feeds the
        source-trust ledger for every source the prediction cited."""
        row = self.conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if row is None or row["graded_ts"] is not None:
            return "skipped"
        actual = (
            "flat" if abs(realized_move) < flat_band else ("up" if realized_move > 0 else "down")
        )
        outcome = "hit" if actual == row["direction"] else "miss"
        # Brier on the directional claim: p = confidence that direction is right.
        p = float(row["confidence"])
        brier = (p - (1.0 if outcome == "hit" else 0.0)) ** 2
        self.conn.execute(
            "UPDATE predictions SET graded_ts=?, outcome=?, realized_move=?, brier=? WHERE id=?",
            (_now(), outcome, realized_move, brier, prediction_id),
        )
        for source in (row["sources"] or "").split():
            self.bump_trust(source, hit=(outcome == "hit"))
        self.journal(
            "prediction_graded",
            {"id": prediction_id, "agent": row["agent"], "outcome": outcome, "brier": brier},
        )
        return outcome

    def calibration(self) -> list[dict[str, Any]]:
        """Per-agent scorecard: n graded, hit rate, mean Brier."""
        rows = self.conn.execute(
            """SELECT agent, COUNT(*) AS n,
                      AVG(CASE WHEN outcome='hit' THEN 1.0 ELSE 0.0 END) AS hit_rate,
                      AVG(brier) AS brier
               FROM predictions WHERE graded_ts IS NOT NULL
               GROUP BY agent ORDER BY brier ASC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- trust

    def bump_trust(self, source: str, *, hit: bool, kind: str | None = None) -> None:
        ts = _now()
        self.conn.execute(
            """INSERT INTO source_trust (source, hits, misses, first_seen, last_seen, kind)
               VALUES (?, ?, ?, ?, ?, COALESCE(?, 'unknown'))
               ON CONFLICT(source) DO UPDATE SET
                 hits   = hits + excluded.hits,
                 misses = misses + excluded.misses,
                 last_seen = excluded.last_seen,
                 kind = COALESCE(?, kind)""",
            (source, 1 if hit else 0, 0 if hit else 1, ts, ts, kind, kind),
        )

    def trust(self, source: str) -> float:
        """Posterior mean of Beta(1+hits, 1+misses). New sources -> 0.5."""
        row = self.conn.execute(
            "SELECT hits, misses FROM source_trust WHERE source = ?", (source,)
        ).fetchone()
        if row is None:
            return 0.5
        return (1.0 + row["hits"]) / (2.0 + row["hits"] + row["misses"])

    def trust_table(self, min_graded: int = 1) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM source_trust WHERE hits + misses >= ? ORDER BY hits + misses DESC",
            (min_graded,),
        ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "source": r["source"],
                    "kind": r["kind"],
                    "graded": r["hits"] + r["misses"],
                    "trust": (1.0 + r["hits"]) / (2.0 + r["hits"] + r["misses"]),
                }
            )
        return out

    # ------------------------------------------------------------ summary

    def stats(self) -> dict[str, int]:
        c = self.conn
        return {
            "journal": c.execute("SELECT COUNT(*) FROM journal").fetchone()[0],
            "episodes": c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "lessons": c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
            "predictions": c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "sources": c.execute("SELECT COUNT(*) FROM source_trust").fetchone()[0],
            "dossiers": len(self.dossiers()),
        }


def default_store() -> MemoryStore:
    """Production store under ``settings.state_dir / memory``."""
    from trading.core.config import settings

    return MemoryStore(Path(settings.state_dir) / "memory")


__all__ = ["MemoryStore", "default_store"]
