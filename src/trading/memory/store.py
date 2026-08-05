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

import contextlib
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

-- The counterfactual ledger: what the desk considered and did NOT do.
--
-- Every other table here records outcomes of actions taken, which makes
-- the whole memory blind to the only comparison that establishes whether
-- selection has edge: did the names we picked beat the names we passed
-- on? The ranked candidate ladder is computed every cycle and currently
-- discarded within minutes. This is where it goes instead.
--
-- Forward returns are stored alongside the benchmark over the identical
-- window, because in a rising market a ledger of passed names looks
-- excellent on absolute return alone and the comparison is meaningless.
CREATE TABLE IF NOT EXISTS shadow (
    id           TEXT PRIMARY KEY,       -- sh-<uuid8>
    ts           REAL NOT NULL,
    symbol       TEXT NOT NULL,
    origin       TEXT NOT NULL,          -- ladder|committee|mandate|risk|operator
    disposition  TEXT NOT NULL,          -- taken|passed|cut_by_risk|cut_by_cap
    rank         INTEGER,                -- position in the ranked ladder, if any
    score        REAL,                   -- the ranking score at the time
    why          TEXT NOT NULL DEFAULT '',
    conditions   TEXT NOT NULL DEFAULT '{}',  -- JSON regime fingerprint
    px_at        REAL,                   -- close on the day of the decision
    pctile_52w   REAL,                   -- 0=52w low, 1=52w high, at decision time
    r5           REAL,
    r21          REAL,
    r63          REAL,
    bench5       REAL,
    bench21      REAL,
    bench63      REAL,
    graded_ts    REAL                    -- set once the 63d leg lands
);
CREATE INDEX IF NOT EXISTS idx_shadow_ts ON shadow(ts);
CREATE INDEX IF NOT EXISTS idx_shadow_symbol ON shadow(symbol);
CREATE INDEX IF NOT EXISTS idx_shadow_open ON shadow(graded_ts);
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
            self._migrate(self._conn)
        return self._conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive column migrations.

        ``CREATE TABLE IF NOT EXISTS`` silently does nothing when the
        table already exists, so a column added to ``_SCHEMA`` never
        reaches a database created before the change. Each entry here is
        an ``ADD COLUMN`` that is safe to attempt repeatedly — a
        duplicate-column error means the migration already ran.

        Additive only. Nothing in this module drops or rewrites a column;
        the memory spine is append-only by design.
        """
        for table, column, decl in (("shadow", "pctile_52w", "REAL"),):
            with contextlib.suppress(sqlite3.OperationalError):  # already present
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

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
        return [self._journal_row(r) for r in rows]

    def journal_window(
        self,
        days: float,
        *,
        kinds: list[str] | None = None,
        per_kind_limit: int = 40,
    ) -> dict[str, list[dict[str, Any]]]:
        """Journal rows from the last ``days``, bucketed by kind.

        ``journal_tail(n)`` takes the newest N rows of anything, which is
        not a time window at all: one committee run alone writes ten rows
        (eight takes, a debate, a ruling), so "the last 80 rows" can be
        three days in a busy week and a fortnight in a quiet one. Any
        caller reasoning about "this week" was reasoning about the wrong
        set of rows.

        Bucketing by kind then matters for the same reason. Graded
        outcomes are the only rows carrying measured truth, and in a flat
        list they compete for prompt space with daily heartbeats that
        carry none — so a busy week could push every outcome out of view.
        A per-kind budget guarantees each kind survives.
        """
        cutoff = _now() - days * 86400.0
        q = "SELECT * FROM journal WHERE ts >= ?"
        args: list[Any] = [cutoff]
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        q += " ORDER BY id DESC"

        out: dict[str, list[dict[str, Any]]] = {}
        for r in self.conn.execute(q, tuple(args)):
            bucket = out.setdefault(r["kind"], [])
            if len(bucket) < per_kind_limit:
                bucket.append(self._journal_row(r))
        # Oldest-first within each kind reads as a narrative of the week.
        for bucket in out.values():
            bucket.reverse()
        return out

    @staticmethod
    def _journal_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"],
            "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc),
            "kind": r["kind"],
            "actor": r["actor"],
            "payload": json.loads(r["payload"]),
        }

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
        self,
        statement: str,
        *,
        origin_episodes: list[str] | None = None,
        tags: str = "",
        status: str = "candidate",
    ) -> str:
        """Record a lesson. ``candidate`` by default — the historian's
        proposals must earn ``established`` through +3 net supporting
        episodes (see ``add_evidence``).

        ``status='established'`` exists for ONE caller: the operator
        stating a lesson in a hard tone from Telegram. He has standing the
        historian does not — it is his desk, and an instruction phrased as
        an instruction should not have to wait a month of episodes to be
        heard. Tone grading lives in ``copilot.mandates.grade_strength``;
        everything softer than that still arrives as a candidate.
        """
        if status not in ("candidate", "established"):
            raise ValueError(f"lesson status must be candidate|established, got {status!r}")
        lid = _short("ls")
        ts = _now()
        self.conn.execute(
            "INSERT INTO lessons (id, created_ts, statement, tags, status) VALUES (?, ?, ?, ?, ?)",
            (lid, ts, statement, tags, status),
        )
        for eid in origin_episodes or []:
            self.conn.execute(
                "INSERT OR IGNORE INTO lesson_evidence VALUES (?, ?, 'origin', ?)",
                (lid, eid, ts),
            )
        self._write_lesson_card(lid)
        self.journal("lesson_created", {"id": lid, "statement": statement, "status": status})
        return lid

    def set_lesson_status(self, lesson_id: str, status: str) -> bool:
        """Promote or demote a lesson by hand. Append-only in spirit: the
        card and the journal keep every transition, so a lesson the
        operator hardened and later softened reads as a change of mind
        rather than as though it had always been tentative."""
        if status not in ("candidate", "established"):
            raise ValueError(f"lesson status must be candidate|established, got {status!r}")
        cur = self.conn.execute(
            "UPDATE lessons SET status = ? WHERE id = ? AND status != 'retired'",
            (status, lesson_id),
        )
        if not cur.rowcount:
            return False
        self._write_lesson_card(lesson_id)
        self.journal("lesson_status_changed", {"id": lesson_id, "status": status})
        return True

    def operator_lessons(self, status: str = "candidate") -> list[sqlite3.Row]:
        """Operator-authored lessons at ``status``. Tagged rather than kept
        in a separate table so they age through the same lifecycle as the
        historian's."""
        return self.conn.execute(
            "SELECT * FROM lessons WHERE status = ? AND tags LIKE '%operator%' "
            "ORDER BY created_ts DESC",
            (status,),
        ).fetchall()

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

    # ------------------------------------------------------------- shadow

    def add_shadow(
        self,
        *,
        symbol: str,
        origin: str,
        disposition: str,
        rank: int | None = None,
        score: float | None = None,
        why: str = "",
        conditions: dict[str, Any] | None = None,
        px_at: float | None = None,
        pctile_52w: float | None = None,
        ts: float | None = None,
    ) -> str:
        """Record one considered-and-decided name. Never raises on a dup."""
        sid = _short("sh")
        self.conn.execute(
            """INSERT INTO shadow
               (id, ts, symbol, origin, disposition, rank, score, why,
                conditions, px_at, pctile_52w)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sid,
                ts if ts is not None else _now(),
                symbol.upper(),
                origin,
                disposition,
                rank,
                score,
                why[:300],
                json.dumps(conditions or {}, default=str),
                px_at,
                pctile_52w,
            ),
        )
        return sid

    # The legs a shadow row is graded at, and the column pair each fills.
    SHADOW_LEGS: tuple[tuple[int, str, str], ...] = (
        (5, "r5", "bench5"),
        (21, "r21", "bench21"),
        (63, "r63", "bench63"),
    )

    def ungraded_shadow(self, leg_days: int, asof: datetime | None = None) -> list[sqlite3.Row]:
        """Rows old enough for ``leg_days`` whose leg is still empty.

        Legs are filled independently rather than all-at-once at 63d: a
        5-day read available next week is worth more than a complete row
        available in a quarter, and the 5d column is what makes a bad
        selection process visible early."""
        col = {d: c for d, c, _b in self.SHADOW_LEGS}[leg_days]
        cutoff = (asof or datetime.now(tz=timezone.utc)).timestamp() - leg_days * 86400.0
        return self.conn.execute(
            f"SELECT * FROM shadow WHERE {col} IS NULL AND ts <= ? ORDER BY ts",
            (cutoff,),
        ).fetchall()

    def grade_shadow_leg(
        self, shadow_id: str, leg_days: int, *, ret: float, bench: float | None = None
    ) -> None:
        """Fill one forward-return leg. ``bench`` is the benchmark over the
        identical window — a return without it is not a result."""
        col, bcol = {d: (c, b) for d, c, b in self.SHADOW_LEGS}[leg_days]
        graded = ", graded_ts = ?" if leg_days == self.SHADOW_LEGS[-1][0] else ""
        params: list[Any] = [ret, bench]
        if graded:
            params.append(_now())
        params.append(shadow_id)
        self.conn.execute(
            f"UPDATE shadow SET {col} = ?, {bcol} = ?{graded} WHERE id = ?",
            tuple(params),
        )

    def edge_report(self, leg_days: int = 21, since_days: int = 365) -> list[dict[str, Any]]:
        """Picks vs passes, net of the benchmark, grouped by origin.

        The headline number is ``spread`` — mean excess return of taken
        names minus mean excess return of passed names. Positive means the
        selection step added something; negative means the desk would have
        done better with the names it rejected, which is the finding this
        whole table exists to be able to report.

        ``n`` is returned alongside every figure and is not decoration: a
        spread computed on nine names is an anecdote.
        """
        return self._edge_split(
            "origin", leg_days=leg_days, since_days=since_days, label_key="origin"
        )

    # --- the "why" slices -------------------------------------------------
    #
    # /edge answers whether the selection step added value. These answer
    # where the answer comes from. All three are plain SQL over columns
    # already stored at decision time: no model is asked to speculate
    # about causes, because a fluent invented explanation is worse than
    # no explanation — it gets remembered.

    def edge_by_rank(
        self,
        leg_days: int = 21,
        since_days: int = 365,
        buckets: tuple[tuple[str, int, int], ...] = (
            ("1-5", 1, 5),
            ("6-15", 6, 15),
            ("16-30", 16, 30),
        ),
    ) -> list[dict[str, Any]]:
        """Mean excess return by position on the ranked ladder.

        The discrimination test, and the most important of the three. If
        the top bucket beats the bottom, the score works and only the cut
        is misplaced — a tuning problem. If the buckets are flat, the
        score is not ranking anything and the desk is drawing at random
        from a shortlist, which no amount of cut-tuning fixes.

        Reported over all rows regardless of disposition: within a rank
        bucket the taken/passed split is mostly an artefact of where the
        cut fell, so splitting it here would answer a different question.
        """
        case = " ".join(
            f"WHEN rank BETWEEN {lo} AND {hi} THEN '{label}'" for label, lo, hi in buckets
        )
        return self._edge_grouped(
            f"CASE {case} ELSE 'other' END",
            leg_days=leg_days,
            since_days=since_days,
            label_key="rank_bucket",
            where="rank IS NOT NULL",
            order=[label for label, _lo, _hi in buckets],
        )

    def edge_by_condition(
        self, key: str = "vol_bucket", leg_days: int = 21, since_days: int = 365
    ) -> list[dict[str, Any]]:
        """Picks vs passes, split by what the market was doing that day.

        This is the slice that turns a flat verdict into a usable rule.
        "Our picks underperform" is not tradeable; "our picks beat passes
        in high dispersion and lose in low dispersion" is a condition the
        desk can check before sizing.

        ``key`` names a field inside the stored regime fingerprint.
        """
        if not key.replace("_", "").isalnum():
            raise ValueError(f"unsafe condition key: {key!r}")
        return self._edge_split(
            f"json_extract(conditions, '$.{key}')",
            leg_days=leg_days,
            since_days=since_days,
            label_key="condition",
            where=f"json_extract(conditions, '$.{key}') IS NOT NULL",
        )

    def edge_by_entry(
        self,
        leg_days: int = 21,
        since_days: int = 365,
        edges: tuple[float, ...] = (0.5, 0.8, 0.95),
    ) -> list[dict[str, Any]]:
        """Mean excess return by where in the 52-week range the name sat.

        Tests a rule the Quant charter already asserts — that a name at
        the very top of its range is maximally far from any trend stop —
        against what actually happened, rather than leaving it as a
        plausible-sounding instruction in a prompt.
        """
        lo, mid, hi = edges
        case = (
            f"CASE WHEN pctile_52w < {lo} THEN 'below {lo:g}' "
            f"WHEN pctile_52w < {mid} THEN '{lo:g}-{mid:g}' "
            f"WHEN pctile_52w < {hi} THEN '{mid:g}-{hi:g}' "
            f"ELSE 'above {hi:g}' END"
        )
        return self._edge_grouped(
            case,
            leg_days=leg_days,
            since_days=since_days,
            label_key="entry_bucket",
            where="pctile_52w IS NOT NULL",
            order=[f"below {lo:g}", f"{lo:g}-{mid:g}", f"{mid:g}-{hi:g}", f"above {hi:g}"],
        )

    # --- shared machinery -------------------------------------------------

    def _leg_cols(self, leg_days: int) -> tuple[str, str]:
        return {d: (c, b) for d, c, b in self.SHADOW_LEGS}[leg_days]

    def _edge_grouped(
        self,
        group_sql: str,
        *,
        leg_days: int,
        since_days: int,
        label_key: str,
        where: str = "1=1",
        order: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """One row per group: n, distinct names, and mean excess return."""
        col, bcol = self._leg_cols(leg_days)
        rows = self.conn.execute(
            f"""SELECT {group_sql} AS grp, COUNT(*) AS n,
                       COUNT(DISTINCT symbol) AS n_symbols,
                       AVG({col} - COALESCE({bcol}, 0.0)) AS excess
                FROM shadow
                WHERE {col} IS NOT NULL AND ts >= ? AND {where}
                GROUP BY grp""",
            (_now() - since_days * 86400.0,),
        ).fetchall()
        out = [
            {
                label_key: r["grp"],
                "n": r["n"],
                "n_symbols": r["n_symbols"],
                "excess": r["excess"],
                "leg_days": leg_days,
            }
            for r in rows
        ]
        if order:
            rank = {label: i for i, label in enumerate(order)}
            out.sort(key=lambda s: rank.get(str(s[label_key]), len(rank)))
        else:
            out.sort(key=lambda s: -s["n"])
        return out

    def _edge_split(
        self,
        group_sql: str,
        *,
        leg_days: int,
        since_days: int,
        label_key: str,
        where: str = "1=1",
    ) -> list[dict[str, Any]]:
        """One row per group, split into taken vs passed with the spread.

        Both a row count and a distinct-symbol count come back. The ladder
        re-ranks the same names every day, so rows accumulate at ~30/day
        over a universe that turns over slowly, and each row's forward
        return overlaps its neighbour's by all but one day. A raw n of
        1200 can be forty independent observations wearing a convincing
        costume — so callers gate their thin-sample warnings on
        ``n_*_symbols`` and show ``n_*`` only as context.
        """
        col, bcol = self._leg_cols(leg_days)
        rows = self.conn.execute(
            f"""SELECT {group_sql} AS grp,
                       CASE WHEN disposition = 'taken' THEN 'taken' ELSE 'passed' END AS side,
                       COUNT(*) AS n,
                       COUNT(DISTINCT symbol) AS n_symbols,
                       AVG({col}) AS ret,
                       AVG({col} - COALESCE({bcol}, 0.0)) AS excess
                FROM shadow
                WHERE {col} IS NOT NULL AND ts >= ? AND {where}
                GROUP BY grp, side""",
            (_now() - since_days * 86400.0,),
        ).fetchall()

        by_group: dict[str, dict[str, Any]] = {}
        for r in rows:
            slot = by_group.setdefault(r["grp"], {label_key: r["grp"], "leg_days": leg_days})
            slot[f"n_{r['side']}"] = r["n"]
            slot[f"n_{r['side']}_symbols"] = r["n_symbols"]
            slot[f"{r['side']}_ret"] = r["ret"]
            slot[f"{r['side']}_excess"] = r["excess"]

        out: list[dict[str, Any]] = []
        for slot in by_group.values():
            taken = slot.get("taken_excess")
            passed = slot.get("passed_excess")
            # A spread needs both sides. A group that only ever produces
            # 'taken' rows (a mandate the desk always honours) has no
            # counterfactual and must report None rather than zero.
            slot["spread"] = (
                None if taken is None or passed is None else float(taken) - float(passed)
            )
            for key in ("n_taken", "n_passed", "n_taken_symbols", "n_passed_symbols"):
                slot.setdefault(key, 0)
            out.append(slot)
        out.sort(key=lambda s: -(s["n_taken"] + s["n_passed"]))
        return out

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
            "shadow": c.execute("SELECT COUNT(*) FROM shadow").fetchone()[0],
            "dossiers": len(self.dossiers()),
        }


def default_store() -> MemoryStore:
    """Production store under ``settings.state_dir / memory``."""
    from trading.core.config import settings

    return MemoryStore(Path(settings.state_dir) / "memory")


__all__ = ["MemoryStore", "default_store"]
