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
import math
import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
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
    status      TEXT NOT NULL DEFAULT 'candidate',  -- candidate|established|challenged|retired
    support     INTEGER NOT NULL DEFAULT 0,
    contradict  INTEGER NOT NULL DEFAULT 0,
    retired_ts  REAL,
    retired_why TEXT,
    tags        TEXT NOT NULL DEFAULT '',
    conditions  TEXT NOT NULL DEFAULT '{}',  -- structured regime snapshot + stated scope
    last_reviewed_ts REAL                    -- review rotation only; never an expiry clock
);

CREATE TABLE IF NOT EXISTS lesson_evidence (
    lesson_id   TEXT NOT NULL,
    episode_id  TEXT NOT NULL,
    relation    TEXT NOT NULL,           -- supports|contradicts|origin
    ts          REAL NOT NULL,
    evidence_kind TEXT NOT NULL DEFAULT 'review',  -- outcome|review|origin
    reason      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (lesson_id, episode_id, relation)
);

-- One immutable review card for each Learning Curator pass.  The lesson
-- tables hold the current state; these tables answer the equally important
-- question of *why* a state was reviewed or proposed for archival.
CREATE TABLE IF NOT EXISTS curator_runs (
    id          TEXT PRIMARY KEY,       -- cr-<uuid8>
    ts          REAL NOT NULL,
    status      TEXT NOT NULL,          -- completed|degraded|failed
    conditions  TEXT NOT NULL DEFAULT '{}',
    reviewed    INTEGER NOT NULL DEFAULT 0,
    created     INTEGER NOT NULL DEFAULT 0,
    voted       INTEGER NOT NULL DEFAULT 0,
    vote_ok     INTEGER,                -- NULL when no vote was needed
    archive_recommendations INTEGER NOT NULL DEFAULT 0,
    reason      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_curator_runs_ts ON curator_runs(ts);

CREATE TABLE IF NOT EXISTS curator_actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    lesson_id     TEXT,
    rank          INTEGER,
    action        TEXT NOT NULL,        -- reviewed|created|promoted|challenged|archive_recommended
    before_status TEXT,
    after_status  TEXT,
    reason        TEXT NOT NULL DEFAULT '',
    evidence_ids  TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_curator_actions_run ON curator_actions(run_id, id);
CREATE INDEX IF NOT EXISTS idx_curator_actions_lesson ON curator_actions(lesson_id);

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
    snapshot     TEXT NOT NULL DEFAULT '{}',  -- immutable decision provenance
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


_LESSON_STATUSES = frozenset(("candidate", "established", "challenged", "retired"))
_ACTIVE_LESSON_STATUSES = frozenset(("candidate", "established", "challenged"))
_PROMOTION_EVIDENCE_KINDS = frozenset(("outcome",))
_MIN_OUTCOME_EVIDENCE = 3
_CURATOR_STALE_AFTER = timedelta(days=5)
_LESSON_LIFECYCLE_KINDS = (
    "lesson_created",
    "lesson_status_changed",
    "lesson_established",
    "lesson_challenged",
    "lesson_retired",
    "lesson_restored",
)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_RETRIEVAL_STOP_WORDS = frozenset(
    {
        "after",
        "against",
        "always",
        "because",
        "before",
        "broad",
        "could",
        "days",
        "does",
        "from",
        "have",
        "into",
        "market",
        "more",
        "only",
        "over",
        "should",
        "that",
        "than",
        "their",
        "there",
        "these",
        "this",
        "under",
        "when",
        "with",
        "within",
    }
)


def lesson_condition_fingerprint(context: dict[str, Any]) -> dict[str, Any]:
    """Compact, reproducible snapshot used to retrieve rather than expire lessons.

    A regime can persist for months and may go quiet before enough outcome
    evidence arrives. We therefore never use age as a truth judgement. The
    snapshot answers the narrower question, "which old claim is worth
    reopening *today*?" Missing monitors simply omit a key; pretending a
    missing VIX or macro read is a neutral regime would be false precision.
    """
    out: dict[str, Any] = {}
    macro = context.get("macro_dial")
    if isinstance(macro, dict):
        raw_composite = macro.get("composite")
        if isinstance(raw_composite, (str, int, float)):
            try:
                composite = float(raw_composite)
            except ValueError:
                composite = None
            if composite is not None and math.isfinite(composite):
                out["macro_bucket"] = (
                    "stress" if composite >= 1.5 else "easing" if composite <= -1.5 else "neutral"
                )
    vol = context.get("vol_surface")
    if isinstance(vol, dict):
        raw_atm_iv = vol.get("atm_iv")
        if isinstance(raw_atm_iv, (str, int, float)):
            try:
                atm_iv = float(raw_atm_iv)
            except ValueError:
                atm_iv = None
            if atm_iv is not None and math.isfinite(atm_iv):
                out["vol_bucket"] = (
                    "low"
                    if atm_iv < 0.16
                    else "normal"
                    if atm_iv < 0.25
                    else "elevated"
                    if atm_iv < 0.35
                    else "stress"
                )
    style = context.get("style_leader")
    if isinstance(style, str) and style.strip():
        out["style_leader"] = style.strip()
    triggers = context.get("spy_vix_triggers")
    if isinstance(triggers, list):
        names = sorted(
            {
                str(item.get("name", "")).strip()
                for item in triggers
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            }
        )
        if names:
            out["triggers"] = names[:8]
    return out


def _stored_conditions(row: sqlite3.Row) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(regime_snapshot, stated_scope)`` from current or legacy rows."""
    try:
        raw = json.loads(row["conditions"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        return {}, {}
    snapshot = raw.get("snapshot", raw)
    scope = raw.get("scope", {})
    return (
        dict(snapshot) if isinstance(snapshot, dict) else {},
        dict(scope) if isinstance(scope, dict) else {},
    )


def _condition_match(
    stored: dict[str, Any], current: dict[str, Any]
) -> tuple[float, list[str], list[str]]:
    """A small transparent relevance score — no opaque embedding or decay."""
    matched: list[str] = []
    different: list[str] = []
    compared = 0
    for key in ("macro_bucket", "vol_bucket", "style_leader"):
        if key in stored and key in current:
            compared += 1
            if stored[key] == current[key]:
                matched.append(key)
            else:
                different.append(key)
    stored_triggers = {str(x) for x in stored.get("triggers", [])}
    current_triggers = {str(x) for x in current.get("triggers", [])}
    if stored_triggers and current_triggers:
        compared += 1
        overlap = stored_triggers & current_triggers
        if overlap:
            matched.append("triggers:" + ",".join(sorted(overlap)[:3]))
        else:
            different.append("triggers")
    return (len(matched) / compared if compared else 0.0), matched, different


def is_operator_lesson_tags(tags: object) -> bool:
    """Whether tags carry the exact, reserved operator-authorship token.

    ``/lesson`` writes ``tags="operator <strength>"``; the historian's own
    proposals never carry that token.  Only whitespace/comma/semicolon
    delimiters are accepted, so tags such as ``operator-ish`` or
    ``cooperator`` cannot smuggle a machine-generated rule into the
    guaranteed slots.
    """
    words = re.split(r"[\s,;]+", str(tags or "").lower().strip())
    return "operator" in words


def _is_operator_lesson(card: dict[str, Any]) -> bool:
    """True for a lesson the operator stated himself."""
    return is_operator_lesson_tags(card.get("tags"))


def _condition_signature(conditions: dict[str, Any]) -> str:
    """Stable comparison key that also works when a fingerprint has lists."""
    return json.dumps(conditions, sort_keys=True, separators=(",", ":"), default=str)


def _keywords(text: str) -> set[str]:
    """Small, deterministic retrieval vocabulary; never an opaque embedding.

    The historian's memory must be auditable. Lexical retrieval is less
    glamorous than a vector store, but every match can be shown to the
    operator and recomputed from the canonical memory tables.
    """
    return {
        word
        for word in _WORD_RE.findall(text.lower())
        if word not in _RETRIEVAL_STOP_WORDS and not word.isdecimal()
    }


def _evidence_kind(evidence_id: str, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in ("outcome", "review", "origin"):
            raise ValueError(f"unknown lesson evidence kind: {explicit!r}")
        return explicit
    # Predictions and closed trading episodes are independently recorded
    # outcomes. A synthetic week tag is useful review provenance but must
    # never promote a permanent desk belief on its own.
    return "outcome" if evidence_id.startswith(("pr-", "ep-")) else "review"


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
        self.reviews_dir = self.root / "reviews"
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
        for table, column, decl in (
            ("shadow", "pctile_52w", "REAL"),
            ("lesson_evidence", "evidence_kind", "TEXT NOT NULL DEFAULT 'review'"),
            ("lesson_evidence", "reason", "TEXT NOT NULL DEFAULT ''"),
            ("lessons", "conditions", "TEXT NOT NULL DEFAULT '{}'"),
            ("lessons", "last_reviewed_ts", "REAL"),
            ("shadow", "snapshot", "TEXT NOT NULL DEFAULT '{}'"),
        ):
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
        conditions: dict[str, Any] | None = None,
        actor: str = "system",
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
            """INSERT INTO lessons (id, created_ts, statement, tags, status, conditions)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (lid, ts, statement, tags, status, json.dumps(conditions or {}, default=str)),
        )
        for eid in origin_episodes or []:
            self.conn.execute(
                """INSERT OR IGNORE INTO lesson_evidence
                   (lesson_id, episode_id, relation, ts, evidence_kind, reason)
                   VALUES (?, ?, 'origin', ?, 'origin', '')""",
                (lid, eid, ts),
            )
        self.journal(
            "lesson_created",
            {"id": lid, "statement": statement, "status": status, "conditions": conditions or {}},
            actor=actor,
        )
        self._write_lesson_card(lid)
        return lid

    def set_lesson_status(self, lesson_id: str, status: str, *, actor: str = "system") -> bool:
        """Promote or demote a lesson by hand. Append-only in spirit: the
        card and the journal keep every transition, so a lesson the
        operator hardened and later softened reads as a change of mind
        rather than as though it had always been tentative."""
        if status not in _ACTIVE_LESSON_STATUSES:
            raise ValueError(
                f"lesson status must be candidate|established|challenged, got {status!r}"
            )
        old = self.conn.execute("SELECT status FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        if old is None or old["status"] == "retired":
            return False
        cur = self.conn.execute(
            "UPDATE lessons SET status = ? WHERE id = ? AND status != 'retired'",
            (status, lesson_id),
        )
        if not cur.rowcount:
            return False
        self.journal(
            "lesson_status_changed",
            {
                "id": lesson_id,
                "status": status,
                "from_status": old["status"],
                "to_status": status,
            },
            actor=actor,
        )
        self._write_lesson_card(lesson_id)
        return True

    def operator_lessons(self, status: str = "candidate") -> list[sqlite3.Row]:
        """Operator-authored lessons at ``status``. Tagged rather than kept
        in a separate table so they age through the same lifecycle as the
        historian's."""
        rows = self.conn.execute(
            "SELECT * FROM lessons WHERE status = ? ORDER BY created_ts DESC", (status,)
        ).fetchall()
        return [row for row in rows if is_operator_lesson_tags(row["tags"])]

    def completed_outcome_ids(self, evidence_ids: list[str] | set[str]) -> set[str]:
        """Return only real, completed prediction/episode identifiers.

        A journal row says an outcome was *reported* to a reviewer. This
        method is the persistence-side check that it was actually measured,
        so a synthetic or malformed journal event cannot become provenance
        for a durable lesson.
        """
        requested = {str(value) for value in evidence_ids if isinstance(value, str)}
        prediction_ids = sorted(value for value in requested if value.startswith("pr-"))
        episode_ids = sorted(value for value in requested if value.startswith("ep-"))
        verified: set[str] = set()
        if prediction_ids:
            marks = ",".join("?" for _ in prediction_ids)
            rows = self.conn.execute(
                f"SELECT id FROM predictions WHERE graded_ts IS NOT NULL AND id IN ({marks})",
                prediction_ids,
            ).fetchall()
            verified.update(str(row["id"]) for row in rows)
        if episode_ids:
            marks = ",".join("?" for _ in episode_ids)
            rows = self.conn.execute(
                f"SELECT id FROM episodes WHERE id IN ({marks})", episode_ids
            ).fetchall()
            verified.update(str(row["id"]) for row in rows)
        return verified

    def add_evidence(
        self,
        lesson_id: str,
        episode_id: str,
        *,
        supports: bool,
        reason: str = "",
        evidence_kind: str | None = None,
        actor: str = "system",
    ) -> bool:
        """Record one distinct item of lesson evidence and return whether it was new.

        The primary key is the deduplication boundary. Incrementing the
        lesson counter after an ignored insert made a repeat delivery look
        like fresh support, which is especially dangerous now that the
        historian reviews evidence twice a week. Only an independently
        recorded ``outcome`` (a graded prediction or closed episode) can
        promote a candidate; a historian's weekly review remains visible
        but is deliberately insufficient by itself.
        """
        rel = "supports" if supports else "contradicts"
        kind = _evidence_kind(episode_id, evidence_kind)
        lesson = self.conn.execute(
            "SELECT status FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        # Archived lessons preserve their evidence and card, but must not
        # silently start influencing the active book again.  An explicit
        # restore puts the lesson back into ``candidate`` before fresh
        # evidence can be considered.
        if lesson is None or lesson["status"] == "retired":
            return False
        if kind == "outcome" and episode_id not in self.completed_outcome_ids([episode_id]):
            return False
        # A realized outcome cannot honestly both support and contradict the
        # same claim. The old primary key included ``relation``, which made
        # that contradiction possible on a retry with flipped model output.
        if self.conn.execute(
            "SELECT 1 FROM lesson_evidence WHERE lesson_id = ? AND episode_id = ? LIMIT 1",
            (lesson_id, episode_id),
        ).fetchone():
            return False
        inserted = self.conn.execute(
            """INSERT OR IGNORE INTO lesson_evidence
               (lesson_id, episode_id, relation, ts, evidence_kind, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (lesson_id, episode_id, rel, _now(), kind, reason[:500]),
        )
        if not inserted.rowcount:
            return False
        col = "support" if supports else "contradict"
        self.conn.execute(f"UPDATE lessons SET {col} = {col} + 1 WHERE id = ?", (lesson_id,))
        # Promotion requires measured outcomes, not repeated LLM review.
        row = self.conn.execute(
            "SELECT status, support, contradict FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        outcome_counts = self.conn.execute(
            """SELECT
                 SUM(CASE WHEN relation = 'supports' THEN 1 ELSE 0 END) AS support,
                 SUM(CASE WHEN relation = 'contradicts' THEN 1 ELSE 0 END) AS contradict
               FROM lesson_evidence
               WHERE lesson_id = ? AND evidence_kind IN ('outcome')""",
            (lesson_id,),
        ).fetchone()
        outcome_support = int(outcome_counts["support"] or 0) if outcome_counts else 0
        outcome_contradict = int(outcome_counts["contradict"] or 0) if outcome_counts else 0
        if (
            row
            and row["status"] == "candidate"
            and outcome_support - outcome_contradict >= _MIN_OUTCOME_EVIDENCE
        ):
            self.conn.execute(
                "UPDATE lessons SET status = 'established' WHERE id = ?", (lesson_id,)
            )
            self.journal(
                "lesson_established",
                {
                    "id": lesson_id,
                    "outcome_support": outcome_support,
                    "outcome_contradict": outcome_contradict,
                },
                actor=actor,
            )
        elif (
            row
            and row["status"] == "established"
            and outcome_contradict >= _MIN_OUTCOME_EVIDENCE
            and outcome_contradict >= outcome_support
        ):
            # Challenged lessons immediately leave the agent context, but
            # remain visible to the historian and operator until a review
            # retires or manually restores them. This is safer than letting
            # a once-established belief continue to influence a trade while
            # its empirical support has reversed.
            self.conn.execute("UPDATE lessons SET status = 'challenged' WHERE id = ?", (lesson_id,))
            self.journal(
                "lesson_challenged",
                {
                    "id": lesson_id,
                    "outcome_support": outcome_support,
                    "outcome_contradict": outcome_contradict,
                },
                actor=actor,
            )
        self._write_lesson_card(lesson_id)
        return True

    def retire_lesson(self, lesson_id: str, why: str, *, actor: str = "system") -> bool:
        """Retired, never deleted — the card keeps its full history."""
        reason = " ".join(why.split())[:500]
        cur = self.conn.execute(
            """UPDATE lessons SET status='retired', retired_ts=?, retired_why=?
               WHERE id=? AND status != 'retired'""",
            (_now(), reason, lesson_id),
        )
        if not cur.rowcount:
            return False
        self.journal("lesson_retired", {"id": lesson_id, "why": reason}, actor=actor)
        self._write_lesson_card(lesson_id)
        return True

    def restore_retired_lesson(self, lesson_id: str, why: str, *, actor: str = "system") -> bool:
        """Restore an archived lesson to ``candidate`` with an audit trail.

        Restoration is deliberately conservative: it never makes a prior
        belief active again.  A candidate remains outside agent context, but
        can earn establishment through fresh measured outcomes.  The previous
        archive timestamp and reason remain on the lesson card, while the
        immutable journal records who restored it and why.
        """
        reason = " ".join(why.split())[:500]
        if not reason:
            raise ValueError("a restoration reason is required")
        row = self.conn.execute("SELECT status FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        if row is None or row["status"] != "retired":
            return False
        self.conn.execute(
            "UPDATE lessons SET status = 'candidate', last_reviewed_ts = NULL WHERE id = ?",
            (lesson_id,),
        )
        self.journal(
            "lesson_restored",
            {"id": lesson_id, "from_status": "retired", "to_status": "candidate", "why": reason},
            actor=actor,
        )
        self._write_lesson_card(lesson_id)
        return True

    def lessons(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            if status not in _LESSON_STATUSES:
                raise ValueError(f"unknown lesson status: {status!r}")
            return self.conn.execute(
                "SELECT * FROM lessons WHERE status = ? ORDER BY support - contradict DESC",
                (status,),
            ).fetchall()
        return self.conn.execute("SELECT * FROM lessons ORDER BY created_ts DESC").fetchall()

    def _lesson_cards_for_status(
        self, status: str, current_conditions: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # ``lessons.support`` / ``contradict`` intentionally include review
        # provenance.  That is useful history, but it is not measured market
        # evidence and must never decide which machine-authored lesson wins a
        # scarce prompt slot.  Calculate outcome-only counts at the query
        # boundary so every reviewer and retriever uses the same rule.
        rows = self.conn.execute(
            """SELECT l.*,
                      COALESCE(SUM(CASE WHEN e.evidence_kind = 'outcome'
                                           AND e.relation = 'supports' THEN 1 ELSE 0 END), 0)
                          AS outcome_support,
                      COALESCE(SUM(CASE WHEN e.evidence_kind = 'outcome'
                                           AND e.relation = 'contradicts' THEN 1 ELSE 0 END), 0)
                          AS outcome_contradict
               FROM lessons AS l
               LEFT JOIN lesson_evidence AS e ON e.lesson_id = l.id
               WHERE l.status = ?
               GROUP BY l.id""",
            (status,),
        ).fetchall()
        cards: list[dict[str, Any]] = []
        for row in rows:
            snapshot, scope = _stored_conditions(row)
            relevance, matched, different = _condition_match(snapshot, current_conditions)
            outcome_support = int(row["outcome_support"] or 0)
            outcome_contradict = int(row["outcome_contradict"] or 0)
            cards.append(
                {
                    "id": row["id"],
                    "status": row["status"],
                    "statement": row["statement"],
                    "support": int(row["support"]),
                    "contradict": int(row["contradict"]),
                    "outcome_support": outcome_support,
                    "outcome_contradict": outcome_contradict,
                    "outcome_observations": outcome_support + outcome_contradict,
                    "tags": row["tags"],
                    "conditions": snapshot,
                    "scope": scope,
                    "relevance": relevance,
                    "matched_conditions": matched,
                    "different_conditions": different,
                    "last_reviewed_ts": row["last_reviewed_ts"],
                    "created_ts": float(row["created_ts"]),
                }
            )
        return cards

    @staticmethod
    def _evidence_strength(card: dict[str, Any]) -> int:
        """Measured-outcome strength used for machine lesson ranking only."""
        return int(card["outcome_support"]) - int(card["outcome_contradict"])

    def retrieve_lessons(
        self,
        current_conditions: dict[str, Any],
        *,
        status: str = "established",
        max_relevant: int = 3,
        max_diversifiers: int = 2,
    ) -> list[dict[str, Any]]:
        """Return a small, explainable mix of relevant and broad priors.

        Similarity is a retrieval aid, never a veto. Most slots favour
        matched conditions; the remaining slots deliberately carry durable
        lessons from a different or unknown setting so a long-lived regime
        cannot turn the prompt into a self-confirming echo chamber.
        """
        if status not in _ACTIVE_LESSON_STATUSES:
            raise ValueError(f"retrieval status must be active, got {status!r}")
        cards = self._lesson_cards_for_status(status, current_conditions)
        selected: list[dict[str, Any]] = []

        # Operator-stated lessons are seated FIRST, outside the ranking.
        #
        # They used to be effectively unreachable. `/lesson` stores with no
        # conditions, so `_condition_match` scores them 0.0 and the loop
        # below skips every zero-relevance card; they then landed in the
        # `broad` bucket, which guarantees exactly one slot, awarded by
        # support minus contradict. A freshly stated operator lesson has zero
        # evidence, so any legacy rule with one supporting episode outranked
        # it and the operator's own conclusion never reached a prompt.
        #
        # Ranking the operator against the historian was the category
        # error. His lessons are not regime-matched heuristics competing on
        # evidence; they are standing beliefs the desk is meant to hold.
        # They take slots off the top and the ranked picks fill the rest.
        operator_cards = sorted(
            (c for c in cards if _is_operator_lesson(c)),
            key=lambda c: c["created_ts"],
            reverse=True,
        )
        for card in operator_cards[:max_relevant]:
            selected.append({**card, "retrieval_role": "operator_stated"})

        ranked = sorted(
            (c for c in cards if not _is_operator_lesson(c)),
            key=lambda c: (c["relevance"], self._evidence_strength(c), c["created_ts"]),
            reverse=True,
        )
        for card in ranked:
            if card["relevance"] <= 0 or len(selected) >= max_relevant:
                continue
            role = "regime_match" if card["relevance"] == 1.0 else "partial_match"
            selected.append({**card, "retrieval_role": role})

        # The diversifier selection is deterministic and condition-distinct.
        # A legacy lesson (no snapshot) is a broad prior, not discarded data.
        seen = {_condition_signature(c["conditions"]) for c in selected}
        remaining = [c for c in cards if c["id"] not in {s["id"] for s in selected}]
        # Reserve the first counterweight for a legacy broad prior when one
        # exists. It is maximally different from a current-regime claim,
        # but still ranked by evidence rather than being a random weak rule.
        broad = sorted(
            (c for c in remaining if not c["conditions"]),
            key=lambda c: (self._evidence_strength(c), c["created_ts"]),
            reverse=True,
        )
        diverse = sorted(
            (c for c in remaining if c["conditions"]),
            key=lambda c: (c["relevance"], -self._evidence_strength(c), c["created_ts"]),
        )
        # Put one broad prior in first, then prefer genuinely different
        # regimes. If the vault predates snapshots, more than one legacy
        # lesson may fill otherwise-empty counterweight slots; identical
        # provenance must not make every older lesson permanently invisible.
        if broad and len(selected) < max_relevant + max_diversifiers:
            selected.append({**broad[0], "retrieval_role": "broad_prior"})
        for card in diverse:
            if len(selected) >= max_relevant + max_diversifiers:
                break
            signature = _condition_signature(card["conditions"])
            if signature in seen:
                continue
            seen.add(signature)
            selected.append({**card, "retrieval_role": "diversifier"})
        for card in broad[1:]:
            if len(selected) >= max_relevant + max_diversifiers:
                break
            selected.append({**card, "retrieval_role": "broad_prior"})
        return selected

    def candidate_review_queue(
        self, current_conditions: dict[str, Any], *, limit: int = 12
    ) -> list[dict[str, Any]]:
        """Bound weekly candidate review without treating silence as failure.

        No candidate is expired, demoted or forgotten. This queue merely
        decides which bounded set gets the next scarce reviewer pass. A
        condition match wins; otherwise the least-recently-reviewed
        candidate rotates in, so an enduring but quiet regime is revisited.
        """
        cards = self._lesson_cards_for_status("candidate", current_conditions)
        return sorted(
            cards,
            key=lambda c: (
                -float(c["relevance"]),
                float(c["last_reviewed_ts"] or 0.0),
                -self._evidence_strength(c),
                float(c["created_ts"]),
            ),
        )[:limit]

    def lessons_for_historian_review(
        self, current_conditions: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """The bounded rulebook for one Historian pass, never a deletion policy."""
        return [
            *self.candidate_review_queue(current_conditions, limit=12),
            *self.retrieve_lessons(
                current_conditions, status="established", max_relevant=4, max_diversifiers=1
            ),
            *self.retrieve_lessons(
                current_conditions, status="challenged", max_relevant=2, max_diversifiers=1
            ),
        ]

    def lesson_review(
        self, current_conditions: dict[str, Any], *, candidate_limit: int = 12
    ) -> dict[str, Any]:
        """Return the deterministic worklist for one Learning Curator pass.

        This is deliberately a review allocator, not an expiry engine.  Age
        only rotates scarce attention; measured outcomes determine standing.
        Archive recommendations are advisory and require an operator-approved
        desk change before a lesson leaves the active vault.
        """
        candidates = self.candidate_review_queue(current_conditions, limit=candidate_limit)
        established = self.retrieve_lessons(
            current_conditions, status="established", max_relevant=4, max_diversifiers=1
        )
        challenged = self.retrieve_lessons(
            current_conditions, status="challenged", max_relevant=2, max_diversifiers=1
        )
        queue = [*candidates, *established, *challenged]
        review_actions: list[dict[str, Any]] = []
        for rank, card in enumerate(queue, start=1):
            net = self._evidence_strength(card)
            if card["status"] == "candidate":
                action = "awaiting_evidence" if card["outcome_observations"] == 0 else "reviewed"
                reason = (
                    "No completed outcome linked yet; keep as a candidate."
                    if action == "awaiting_evidence"
                    else f"{net:+d} net measured outcomes; candidate remains evidence-gated."
                )
            elif card["status"] == "established":
                action = "kept"
                reason = f"Active with {net:+d} net measured outcomes."
            else:
                action = "reviewed"
                reason = (
                    f"Challenged with {net:+d} net measured outcomes; excluded from agent context."
                )
            review_actions.append(
                {
                    "lesson_id": card["id"],
                    "rank": rank,
                    "action": action,
                    "before_status": card["status"],
                    "after_status": card["status"],
                    "reason": reason,
                    "evidence_ids": [
                        evidence["id"]
                        for evidence in self.lesson_evidence(card["id"], limit=8)
                        if evidence["kind"] == "outcome"
                    ],
                }
            )

        all_established = self._lesson_cards_for_status("established", current_conditions)
        ranked_established = sorted(
            all_established,
            key=lambda c: (
                self._evidence_strength(c),
                int(c["outcome_observations"]),
                float(c["relevance"]),
                float(c["created_ts"]),
            ),
            reverse=True,
        )
        all_challenged = self._lesson_cards_for_status("challenged", current_conditions)
        archive_recommendations: list[dict[str, Any]] = []
        for card in all_challenged:
            if _is_operator_lesson(card):
                continue
            support, contradict = int(card["outcome_support"]), int(card["outcome_contradict"])
            if contradict < _MIN_OUTCOME_EVIDENCE or contradict < support:
                continue
            archive_recommendations.append(
                {
                    "lesson_id": card["id"],
                    "statement": card["statement"],
                    "outcome_support": support,
                    "outcome_contradict": contradict,
                    "reason": (
                        f"{contradict} measured contradictions versus {support} supports; "
                        "machine-authored lesson is already challenged."
                    ),
                    "requires_operator_approval": True,
                }
            )
        archive_recommendations.sort(
            key=lambda c: (
                int(c["outcome_contradict"]) - int(c["outcome_support"]),
                c["lesson_id"],
            ),
            reverse=True,
        )

        counts = {status: 0 for status in _LESSON_STATUSES}
        for row in self.conn.execute("SELECT status, COUNT(*) AS n FROM lessons GROUP BY status"):
            counts[str(row["status"])] = int(row["n"])
        candidate_unreviewed = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM lessons WHERE status = 'candidate' AND last_reviewed_ts IS NULL"
            ).fetchone()[0]
        )
        return {
            "status_counts": counts,
            "candidate_unreviewed": candidate_unreviewed,
            "queue": queue,
            "review_actions": review_actions,
            "ranked_established": ranked_established,
            "archive_recommendations": archive_recommendations,
        }

    def archive_challenged_lesson(self, lesson_id: str, why: str, *, actor: str = "system") -> bool:
        """Archive only a measured, challenged machine lesson after approval.

        This is intentionally narrower than ``retire_lesson``: the latter is
        the existing explicit operator path, while this method enforces the
        Curator's evidence and authorship rules at the persistence boundary.
        """
        recommendation = next(
            (
                row
                for row in self.lesson_review({})["archive_recommendations"]
                if row["lesson_id"] == lesson_id
            ),
            None,
        )
        if recommendation is None:
            return False
        reason = " ".join(why.split())[:500] or str(recommendation["reason"])
        return self.retire_lesson(lesson_id, reason, actor=actor)

    def mark_lessons_reviewed(self, lesson_ids: list[str]) -> None:
        """Record review rotation only; it never changes a lesson's standing."""
        ids = list(dict.fromkeys(str(lid) for lid in lesson_ids if str(lid).startswith("ls-")))
        if not ids:
            return
        marks = ",".join("?" for _ in ids)
        self.conn.execute(
            f"UPDATE lessons SET last_reviewed_ts = ? WHERE status != 'retired' AND id IN ({marks})",
            (_now(), *ids),
        )

    def lesson_evidence(self, lesson_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """Auditable evidence cards for a lesson, newest first.

        Evidence is returned with its provenance rather than only a counter.
        The Historian can therefore distinguish a reviewed claim from one
        tied to a realized prediction or completed trade.
        """
        rows = self.conn.execute(
            """SELECT episode_id, relation, ts, evidence_kind, reason
               FROM lesson_evidence WHERE lesson_id = ?
               ORDER BY ts DESC LIMIT ?""",
            (lesson_id, limit),
        ).fetchall()
        return [
            {
                "id": r["episode_id"],
                "relation": r["relation"],
                "kind": r["evidence_kind"],
                "ts": datetime.fromtimestamp(r["ts"], tz=timezone.utc).date().isoformat(),
                "reason": r["reason"],
            }
            for r in rows
        ]

    def _lesson_lifecycle_events(
        self,
        *,
        lesson_id: str | None = None,
        limit: int | None = None,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Read lesson-only lifecycle events without chatty journal rows.

        The journal intentionally stores every system event. Lifecycle views
        must query their own event family rather than taking a generic tail,
        otherwise a busy committee can hide the latest archive or restore.
        """
        marks = ",".join("?" for _ in _LESSON_LIFECYCLE_KINDS)
        order = "DESC" if newest_first else "ASC"
        query = f"SELECT * FROM journal WHERE kind IN ({marks}) ORDER BY id {order}"
        args: list[Any] = list(_LESSON_LIFECYCLE_KINDS)
        if limit is not None and lesson_id is None:
            query += " LIMIT ?"
            args.append(max(0, limit))
        events: list[dict[str, Any]] = []
        for row in self.conn.execute(query, args):
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if lesson_id is not None and payload.get("id") != lesson_id:
                continue
            events.append(
                {
                    "ts": datetime.fromtimestamp(row["ts"], tz=timezone.utc),
                    "kind": str(row["kind"]),
                    "actor": str(row["actor"]),
                    "payload": payload,
                }
            )
            if limit is not None and lesson_id is not None and len(events) >= limit:
                break
        return events

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
        snapshot, scope = _stored_conditions(row)
        if snapshot or scope:
            lines += [
                "",
                "## Retrieval context",
                "",
                "```json",
                json.dumps({"snapshot": snapshot, "scope": scope}, indent=2, sort_keys=True),
                "```",
            ]
        for e in ev:
            ts = datetime.fromtimestamp(e["ts"], tz=timezone.utc).date().isoformat()
            tail = f" — {e['reason']}" if e["reason"] else ""
            lines.append(
                f"- {ts} **{e['relation']}** ({e['evidence_kind']}) [[{e['episode_id']}]]{tail}"
            )
        if row["status"] == "retired":
            died = datetime.fromtimestamp(row["retired_ts"], tz=timezone.utc).date().isoformat()
            lines += ["", f"## Archived (Retired) {died}", "", row["retired_why"] or ""]
        elif row["retired_ts"] is not None:
            archived = datetime.fromtimestamp(row["retired_ts"], tz=timezone.utc).date().isoformat()
            lines += [
                "",
                f"## Previously archived {archived}",
                "",
                row["retired_why"] or "",
            ]
        lifecycle_labels = {
            "lesson_created": "created",
            "lesson_status_changed": "status changed",
            "lesson_established": "established",
            "lesson_challenged": "challenged",
            "lesson_retired": "archived",
            "lesson_restored": "restored",
        }
        lifecycle = self._lesson_lifecycle_events(lesson_id=lesson_id, newest_first=False)
        if lifecycle:
            lines += ["", "## Lifecycle", ""]
            for event in lifecycle:
                payload = event["payload"]
                detail = str(payload.get("why", "")).strip()
                if not detail and event["kind"] == "lesson_status_changed":
                    before = str(payload.get("from_status", "")).strip()
                    after = str(payload.get("to_status", payload.get("status", ""))).strip()
                    detail = " → ".join(value for value in (before, after) if value)
                suffix = f" — {detail}" if detail else ""
                stamp = event["ts"].date().isoformat()
                label = lifecycle_labels.get(event["kind"], event["kind"])
                lines.append(f"- {stamp} **{label}** by {event['actor']}{suffix}")
        _atomic_write(self.lessons_dir / f"{lesson_id}.md", "\n".join(lines) + "\n")

    def record_curator_run(
        self,
        *,
        status: str,
        conditions: dict[str, Any],
        reviewed: int,
        created: int,
        voted: int,
        vote_ok: bool | None,
        archive_recommendations: int,
        actions: list[dict[str, Any]],
        reason: str = "",
        ts: datetime | None = None,
    ) -> str:
        """Persist one immutable Learning Curator review and its actions.

        The current lesson row answers what the desk believes now.  This
        report answers what was seen, ranked, deferred, or recommended on a
        particular Tuesday/Friday pass.  It is advisory-only and has no path
        to strategy, risk, or broker state.
        """
        if status not in {"completed", "degraded", "failed"}:
            raise ValueError(f"unknown curator run status: {status!r}")
        if not isinstance(conditions, dict):
            raise ValueError("curator run conditions must be an object")
        if not isinstance(actions, list):
            raise ValueError("curator run actions must be a list")
        if vote_ok is not None and type(vote_ok) is not bool:
            raise ValueError("curator run vote_ok must be true, false, or null")
        at = ts or datetime.now(tz=timezone.utc)
        if at.tzinfo is None:
            raise ValueError("curator run timestamp must be timezone-aware")
        try:
            counts = {
                "reviewed": max(0, int(reviewed)),
                "created": max(0, int(created)),
                "voted": max(0, int(voted)),
                "archive_recommendations": max(0, int(archive_recommendations)),
            }
        except (TypeError, ValueError) as e:
            raise ValueError("curator run counts must be integers") from e

        clean_actions: list[dict[str, Any]] = []
        for raw_action in actions:
            if not isinstance(raw_action, dict):
                raise ValueError("each curator action must be an object")
            raw_rank = raw_action.get("rank")
            try:
                rank = int(raw_rank) if raw_rank is not None else None
            except (TypeError, ValueError) as e:
                raise ValueError("curator action rank must be an integer or null") from e
            evidence_ids = raw_action.get("evidence_ids", [])
            if not isinstance(evidence_ids, list):
                raise ValueError("curator action evidence_ids must be a list")
            action_name = " ".join(str(raw_action.get("action", "reviewed")).split())[:80]
            clean_actions.append(
                {
                    "lesson_id": str(raw_action.get("lesson_id", "")) or None,
                    "rank": rank,
                    "action": action_name or "reviewed",
                    "before_status": str(raw_action.get("before_status", "")) or None,
                    "after_status": str(raw_action.get("after_status", "")) or None,
                    "reason": " ".join(str(raw_action.get("reason", "")).split())[:500],
                    "evidence_ids": [str(value) for value in evidence_ids[:12]],
                }
            )

        run_id = _short("cr")
        clean_reason = " ".join(reason.split())[:1_000]
        payload = {
            "id": run_id,
            "status": status,
            **counts,
            "vote_ok": vote_ok,
            "reason": clean_reason,
        }
        # SQLite otherwise autocommits each statement. Keep the run, its
        # action rows, journal entry, and Markdown audit together so the
        # dashboard cannot show a successful review with a partial audit.
        conn = self.conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO curator_runs
                   (id, ts, status, conditions, reviewed, created, voted, vote_ok,
                    archive_recommendations, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    at.timestamp(),
                    status,
                    json.dumps(conditions, default=str, sort_keys=True),
                    counts["reviewed"],
                    counts["created"],
                    counts["voted"],
                    None if vote_ok is None else int(vote_ok),
                    counts["archive_recommendations"],
                    clean_reason,
                ),
            )
            for clean in clean_actions:
                conn.execute(
                    """INSERT INTO curator_actions
                       (run_id, lesson_id, rank, action, before_status, after_status, reason, evidence_ids)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        clean["lesson_id"],
                        clean["rank"],
                        clean["action"],
                        clean["before_status"],
                        clean["after_status"],
                        clean["reason"],
                        json.dumps(clean["evidence_ids"]),
                    ),
                )
            self.journal("lesson_curation", payload, actor="learning_curator")
            self._write_curator_report(run_id, at, payload, conditions, clean_actions)
            conn.execute("COMMIT")
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        return run_id

    def _write_curator_report(
        self,
        run_id: str,
        at: datetime,
        payload: dict[str, Any],
        conditions: dict[str, Any],
        actions: list[dict[str, Any]],
    ) -> None:
        """Render a human-readable immutable companion to ``curator_runs``."""
        lines = [
            "---",
            f"id: {run_id}",
            f"status: {payload['status']}",
            f"reviewed: {payload['reviewed']}",
            f"created: {payload['created']}",
            f"voted: {payload['voted']}",
            f"vote_ok: {payload['vote_ok']}",
            f"archive_recommendations: {payload['archive_recommendations']}",
            f"timestamp: {at.astimezone(timezone.utc).isoformat()}",
            "---",
            "",
            "# Learning Curator review",
        ]
        if payload["reason"]:
            lines += ["", "## Run note", "", str(payload["reason"])]
        lines += [
            "",
            "## Conditions",
            "",
            "```json",
            json.dumps(conditions, indent=2, sort_keys=True, default=str),
            "```",
            "",
            "## Actions",
        ]
        if not actions:
            lines += ["", "No lesson actions were recorded."]
        for action in actions:
            target = action["lesson_id"] or "run"
            state = " → ".join(
                value for value in (action["before_status"], action["after_status"]) if value
            )
            suffix = f" ({state})" if state else ""
            lines += ["", f"- **{action['action']}** [[{target}]]{suffix}: {action['reason']}"]
            if action["evidence_ids"]:
                lines.append(
                    "  Evidence: " + ", ".join(f"[[{eid}]]" for eid in action["evidence_ids"])
                )
        _atomic_write(self.reviews_dir / f"{run_id}.md", "\n".join(lines) + "\n")

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

    def _session_pending_prediction_ids(self, *, asof: datetime) -> set[str]:
        """Exact daily-bar waits that remain valid at ``asof``.

        The daily journal is intentionally the source of truth for which
        rows were waiting: the store has no price-cache path and must not
        infer that every weekend expiry is healthy.  Its answer expires when
        a later NYSE session has settled, so a stale journal cannot disguise
        a grader/cache failure as a normal pending row.
        """
        try:
            row = self.conn.execute(
                """SELECT ts, payload FROM journal WHERE kind = 'daily'
                   ORDER BY ts DESC LIMIT 1"""
            ).fetchone()
            if row is None:
                return set()
            journal_ts = float(row["ts"])
            # A future-dated journal is not evidence about the current
            # scorecard state. Otherwise calendar settlement, rather than a
            # fixed 48-hour TTL, governs the exemption: a holiday weekend
            # can legitimately last longer than two days.
            if journal_ts > asof.timestamp():
                return set()
            payload = json.loads(row["payload"])
            if not isinstance(payload, dict):
                return set()
            raw_ids = payload.get("awaiting_next_daily_bar_prediction_ids", [])
            if not isinstance(raw_ids, list):
                return set()
            from trading.runtime.portfolio_stats import nyse_session_settled_since

            journal_at = datetime.fromtimestamp(journal_ts, tz=timezone.utc)
            if nyse_session_settled_since(journal_at, asof):
                return set()
            return {str(prediction_id) for prediction_id in raw_ids}
        except Exception:
            # A malformed journal or unavailable local calendar is never
            # permission to suppress an overdue count in long-horizon
            # memory. This helper is deliberately fail-closed.
            return set()

    def scorecard_backfill_targets(
        self, *, asof: datetime | None = None, max_age_days: int = 730
    ) -> list[dict[str, Any]]:
        """Distinct matured subjects whose missing prices block scoring.

        The cache refresher uses this to repair the scorecard deliberately,
        rather than only fetching today's configured trading universe and
        hoping it happens to contain yesterday's prediction subjects.
        """
        cutoff = (asof or datetime.now(tz=timezone.utc)).timestamp()
        earliest = cutoff - max_age_days * 86400.0
        rows = self.conn.execute(
            """SELECT subject, MIN(ts) AS earliest_ts, COUNT(*) AS n
               FROM predictions
               WHERE graded_ts IS NULL AND due_ts <= ? AND ts >= ?
               GROUP BY subject ORDER BY earliest_ts""",
            (cutoff, earliest),
        ).fetchall()
        return [dict(r) for r in rows]

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
            {
                "id": prediction_id,
                "agent": row["agent"],
                "subject": row["subject"],
                "direction": row["direction"],
                "horizon_days": row["horizon_days"],
                "confidence": row["confidence"],
                "outcome": outcome,
                "realized_move": realized_move,
                "brier": brier,
                "statement": str(row["statement"])[:300],
            },
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

    def historian_dossier(
        self,
        *,
        focus_statements: list[str],
        since_days: int = 365,
        recent_limit: int = 12,
        retired_limit: int = 8,
        asof: datetime | None = None,
    ) -> dict[str, Any]:
        """Compact, auditable long-horizon memory for a Historian pass.

        This is deliberately SQL and lexical matching rather than a hidden
        embedding index: every aggregate and every "similar prior lesson"
        can be reproduced from the permanent store. The bounded detail lets
        the prompt retain a year of measured history without pretending that
        hundreds of raw journal rows are useful context.
        """
        now = asof or datetime.now(tz=timezone.utc)
        if now.tzinfo is None:
            raise ValueError("historian dossier asof must be timezone-aware")
        now_ts = now.timestamp()
        cutoff = now_ts - since_days * 86400.0

        scorecard = self.conn.execute(
            """SELECT COUNT(*) AS graded,
                      AVG(CASE WHEN outcome = 'hit' THEN 1.0 ELSE 0.0 END) AS hit_rate,
                      AVG(brier) AS brier,
                      AVG(realized_move) AS mean_move
               FROM predictions WHERE graded_ts IS NOT NULL AND graded_ts >= ?""",
            (cutoff,),
        ).fetchone()
        mature_ungraded = self.conn.execute(
            "SELECT id FROM predictions WHERE graded_ts IS NULL AND due_ts <= ?", (now_ts,)
        ).fetchall()
        pending_ids = self._session_pending_prediction_ids(asof=now)
        session_pending = sum(str(row["id"]) in pending_ids for row in mature_ungraded)
        overdue_ungraded = len(mature_ungraded) - session_pending
        by_agent = self.conn.execute(
            """SELECT agent, COUNT(*) AS n,
                      AVG(CASE WHEN outcome = 'hit' THEN 1.0 ELSE 0.0 END) AS hit_rate,
                      AVG(brier) AS brier
               FROM predictions
               WHERE graded_ts IS NOT NULL AND graded_ts >= ?
               GROUP BY agent ORDER BY n DESC, agent LIMIT 12""",
            (cutoff,),
        ).fetchall()
        by_subject = self.conn.execute(
            """SELECT subject, horizon_days, COUNT(*) AS n,
                      AVG(CASE WHEN outcome = 'hit' THEN 1.0 ELSE 0.0 END) AS hit_rate,
                      AVG(realized_move) AS mean_move
               FROM predictions
               WHERE graded_ts IS NOT NULL AND graded_ts >= ?
               GROUP BY subject, horizon_days
               ORDER BY n DESC, subject LIMIT 16""",
            (cutoff,),
        ).fetchall()
        predictions = self.conn.execute(
            """SELECT id, agent, subject, direction, horizon_days, confidence,
                      statement, outcome, realized_move, brier, graded_ts
               FROM predictions
               WHERE graded_ts IS NOT NULL AND graded_ts >= ?
               ORDER BY graded_ts DESC LIMIT ?""",
            (cutoff, recent_limit),
        ).fetchall()

        episode_summary = self.conn.execute(
            """SELECT COUNT(*) AS n, AVG(pnl_pct) AS mean_pnl,
                      AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
               FROM episodes WHERE ts_close >= ?""",
            (cutoff,),
        ).fetchone()
        episode_rows = self.conn.execute(
            """SELECT id, symbol, side, ts_open, ts_close, pnl_pct,
                      entry_pctile_52w, context, tags
               FROM episodes WHERE ts_close >= ?
               ORDER BY ts_close DESC LIMIT ?""",
            (cutoff, recent_limit),
        ).fetchall()

        focus = (
            set().union(*(_keywords(s) for s in focus_statements)) if focus_statements else set()
        )
        retired_rows = self.conn.execute(
            """SELECT * FROM lessons WHERE status = 'retired'
               ORDER BY retired_ts DESC LIMIT 100"""
        ).fetchall()

        def retired_score(row: sqlite3.Row) -> tuple[int, float]:
            terms = _keywords(f"{row['statement']} {row['tags']} {row['retired_why'] or ''}")
            # A lexical overlap is an explicit retrieval explanation. The
            # timestamp tie-break makes the result stable and reviewable.
            return (len(focus & terms), float(row["retired_ts"] or 0.0))

        matching = [r for r in retired_rows if retired_score(r)[0] > 0]
        selected = sorted(matching or retired_rows, key=retired_score, reverse=True)[:retired_limit]

        def episode_card(row: sqlite3.Row) -> dict[str, Any]:
            try:
                context = json.loads(row["context"])
            except (TypeError, ValueError):
                context = {}
            return {
                "id": row["id"],
                "symbol": row["symbol"],
                "side": row["side"],
                "opened": datetime.fromtimestamp(row["ts_open"], tz=timezone.utc)
                .date()
                .isoformat(),
                "closed": datetime.fromtimestamp(row["ts_close"], tz=timezone.utc)
                .date()
                .isoformat(),
                "pnl_pct": row["pnl_pct"],
                "entry_pctile_52w": row["entry_pctile_52w"],
                "context": context,
                "tags": row["tags"],
            }

        def prediction_card(row: sqlite3.Row) -> dict[str, Any]:
            card = dict(row)
            card["graded"] = datetime.fromtimestamp(card.pop("graded_ts"), tz=timezone.utc)
            card["graded"] = card["graded"].date().isoformat()
            card["statement"] = str(card["statement"])[:300]
            return card

        return {
            "coverage": {
                "since_days": since_days,
                "focus_terms": sorted(focus)[:24],
                "note": (
                    "Aggregates are measured history. A small sample is not proof; "
                    "do not infer a durable rule from a thin slice."
                ),
            },
            "scorecard": {
                "graded": int(scorecard["graded"] or 0),
                "hit_rate": scorecard["hit_rate"],
                "brier": scorecard["brier"],
                "mean_move": scorecard["mean_move"],
                "overdue_ungraded": overdue_ungraded,
                "session_pending_ungraded": session_pending,
                "by_agent": [dict(r) for r in by_agent],
                "by_subject": [dict(r) for r in by_subject],
                "recent_outcomes": [prediction_card(r) for r in predictions],
            },
            "episodes": {
                "n": int(episode_summary["n"] or 0),
                "mean_pnl": episode_summary["mean_pnl"],
                "win_rate": episode_summary["win_rate"],
                "recent": [episode_card(r) for r in episode_rows],
            },
            "related_retired_lessons": [
                {
                    "id": r["id"],
                    "statement": r["statement"],
                    "tags": r["tags"],
                    "support": r["support"],
                    "contradict": r["contradict"],
                    "retired_why": r["retired_why"],
                    "matched_terms": sorted(
                        focus & _keywords(f"{r['statement']} {r['tags']} {r['retired_why'] or ''}")
                    )[:12],
                    "evidence": self.lesson_evidence(r["id"], limit=4),
                }
                for r in selected
            ],
        }

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
        snapshot: dict[str, Any] | None = None,
        px_at: float | None = None,
        pctile_52w: float | None = None,
        ts: float | None = None,
    ) -> str:
        """Record one considered-and-decided name. Never raises on a dup."""
        sid = _short("sh")
        self.conn.execute(
            """INSERT INTO shadow
               (id, ts, symbol, origin, disposition, rank, score, why,
                conditions, snapshot, px_at, pctile_52w)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                json.dumps(snapshot or {}, default=str),
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

    def curator_summary(self, *, limit: int = 8) -> dict[str, Any]:
        """Read-only Learning Curator health, queue, and lifecycle summary.

        This is intentionally derived from the permanent store instead of a
        dashboard cache, so a restart cannot make a failed or missed review
        appear successful.  ``retired`` is presented as ``archived`` to the
        operator; the underlying append-only lifecycle name is retained for
        compatibility with existing lesson cards and state.
        """
        review = self.lesson_review({})
        raw_counts = dict(review["status_counts"])
        status_counts = {
            "candidate": int(raw_counts.get("candidate", 0)),
            "established": int(raw_counts.get("established", 0)),
            "challenged": int(raw_counts.get("challenged", 0)),
            "archived": int(raw_counts.get("retired", 0)),
        }
        row = self.conn.execute("SELECT * FROM curator_runs ORDER BY ts DESC LIMIT 1").fetchone()
        last_run: dict[str, Any] | None = None
        recent_review_actions: list[dict[str, Any]] = []
        if row is not None:
            try:
                conditions = json.loads(row["conditions"])
            except (TypeError, ValueError, json.JSONDecodeError):
                conditions = {}
            age_seconds = max(0.0, _now() - float(row["ts"]))
            fresh = age_seconds <= _CURATOR_STALE_AFTER.total_seconds()
            if not fresh:
                health = "stale"
            elif row["status"] == "completed":
                health = "ok"
            else:
                health = str(row["status"])
            last_run = {
                "id": row["id"],
                "ts": datetime.fromtimestamp(row["ts"], tz=timezone.utc).isoformat(),
                "status": row["status"],
                "ok": health == "ok",
                "health": health,
                "fresh": fresh,
                "age_hours": round(age_seconds / 3600.0, 1),
                "conditions": conditions if isinstance(conditions, dict) else {},
                "reviewed": int(row["reviewed"]),
                "created": int(row["created"]),
                "voted": int(row["voted"]),
                "vote_ok": None if row["vote_ok"] is None else bool(row["vote_ok"]),
                "archive_recommendations": int(row["archive_recommendations"]),
                "reason": row["reason"],
            }
            actions = self.conn.execute(
                """SELECT lesson_id, rank, action, before_status, after_status, reason, evidence_ids
                   FROM curator_actions WHERE run_id = ? ORDER BY id LIMIT ?""",
                (row["id"], limit),
            ).fetchall()
            for action in actions:
                try:
                    evidence_ids = json.loads(action["evidence_ids"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    evidence_ids = []
                recent_review_actions.append(
                    {
                        "lesson_id": action["lesson_id"],
                        "rank": action["rank"],
                        "action": action["action"],
                        "before_status": action["before_status"],
                        "after_status": action["after_status"],
                        "reason": action["reason"],
                        "evidence_ids": evidence_ids if isinstance(evidence_ids, list) else [],
                    }
                )

        lifecycle_kinds = {
            "lesson_created": "created",
            "lesson_status_changed": "status changed",
            "lesson_established": "promoted",
            "lesson_challenged": "challenged",
            "lesson_retired": "archived",
            "lesson_restored": "restored",
        }
        lifecycle: list[dict[str, Any]] = []
        for event in self._lesson_lifecycle_events(limit=max(limit * 2, 20)):
            action = lifecycle_kinds.get(event["kind"])
            payload = event["payload"]
            if action is None:
                continue
            lesson_id = str(payload.get("id", ""))
            if not lesson_id.startswith("ls-"):
                continue
            lesson = self.conn.execute(
                "SELECT statement, support, contradict FROM lessons WHERE id = ?", (lesson_id,)
            ).fetchone()
            reason = str(payload.get("why", "")).strip()
            if not reason and event["kind"] == "lesson_status_changed":
                before = str(payload.get("from_status", "")).strip()
                after = str(payload.get("to_status", payload.get("status", ""))).strip()
                reason = " → ".join(value for value in (before, after) if value)
            lifecycle.append(
                {
                    "ts": event["ts"].isoformat(),
                    "action": action,
                    "actor": event["actor"],
                    "lesson_id": lesson_id,
                    "statement": "" if lesson is None else lesson["statement"],
                    "support": 0 if lesson is None else int(lesson["support"]),
                    "contradict": 0 if lesson is None else int(lesson["contradict"]),
                    "reason": reason,
                }
            )
            if len(lifecycle) >= limit:
                break

        def compact(card: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": card["id"],
                "statement": card["statement"],
                "outcome_support": int(card["outcome_support"]),
                "outcome_contradict": int(card["outcome_contradict"]),
                "outcome_observations": int(card["outcome_observations"]),
                "relevance": float(card["relevance"]),
            }

        return {
            "last_run": last_run,
            "status_counts": status_counts,
            "queue": {"candidate_unreviewed": int(review["candidate_unreviewed"])},
            "top_lessons": [compact(card) for card in review["ranked_established"][:limit]],
            "archive_recommendations": [
                {
                    "id": item["lesson_id"],
                    "statement": item["statement"],
                    "outcome_support": item["outcome_support"],
                    "outcome_contradict": item["outcome_contradict"],
                    "reason": item["reason"],
                    "requires_operator_approval": item["requires_operator_approval"],
                }
                for item in review["archive_recommendations"][:limit]
            ],
            "recent_review_actions": recent_review_actions,
            "changes": lifecycle,
        }

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
