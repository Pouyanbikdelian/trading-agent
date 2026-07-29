"""Conversation memory for the copilot, and the objections inside it.

Until now every message stood alone. The operator had to pack all context
into one sentence — "is the MU thesis still valid?" rather than "and MU?"
— which is not how anyone talks, least of all on a phone. This module
keeps a short rolling window of turns so follow-ups resolve.

Three deliberate limits
-----------------------
*Short.* Only the last few turns go into the prompt. This is a cheap
model on a small context, and the evidence JSON is the part that should
get the room. Older turns are dropped, not summarised — a summary of a
conversation is another thing that can be wrong.

*Expiring.* A thread goes stale after ``IDLE_EXPIRY_S``. Asking "and
XLE?" at 9am should not inherit last night's argument about semis; the
operator has slept and moved on, and silently answering in the wrong
context is worse than asking them to restate.

*Not evidence.* Turns are conversation, never facts. The charter already
treats transcript text as quoted data; thread turns are weaker still —
they are what the operator and the copilot *said*, including anything the
copilot got wrong. Nothing here may be cited as a source.

Objections
----------
When the operator pushes back — "why not XLE instead?", "that's too much
semis" — that is the most valuable thing in the chat and the thing most
certainly lost to scrollback. Detection is deterministic marker matching,
not an LLM call: cheap, predictable, and a false negative merely means an
exchange is stored as ordinary chat. Flagged objections are journaled so
``agents.context`` can put them in front of the committee, and so
``/why SYM`` can later cite the day the operator disagreed.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

_LOG = logger.bind(component="copilot.thread")

THREAD_FILE = "copilot_thread.json"
MAX_TURNS = 6  # 3 exchanges; enough for "and XLE?" to resolve
MAX_TURN_CHARS = 600
IDLE_EXPIRY_S = 45 * 60.0

# Pushback markers. Tuned for precision over recall: a missed objection is
# stored as ordinary chat and costs nothing, whereas a false positive puts
# words in the operator's mouth in front of the committee.
_OBJECTION_PATTERNS = (
    r"\bwhy not\b",
    r"\bwhy did(?:n't| not)\b",
    r"\binstead of\b",
    r"\brather than\b",
    r"\bi disagree\b",
    r"\bdisagree\b",
    r"\bi'?d rather\b",
    r"\bi think (?:we|you)\b",
    r"\bshould(?:n't| not)? (?:we|you)\b",
    r"\bwe should\b",
    r"\byou should\b",
    r"\bthat'?s (?:too|wrong|not right)\b",
    r"\btoo much\b",
    r"\btoo concentrated\b",
    r"\bmakes no sense\b",
    r"\bdoesn'?t make sense\b",
    r"\bnot convinced\b",
    r"\bpush ?back\b",
)
_OBJECTION_RE = re.compile("|".join(_OBJECTION_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class Turn:
    role: str  # "operator" | "copilot"
    text: str
    ts: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _age_seconds(ts: str) -> float:
    try:
        return (datetime.now(tz=timezone.utc) - datetime.fromisoformat(str(ts))).total_seconds()
    except ValueError:
        return float("inf")


class Thread:
    """The recent conversation, persisted so a bot restart doesn't reset it."""

    def __init__(self, state_dir: Path) -> None:
        self.path = Path(state_dir) / THREAD_FILE

    def load(self) -> tuple[list[Turn], str | None]:
        """Return ``(turns, last_symbol)``, both empty if stale or absent.

        Expiry is enforced on read rather than by a sweeper: the only
        thing that cares is the next question, and it is the only thing
        that knows what time it is now.
        """
        try:
            payload = json.loads(self.path.read_text())
        except Exception:
            return [], None
        turns_raw = payload.get("turns") or []
        if not turns_raw:
            return [], None
        if _age_seconds(str(turns_raw[-1].get("ts", ""))) > IDLE_EXPIRY_S:
            return [], None
        turns = [
            Turn(str(t.get("role", "")), str(t.get("text", "")), str(t.get("ts", "")))
            for t in turns_raw[-MAX_TURNS:]
        ]
        sym = payload.get("last_symbol")
        return turns, (str(sym).upper() if sym else None)

    def append(self, role: str, text: str, *, symbol: str | None = None) -> None:
        """Add a turn, trimming the window. Never raises."""
        try:
            turns, prior_symbol = self.load()
            turns.append(Turn(role, str(text)[:MAX_TURN_CHARS], _now_iso()))
            _atomic_write(
                self.path,
                {
                    "turns": [t.to_dict() for t in turns[-MAX_TURNS:]],
                    # Carry the last known symbol forward so a bare
                    # follow-up ("and the thesis?") still has a subject.
                    "last_symbol": (symbol or prior_symbol or None),
                },
            )
        except Exception:
            _LOG.exception("thread append failed")

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            _LOG.exception("thread clear failed")

    def as_evidence(self) -> list[dict[str, str]]:
        """The window, shaped for the prompt. Empty when stale."""
        turns, _sym = self.load()
        return [t.to_dict() for t in turns]


def looks_like_objection(text: str) -> bool:
    """True when the operator appears to be pushing back, not just asking.

    Deliberately conservative — see the module docstring on why precision
    beats recall here."""
    return bool(_OBJECTION_RE.search(text or ""))


def extract_symbols(text: str, known: set[str] | None = None) -> list[str]:
    """Uppercase tokens that look like tickers, filtered by ``known``.

    Without a known-symbol set this over-matches ("I", "IT", "OK"), so an
    empty ``known`` yields an empty list rather than noise."""
    if not known:
        return []
    found = [w for w in re.findall(r"\b[A-Z]{1,5}\b", text or "") if w in known]
    return sorted(dict.fromkeys(found))


def record_exchange(
    state_dir: Path,
    question: str,
    reply: str,
    *,
    symbol: str | None = None,
    known_symbols: set[str] | None = None,
) -> str | None:
    """Persist one exchange to the thread and the memory journal.

    Returns the journal kind written (``"operator_objection"`` or
    ``"chat"``), or None if the journal was unavailable — the thread is
    still updated in that case, because conversation continuity must not
    depend on the memory store being healthy.
    """
    thread = Thread(state_dir)
    thread.append("operator", question, symbol=symbol)
    thread.append("copilot", reply, symbol=symbol)

    is_objection = looks_like_objection(question)
    kind = "operator_objection" if is_objection else "chat"
    try:
        from trading.memory.store import MemoryStore

        mem = MemoryStore(Path(state_dir) / "memory")
        try:
            symbols = extract_symbols(question, known_symbols)
            if symbol and symbol not in symbols:
                symbols.append(symbol)
            mem.journal(
                kind,
                {
                    "question": question[:1000],
                    "reply": reply[:2000],
                    "symbols": symbols,
                    # Objections are the operator's standing view until
                    # withdrawn; the committee reads them as such.
                    "is_objection": is_objection,
                },
                actor="operator",
            )
        finally:
            mem.close()
        return kind
    except Exception:
        _LOG.exception("journalling the exchange failed")
        return None


def recent_objections(state_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    """Recent operator pushback, newest first, for the committee context."""
    try:
        from trading.memory.store import MemoryStore

        mem = MemoryStore(Path(state_dir) / "memory")
        try:
            # journal_tail is ORDER BY id DESC — already newest first.
            rows = mem.journal_tail(limit, kind="operator_objection")
        finally:
            mem.close()
    except Exception:
        _LOG.exception("reading objections failed")
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        out.append(
            {
                "ts": str(row.get("ts")),
                "said": str(payload.get("question", ""))[:400],
                "symbols": payload.get("symbols") or [],
            }
        )
    return out
