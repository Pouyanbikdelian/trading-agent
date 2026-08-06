"""Standing instructions the operator leaves for the next committee/PM run.

The point is that Yan should not have to open a laptop to say "I have
conviction on GS, look at it next round". He types it into Telegram, the
next run sees it, and the agents must respond to it.

Strength is read from tone
--------------------------
"I want GS, buy it" and "GS might be worth a look" are different
instructions, and flattening them into one level would either ignore the
first or over-weight the second. Three levels, graded by phrasing:

* ``strong``  — "I want X", "buy X", "make sure we hold X". Treated as a
  strong prior: allocate unless there is a concrete reason not to, and if
  declining, say why explicitly.
* ``medium``  — "high conviction on X", "highly consider X". Weighed
  seriously, may be declined with a brief reason.
* ``soft``    — "consider X", "might be worth a look". A suggestion the
  agents may drop silently.

Grading is deterministic marker matching, not an LLM call — the operator
must be able to predict how a phrase will be read, and a model that
silently re-grades tone between runs is not predictable. Every stored
mandate is echoed back with the strength it was given, so a misreading is
visible and correctable *before* the run rather than discovered after.

What a mandate is NOT
---------------------
It is not an order, and it is not a way around the risk limits. Even a
``strong`` mandate reaches the PM as *context for its decision*; the
resulting weights still pass through ``agents.pm._clamp_weights``, so the
per-name caps (10% single stock, 25% ETF), the 50% cluster cap and the
gross cap all still bind. A message typed one-handed on a phone must not
be able to concentrate the book past the limits — the caps exist for the
days when conviction is highest, which are exactly the days a mandate
gets written.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

_LOG = logger.bind(component="copilot.mandates")

MANDATES_FILE = "operator_mandates.json"
DEFAULT_TTL_DAYS = 14
MAX_ACTIVE = 8
MAX_TEXT_CHARS = 500

STRONG = "strong"
MEDIUM = "medium"
SOFT = "soft"

# Ordered strongest-first: the first pattern that matches wins, so an
# unambiguous "buy X" is never demoted by a later hedge word.
_STRENGTH_PATTERNS: tuple[tuple[str, str], ...] = (
    (STRONG, r"\bi want\b"),
    (STRONG, r"\bmake sure\b"),
    (STRONG, r"\bmust (?:include|hold|buy|have)\b"),
    (STRONG, r"\bdefinitely\b"),
    (STRONG, r"\bbuy\b"),
    (STRONG, r"\bget me\b"),
    (STRONG, r"\bi'?m convinced\b"),
    (MEDIUM, r"\bhigh(?:ly)? conviction\b"),
    (MEDIUM, r"\bhighly consider\b"),
    (MEDIUM, r"\bstrongly consider\b"),
    (MEDIUM, r"\bi'?d like\b"),
    (MEDIUM, r"\bshould (?:include|hold|own|add)\b"),
    (MEDIUM, r"\bworth (?:a )?(?:serious|real) look\b"),
    (SOFT, r"\bconsider\b"),
    (SOFT, r"\bi think\b"),
    (SOFT, r"\bmaybe\b"),
    (SOFT, r"\bcould be\b"),
    (SOFT, r"\bmight be\b"),
    (SOFT, r"\bworth a look\b"),
    (SOFT, r"\bkeep an eye\b"),
)

# A mandate is forward-looking: it is about what the NEXT run should do.
# Without this the module would capture every opinion as an instruction.
_FORWARD_MARKERS = (
    r"\bnext (?:round|run|cycle|committee|rebalance|time)\b",
    r"\bgoing forward\b",
    r"\bfrom now on\b",
    r"\bin future\b",
    r"\bnext week\b",
)
_FORWARD_RE = re.compile("|".join(_FORWARD_MARKERS), re.IGNORECASE)

# Instruction-shaped verbs. Combined with a strength marker these make a
# mandate even without an explicit "next round".
_INSTRUCTION_RE = re.compile(
    r"\b(include|add|allocate|buy|hold|own|overweight|underweight|avoid|drop|"
    r"exclude|look at|keep an eye|consider|"
    # Added 2026-08-06. The list above was the entire vocabulary a mandate
    # could be written in, and it silently swallowed the largest strategic
    # instruction the operator has ever sent — a full thematic allocation
    # ("I want you to ... dedicate at least 40% ... to physical AI and
    # quantum ... rest of capital 30% to energy infrastructure"). It
    # graded as `strong` and was still dropped, because he wrote
    # "dedicate" and "focus" and neither was a recognised verb.
    #
    # The lesson is not "add two words": a detector whose recall depends
    # on the operator guessing its vocabulary will keep failing silently.
    # These cover how allocation instructions are actually phrased.
    r"dedicate|devote|commit|focus|concentrate|"
    r"put|place|deploy|invest|weight|tilt|lean|skew|bias|"
    r"target|aim for|cap|limit|reserve|set aside|keep|maintain|"
    r"trim|cut|reduce|increase|raise|lower|rotate|shift|move|"
    r"prioriti[sz]e|emphasi[sz]e|favou?r|prefer|stick with|stay in)\b",
    re.IGNORECASE,
)

# A bare percentage aimed at the book is an instruction even without a
# verb: "rest of capital 30% to energy infrastructure" has no verb at
# all, and "20-30% cash if markets are falling" only has a preposition.
# Requiring a verb there is requiring the operator to write like the
# regex. Paired with a strength marker as usual, so idle talk about
# percentages ("SPY is up 3% today") is not captured.
_ALLOCATION_RE = re.compile(r"\d{1,3}\s*(?:-\s*\d{1,3}\s*)?%|\bpercent\b", re.IGNORECASE)

# Questions are not mandates, however opinionated. "why not GS?" is an
# objection (see thread.py); "should we buy GS?" is a question.
_QUESTION_RE = re.compile(r"^\s*(what|why|who|when|where|how|is|are|do|does|did|can|could)\b", re.I)

# Capability questions — "are you able to consider my conviction on a
# stock?" — are the worst false positive this module can make. They
# contain every marker a mandate has (an instruction verb, a strength
# word) while asking whether the desk *could* do a thing, not telling it
# to. Capturing one makes the bot appear to accept an instruction the
# operator never gave, and the operator only finds out when a run acts on
# it. Matched anywhere in the clause, not just at the start.
_CAPABILITY_RE = re.compile(
    r"\b(?:are|can|could|will|would|do|does|did)\s+(?:you|we|it|they|i)\b"
    r"|\b(?:you|we)\s+able\s+to\b"
    r"|\bable\s+to\b"
    r"|\bis\s+it\s+possible\b"
    r"|\bcapable\s+of\b"
    r"|\bwhat\s+(?:if|happens)\b",
    re.IGNORECASE,
)

# Hypothetical framing. "If I tell you I want GS..." is the operator
# describing a scenario, not opening one.
_HYPOTHETICAL_RE = re.compile(r"^\s*(?:if|suppose|imagine|say)\b", re.IGNORECASE)

# Sentence/clause split. Telegram messages are often several thoughts in
# one bubble ("I want GS next round. Can you do that?") and grading the
# whole blob loses the distinction between the instruction and the
# question wrapped around it.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _SENTENCE_SPLIT_RE.split(text or "") if c.strip()]


def _is_question(clause: str) -> bool:
    """True when this clause asks rather than instructs.

    A trailing '?' is decisive. The operator can always drop it, and a
    rule they can state in one sentence beats a cleverer one they
    cannot predict."""
    c = clause.strip()
    if c.endswith("?"):
        return True
    if _CAPABILITY_RE.search(c):
        return True
    return bool(_QUESTION_RE.match(c)) and not _FORWARD_RE.search(c)


def _clause_is_mandate(clause: str) -> bool:
    """The old whole-message test, applied to one clause."""
    t = clause.strip()
    if len(t) < 6:
        return False
    if _is_question(t) or _HYPOTHETICAL_RE.match(t):
        return False
    if _FORWARD_RE.search(t):
        return True
    # An instruction verb OR a percentage aimed at the book. Either alone
    # is ambiguous ("add cash", "30%"), so both still require a strength
    # phrase — that is what separates an instruction from commentary.
    if not (_INSTRUCTION_RE.search(t) or _ALLOCATION_RE.search(t)):
        return False
    return any(re.search(p, t.lower()) for _s, p in _STRENGTH_PATTERNS)


def mandate_span(text: str) -> str | None:
    """The clause that is actually an instruction, or None.

    Returning the span rather than a bool means the stored mandate — and
    the echo the operator reads back — quotes the instruction itself
    instead of the whole rambling bubble it arrived in."""
    for clause in _clauses(text):
        if _clause_is_mandate(clause):
            return clause
    return None


@dataclass(frozen=True)
class Mandate:
    id: str
    text: str
    strength: str
    created_at: str
    expires_at: str
    symbols: list[str] = field(default_factory=list)
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_expired(self, now: datetime | None = None) -> bool:
        try:
            return (now or datetime.now(tz=timezone.utc)) >= datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False


def grade_strength(text: str) -> str:
    """Read instruction strength from phrasing. Defaults to ``soft``.

    Defaulting soft is deliberate: an unrecognised phrasing should
    under-weight rather than over-weight. The operator can always restate
    more firmly, but cannot un-buy a position taken on a misread."""
    t = (text or "").lower()
    for strength, pattern in _STRENGTH_PATTERNS:
        if re.search(pattern, t):
            return strength
    return SOFT


def looks_like_mandate(text: str) -> bool:
    """True when the message contains an instruction for a future run.

    Requires a forward-looking marker OR (an instruction verb AND a
    strength marker) in some non-interrogative clause."""
    return mandate_span(text) is not None


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, default=str, indent=1)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class MandateStore:
    def __init__(self, state_dir: Path) -> None:
        self.path = Path(state_dir) / MANDATES_FILE

    def _read_all(self) -> list[Mandate]:
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            return []
        out: list[Mandate] = []
        for r in raw if isinstance(raw, list) else []:
            try:
                out.append(
                    Mandate(
                        id=str(r["id"]),
                        text=str(r["text"]),
                        strength=str(r.get("strength", SOFT)),
                        created_at=str(r["created_at"]),
                        expires_at=str(r["expires_at"]),
                        symbols=list(r.get("symbols") or []),
                        status=str(r.get("status", "active")),
                    )
                )
            except Exception:
                continue
        return out

    def active(self, now: datetime | None = None) -> list[Mandate]:
        """Live mandates, strongest first so a truncated prompt keeps them."""
        rank = {STRONG: 0, MEDIUM: 1, SOFT: 2}
        live = [m for m in self._read_all() if m.status == "active" and not m.is_expired(now)]
        live.sort(key=lambda m: (rank.get(m.strength, 3), m.created_at))
        return live[:MAX_ACTIVE]

    def add(
        self,
        text: str,
        *,
        symbols: list[str] | None = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
        strength: str | None = None,
    ) -> Mandate:
        now = datetime.now(tz=timezone.utc)
        m = Mandate(
            id=f"M{int(now.timestamp() * 1000) % 10_000_000}",
            text=str(text)[:MAX_TEXT_CHARS],
            strength=strength or grade_strength(text),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=ttl_days)).isoformat(),
            symbols=sorted(symbols or []),
        )
        # Drop expired rows on write — the file is small and this keeps it
        # from growing without a sweeper job.
        keep = [x for x in self._read_all() if not x.is_expired(now)]
        _atomic_write(self.path, [x.to_dict() for x in [*keep, m]])
        self._journal(m)
        return m

    def set_strength(self, mandate_id: str, strength: str) -> Mandate | None:
        """Re-grade a mandate whose tone was read wrongly."""
        rows = self._read_all()
        out: list[Mandate] = []
        found: Mandate | None = None
        for m in rows:
            if m.id == mandate_id and m.status == "active":
                found = Mandate(
                    id=m.id,
                    text=m.text,
                    strength=strength,
                    created_at=m.created_at,
                    expires_at=m.expires_at,
                    symbols=m.symbols,
                    status=m.status,
                )
                out.append(found)
            else:
                out.append(m)
        if found is not None:
            _atomic_write(self.path, [x.to_dict() for x in out])
        return found

    def cancel(self, mandate_id: str) -> bool:
        rows = self._read_all()
        hit = False
        out: list[Mandate] = []
        for m in rows:
            if m.id == mandate_id and m.status == "active":
                hit = True
                out.append(
                    Mandate(
                        id=m.id,
                        text=m.text,
                        strength=m.strength,
                        created_at=m.created_at,
                        expires_at=m.expires_at,
                        symbols=m.symbols,
                        status="cancelled",
                    )
                )
            else:
                out.append(m)
        if hit:
            _atomic_write(self.path, [x.to_dict() for x in out])
        return hit

    def clear(self) -> int:
        n = len(self.active())
        _atomic_write(self.path, [])
        return n

    def _journal(self, m: Mandate) -> None:
        """Durable record, so a mandate can be reviewed against what the
        run actually did."""
        try:
            from trading.memory.store import MemoryStore

            mem = MemoryStore(self.path.parent / "memory")
            try:
                mem.journal("operator_mandate", m.to_dict(), actor="operator")
            finally:
                mem.close()
        except Exception:
            _LOG.exception("journalling the mandate failed")


def for_context(state_dir: Path) -> list[dict[str, Any]]:
    """Active mandates shaped for the committee/PM context."""
    try:
        return [
            {
                "id": m.id,
                "strength": m.strength,
                "said": m.text,
                "symbols": m.symbols,
                "expires_at": m.expires_at,
            }
            for m in MandateStore(state_dir).active()
        ]
    except Exception:
        _LOG.exception("reading mandates failed")
        return []


# How each strength is to be treated. Shipped to the agents verbatim so
# the operator's tone maps to a documented behaviour rather than to the
# model's mood on the day.
STRENGTH_GUIDANCE = (
    "operator_mandates are standing instructions from the human operator for "
    "THIS run, graded by how firmly he phrased them:\n"
    "- strength 'strong': treat as a strong prior. Act on it unless there is "
    "a concrete, stateable reason not to; if you decline, say so explicitly "
    "and give the reason.\n"
    "- strength 'medium': weigh seriously and address it; you may decline "
    "with a brief reason.\n"
    "- strength 'soft': a suggestion. Consider it, drop it freely if the "
    "data does not support it.\n"
    "NO mandate is an order, and none suspends the risk limits: position, "
    "cluster and gross caps still bind and are applied after you decide. "
    "Never allocate past a cap because the operator asked — say that the cap "
    "binds instead. If a mandate names a symbol outside the allowed universe, "
    "say so plainly rather than substituting something similar."
)
