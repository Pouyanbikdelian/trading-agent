"""The Learning Curator — turns recent evidence into durable lessons.

The memory store has had the full lesson lifecycle since day one —
candidate -> established -> challenged -> retired — but nothing ever
wrote to it. This is the librarian for that filing cabinet.

Twice weekly, after the nightly grading pass, the Historian reads a rolling
week of journal evidence (graded predictions, committee rulings, PM
rebalances) plus the current lesson book, and produces:

* **<=2 new candidate lessons** — durable, falsifiable market
  regularities ("sharp corrections inside uptrends resolved upward
  within N days"), never event recaps ("the Dow fell Wednesday").
* **outcome-linked evidence votes** on existing lessons — every vote names
  a graded prediction or closed episode. Weekly review can interpret the
  evidence, but cannot manufacture it; only three net measured outcomes
  establish a candidate.
* **archive recommendations** for challenged machine lessons whose measured
  evidence has turned decisively against them.  An operator must approve an
  archive; the Curator never silently removes a belief from the vault.

Advisory infrastructure: journal + lesson cards + Telegram. No order path.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from trading.core.logging import logger
from trading.memory.store import MemoryStore

LlmFn = Callable[[str, str], dict[str, Any]]

MAX_NEW_LESSONS = 2

HISTORIAN_CHARTER = (
    "You are the Learning Curator (the desk's Historian) of a systematic trading desk. You will see one "
    "week of journal entries (graded predictions with outcomes, committee "
    "rulings, portfolio changes), a current lesson book, and a compact "
    "long-horizon historical dossier. Your job is distillation, not "
    "narration. Propose at most TWO new lessons, and "
    "ONLY if the week genuinely taught something durable: a falsifiable "
    "regularity that would have been useful BEFORE this week and will be "
    "useful in future weeks. Event recaps, single-instance coincidences, "
    "and vague wisdom ('stay disciplined') are forbidden. Most weeks teach "
    "nothing new — an empty list is a respectable answer.\n\n"
    "Each lesson you propose must be genuinely actionable and understandable "
    "to any reader without extra context. Structure every new lesson as:\n"
    "  title: 5-8 words naming the pattern (e.g. 'Semis lead tech on "
    "momentum breakouts')\n"
    "  body: EXACTLY four sentences covering:\n"
    "    1. What pattern or regularity was observed (the empirical finding).\n"
    "    2. Under what market conditions or triggers it applies "
    "(regime, vol environment, catalyst type).\n"
    "    3. The mechanical reason it works — what drives it structurally "
    "or behaviorally.\n"
    "    4. The concrete action it implies for the desk "
    "(when to act, what size, what to watch for as confirmation).\n\n"
    "SCOPE THE CLAIM — this is what separates a lesson from a superstition. "
    "An unconditioned regularity ('quality names rebound after drawdowns') "
    "is almost always a description of the sample it was found in. A desk "
    "study on 2026-08-05 found exactly that: a rule with a +5.8% edge "
    "pooled over eight years, whose hit rate had been BELOW 50% every year "
    "since 2023 — true on average, useless now, and it would have been "
    "recorded as an established lesson. So state, for every lesson:\n"
    "  applies_when: the conditions under which you claim it holds — "
    "regime, vol, rate environment, sector, market cap, position in the "
    "52-week range. Be specific enough that a reader can tell whether "
    "TODAY qualifies.\n"
    "  fails_when: the conditions under which you expect it to fail, or "
    "where the evidence is silent. A lesson with no stated failure mode "
    "has not been thought through and will be believed for too long.\n"
    "  invalidated_if: the single observation that should retire this "
    "lesson. Name a number or an event, not a feeling.\n"
    "  sample: how many distinct names/weeks/episodes support it, and over "
    "what period. If you cannot say, the lesson is not ready — omit it.\n"
    "Prefer ONE well-conditioned lesson to two vague ones. A lesson whose "
    "conditions you cannot state is a lesson you have not learned.\n\n"
    "The evidence block may contain a 'measured_edge' section: forward "
    "returns of names the desk PICKED versus names it PASSED on, net of "
    "SPY, sliced by ladder rank, market conditions and entry level. That "
    "section is measurement, not opinion — it outranks anything in the "
    "prose. Read 'n_symbols' (distinct names), NOT 'n' (rows): the same "
    "names are re-ranked every day, so rows overstate the sample many "
    "times over. Do not build a lesson on a slice with fewer than 20 "
    "distinct names; say the sample is too thin instead. If the measured "
    "edge contradicts the week's narrative, the measurement wins.\n\n"
    "The historical dossier includes related RETIRED lessons. Treat them as "
    "guardrails against rediscovering an old mistake: explicitly explain why "
    "a materially similar proposal is different, or omit it. A challenged "
    "lesson has already lost sufficient outcome support to leave agent "
    "context. Archival is determined by a separate, evidence-gated operator "
    "workflow; do not recommend or perform it here. Respond ONLY with JSON:\n"
    '{"new_lessons": [{"title": "<5-8 word label>", '
    '"body": "<4-sentence elaboration>", '
    '"applies_when": "<conditions where it holds>", '
    '"fails_when": "<conditions where it fails or evidence is silent>", '
    '"invalidated_if": "<the observation that retires it>", '
    '"sample": "<distinct names / weeks / episodes and over what period>", '
    '"tags": "<comma,separated>", "evidence": "<what this week showed>", '
    '"source_ids": ["<visible pr-... or ep-... outcome ids>"]}]}'
)

# Voting is a separate call with a separate charter, and it never sees
# which lessons this desk's own historian authored. Asked in one breath to
# propose lessons and to judge them, a model leans towards confirming its
# own — and the +3-net-outcome promotion mechanic is only a chaos filter
# if the votes attach to independently recorded evidence.
VOTER_CHARTER = (
    "You are an independent reviewer at a systematic trading desk. You "
    "will see a week of desk evidence and a numbered list of claims about "
    "how markets behave. You did not write these claims and you have no "
    "stake in them. For each claim, decide whether THIS WEEK's evidence "
    "supports it, contradicts it, or does not bear on it at all.\n\n"
    "'Does not bear on it' is the correct answer most of the time and you "
    "should use it freely: a week is a small sample and most weeks are "
    "silent on most claims. Vote only where the evidence in front of you "
    "speaks directly to the claim. A vote you cannot justify in one "
    "concrete sentence — naming a number, a ticker or a dated event from "
    "the evidence — is a vote you should not cast.\n\n"
    "Where a 'measured_edge' or 'historical_dossier' section is present it is measurement rather "
    "than narrative and outranks the prose. Judge sample size by "
    "'n_symbols' (distinct names), not 'n' (rows, which repeat the same "
    "names daily), and ignore any slice under 20 distinct names.\n\n"
    "Respond ONLY with JSON. Omit claims you are not voting on:\n"
    "A vote counts only when it names `evidence_id`: the id of a graded "
    "prediction (`pr-...`) or closed episode (`ep-...`) visible in the "
    "evidence. A synthetic week label, a lesson id, or an invented id is "
    "not evidence. Respond ONLY with JSON:\n"
    '{"votes": [{"lesson_id": "<id>", "supports": true|false, '
    '"evidence_id": "<pr-... or ep-...>", '
    '"why": "<1 concrete sentence citing the evidence>"}]}'
)

# Journal kinds worth distilling, with a per-kind budget so the rows that
# carry measured outcomes cannot be crowded out by heartbeats.
HISTORIAN_KINDS: dict[str, int] = {
    "prediction_graded": 60,  # the only rows carrying measured truth
    "committee": 10,
    "debate": 10,
    "agent_pm": 10,
    "episode": 20,
    "operator_objection": 10,
    "operator_mandate": 10,
    "cycle": 10,
    "take": 15,
}
HISTORIAN_WINDOW_DAYS = 7.0


# The historian writes the desk's PERMANENT beliefs — the only artifact
# here that outlives the week it was made in, and the one every agent
# reads as settled truth once promoted. It ran on the mid-tier model with
# the default 1,200-token ceiling, which is the cheapest configuration in
# the system attached to its most durable output. Two calls a week on the
# frontier model is a rounding error next to the per-cycle committee; a
# badly-conditioned lesson costs for months.
HISTORIAN_MAX_TOKENS = 8000


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt, tier="frontier", max_tokens=HISTORIAN_MAX_TOKENS)


PROMPT_BUDGET = 24_000

# Journal buckets in the order they may be sacrificed when the evidence
# does not fit. Chatter first. The active rulebook and historical dossier
# are never silently dropped: an incomplete permanent-memory review is
# worse than a skipped review with an explicit error.
_TRIM_ORDER: tuple[str, ...] = (
    "take",
    "debate",
    "operator_mandate",
    "cycle",
    "operator_objection",
    "committee",
    "agent_pm",
)


def _budgeted_evidence(
    payload: dict[str, Any],
    budget: int = PROMPT_BUDGET,
    *,
    protected: tuple[str, ...] = ("lesson_book", "historical_dossier"),
) -> str:
    """Serialize under ``budget`` by dropping whole buckets, never slicing.

    Slicing produced invalid JSON ending mid-string, and destroyed
    whichever key sorted last rather than whichever mattered least. A
    caller's protected context is never removed; if it cannot fit, the
    Historian fails safely and the next ops pass makes that visible.
    """
    import copy

    p = copy.deepcopy(payload)

    def fitted() -> str | None:
        s = json.dumps(p, default=str)
        return s if len(s) <= budget else None

    s = fitted()
    if s is not None:
        return s
    journal = p.get("week_journal")
    for kind in _TRIM_ORDER:
        if isinstance(journal, dict) and kind in journal:
            # Halve first, drop second — a thinned bucket still carries
            # the week's shape.
            journal[kind] = journal[kind][: max(1, len(journal[kind]) // 2)]
            s = fitted()
            if s is not None:
                return s
            journal.pop(kind, None)
            s = fitted()
            if s is not None:
                return s
    p.pop("week_journal", None)
    s = fitted()
    if s is not None:
        return s
    unprotected = [key for key in p if key not in protected]
    for key in unprotected:
        p.pop(key, None)
        s = fitted()
        if s is not None:
            return s
    raise ValueError(
        "historian protected memory exceeds prompt budget; refusing a partial permanent-memory review"
    )


def build_week_evidence(mem: MemoryStore, days: float = HISTORIAN_WINDOW_DAYS) -> dict[str, Any]:
    """What the week actually contained — a real time window, by kind.

    Two things this fixes. The window is days rather than a row count, so
    "this week" means this week regardless of how busy the desk was. And
    the rows arrive bucketed with a per-kind budget, so graded outcomes
    survive a noisy week instead of being pushed out by heartbeats.

    ``measured_edge`` is the part with actual numbers in it: forward
    returns of picks versus passes, net of SPY, sliced three ways. Before
    this the historian could only infer whether decisions worked by
    reading prose about them.
    """
    journal = mem.journal_window(
        days, kinds=list(HISTORIAN_KINDS), per_kind_limit=max(HISTORIAN_KINDS.values())
    )
    for kind, budget in HISTORIAN_KINDS.items():
        if kind in journal:
            journal[kind] = journal[kind][-budget:]

    measured: dict[str, Any] = {}
    try:
        # 21d is the horizon the desk actually reasons on; 5d is noise and
        # 63d takes a quarter to say anything.
        measured = {
            "note": (
                "Forward returns net of SPY over the same window. "
                "'n_symbols' is distinct names and is the real sample size; "
                "'n' counts rows, which repeat the same names every day and "
                "overstate it. Ignore any slice under 20 distinct names."
            ),
            "by_origin": mem.edge_report(leg_days=21),
            "by_rank": mem.edge_by_rank(leg_days=21),
            "by_condition": mem.edge_by_condition("vol_bucket", leg_days=21),
            "by_entry_level": mem.edge_by_entry(leg_days=21),
        }
    except Exception:
        logger.bind(component="historian").warning("measured edge unavailable this week")

    return {"window_days": days, "week_journal": journal, "measured_edge": measured}


def run_lesson_vote(
    mem: MemoryStore,
    lesson_book: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    llm: LlmFn | None = None,
    health: dict[str, Any] | None = None,
) -> int:
    """Score existing lessons against the week, in a separate call.

    The reviewer sees the claims stripped of authorship and status — no
    ids it recognises, no support counters hinting which way the desk
    already leans, no marker saying "your historian wrote this one". It
    votes on numbered claims and the numbers are mapped back here.

    Returns the number of votes recorded. Never raises: a failed vote
    leaves the lesson book untouched, which is the safe direction.
    """
    if not lesson_book:
        if health is not None:
            health.update({"ok": None, "reason": "no eligible lessons to vote"})
        return 0
    llm = llm or _default_llm

    # Numbered, shuffled, and stripped. Order carries information too —
    # a book listed oldest-first invites the reviewer to treat the top of
    # the list as the settled ones.
    claims = [{"n": i, "claim": le["statement"]} for i, le in enumerate(lesson_book, start=1)]
    # Stable anonymity: a retry over the same lesson set should not alter
    # the voter prompt purely because process-global random state moved.
    random.Random("|".join(str(le["id"]) for le in lesson_book)).shuffle(claims)
    by_number = {i: le["id"] for i, le in enumerate(lesson_book, start=1)}

    voter_evidence = {
        "week_journal": evidence.get("week_journal", {}),
        "measured_edge": evidence.get("measured_edge", {}),
        "historical_dossier": evidence.get("historical_dossier", {}),
        "claims": claims,
    }
    try:
        prompt = _budgeted_evidence(voter_evidence, protected=("historical_dossier", "claims"))
    except ValueError as e:
        logger.bind(component="historian").warning(f"lesson vote skipped: {e}")
        if health is not None:
            health.update({"ok": False, "reason": str(e)})
        return 0

    # The reviewer may only attach an outcome it was actually shown. Read
    # the *budgeted final prompt*, not the larger pre-budget evidence object;
    # otherwise a trimmed outcome could be cited as if it were visible.
    try:
        prompt_payload = json.loads(prompt)
    except (TypeError, ValueError, json.JSONDecodeError):
        reason = "lesson voter prompt was not valid JSON"
        logger.bind(component="historian").warning(reason)
        if health is not None:
            health.update({"ok": False, "reason": reason})
        return 0
    outcome_ids = mem.completed_outcome_ids(_visible_outcome_ids(prompt_payload))
    # The dossier gives long-horizon context, including outcomes that led
    # to an earlier retired lesson. It is deliberately NOT eligible for a
    # weekly vote: otherwise a new candidate could accumulate three old,
    # cherry-picked outcomes over successive reviews instead of surviving
    # three independent observations after it was proposed.

    try:
        out = llm(VOTER_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="historian").warning(f"lesson vote failed: {e}")
        if health is not None:
            health.update({"ok": False, "reason": f"lesson vote failed: {e}"})
        return 0

    if not isinstance(out, dict):
        reason = "lesson voter returned a non-object JSON payload"
        logger.bind(component="historian").warning(reason)
        if health is not None:
            health.update({"ok": False, "reason": reason})
        return 0
    votes = out.get("votes", [])
    if not isinstance(votes, list):
        reason = "lesson voter returned a non-list votes field"
        logger.bind(component="historian").warning(reason)
        if health is not None:
            health.update({"ok": False, "reason": reason})
        return 0
    # Validate every actionable item before recording any evidence. A
    # missing ``supports`` used to become a real contradiction via bool(),
    # and a malformed later item could leave a partial unaudited vote batch.
    clean_votes: list[tuple[str, dict[str, Any]]] = []
    for vote in votes[:10]:
        if not isinstance(vote, dict):
            reason = "lesson voter returned a non-object vote"
        elif type(vote.get("supports")) is not bool:
            reason = "lesson voter vote supports must be boolean"
        elif not isinstance(vote.get("lesson_id"), (str, int)) or isinstance(
            vote.get("lesson_id"), bool
        ):
            reason = "lesson voter vote lesson_id must be a string or number"
        elif not isinstance(vote.get("evidence_id"), str):
            reason = "lesson voter vote evidence_id must be a string"
        else:
            raw_lesson_id = vote["lesson_id"]
            try:
                lesson_id = by_number.get(int(raw_lesson_id))
            except (TypeError, ValueError):
                lesson_id = (
                    str(raw_lesson_id)
                    if any(le["id"] == str(raw_lesson_id) for le in lesson_book)
                    else None
                )
            if lesson_id is None:
                reason = "lesson voter vote referenced a lesson outside this review"
            elif vote["evidence_id"] not in outcome_ids:
                reason = "lesson voter vote cited an outcome not visible in this review"
            else:
                clean_votes.append((lesson_id, vote))
                continue
        logger.bind(component="historian").warning(reason)
        if health is not None:
            health.update({"ok": False, "reason": reason})
        return 0
    if health is not None:
        health.update({"ok": True, "reason": ""})

    voted = 0
    voted_lessons: set[str] = set()
    for lid, vote in clean_votes:
        if lid in voted_lessons:
            continue
        evidence_id = vote["evidence_id"]
        if mem.add_evidence(
            lid,
            evidence_id,
            supports=bool(vote.get("supports")),
            reason=str(vote.get("why", "")),
            evidence_kind="outcome",
            actor="learning_curator",
        ):
            voted += 1
            voted_lessons.add(lid)
    return voted


def _visible_outcome_ids(evidence: dict[str, Any]) -> set[str]:
    """Completed prediction/episode ids the Curator actually saw this pass."""
    ids: set[str] = set()
    journal = evidence.get("week_journal", {})
    if not isinstance(journal, dict):
        return ids
    for kind, prefix in (("prediction_graded", "pr-"), ("episode", "ep-")):
        rows = journal.get(kind, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            payload = row.get("payload", {}) if isinstance(row, dict) else {}
            value = payload.get("id") if isinstance(payload, dict) else None
            if isinstance(value, str) and value.startswith(prefix):
                ids.add(value)
    return ids


def _safe_historian_tags(raw: object) -> str:
    """Keep model tags descriptive; they must never claim operator authorship."""
    tags = re.findall(r"[a-z0-9][a-z0-9_-]{0,31}", str(raw).lower())
    return " ".join(tag for tag in tags if "operator" not in tag)[:120]


def _lesson_fingerprint(statement: str) -> str:
    """Conservative exact-content dedupe for retries, not semantic deletion."""
    return " ".join(re.findall(r"[a-z0-9]+", statement.lower()))


def run_historian(
    mem: MemoryStore,
    *,
    llm: LlmFn | None = None,
    now: datetime | None = None,
    conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One twice-weekly, evidence-gated Learning Curator pass."""
    llm = llm or _default_llm
    now = now or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        raise ValueError("historian timestamp must be timezone-aware")
    if conditions is not None and not isinstance(conditions, dict):
        raise ValueError("historian conditions must be an object")
    current_conditions = conditions or {}
    review = mem.lesson_review(current_conditions)

    def failed(reason: str) -> dict[str, Any]:
        """Persist failures so stale success can never look like current health."""
        run_id = mem.record_curator_run(
            status="failed",
            conditions=current_conditions,
            reviewed=len(review["queue"]),
            created=0,
            voted=0,
            vote_ok=None,
            archive_recommendations=len(review["archive_recommendations"]),
            actions=list(review["review_actions"]),
            reason=reason,
            ts=now,
        )
        digest = {
            "ok": False,
            "ts": now.isoformat(),
            "reason": reason,
            "curator_run_id": run_id,
            "reviewed": len(review["queue"]),
        }
        mem.journal("historian", digest, actor="learning_curator")
        return digest

    lesson_book = [
        {
            "id": r["id"],
            "status": r["status"],
            "statement": r["statement"],
            "support": r["support"],
            "contradict": r["contradict"],
            "outcome_support": r["outcome_support"],
            "outcome_contradict": r["outcome_contradict"],
            "conditions": r["conditions"],
            "scope": r["scope"],
            "retrieval_role": r.get("retrieval_role", "review_queue"),
            "evidence": mem.lesson_evidence(r["id"], limit=8),
        }
        for r in review["queue"]
    ]
    evidence = build_week_evidence(mem)
    try:
        evidence["historical_dossier"] = mem.historian_dossier(
            focus_statements=[str(le["statement"]) for le in lesson_book]
        )
    except Exception as e:
        logger.bind(component="historian").exception("historical memory retrieval failed")
        return failed(f"historical memory unavailable: {e}")
    # Budgeted, not sliced. A raw [:24000] cut here would truncate
    # whichever key happened to land last — and the keys that matter most
    # (measured_edge, the lesson book) are not first. Same failure the PM
    # prompt had; drop whole low-value buckets instead.
    try:
        prompt = _budgeted_evidence({**evidence, "lesson_book": lesson_book})
    except ValueError as e:
        logger.bind(component="historian").warning(str(e))
        return failed(str(e))

    try:
        out = llm(HISTORIAN_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="historian").warning(f"historian call failed: {e}")
        return failed(f"historian call failed: {e}")
    if not isinstance(out, dict):
        return failed("historian returned a non-object JSON payload")
    new_lessons = out.get("new_lessons", [])
    if not isinstance(new_lessons, list):
        return failed("historian returned a non-list new_lessons field")
    try:
        prompt_payload = json.loads(prompt)
    except (TypeError, ValueError, json.JSONDecodeError):
        return failed("historian prompt was not valid JSON")

    created: list[str] = []
    created_ids: set[str] = set()
    lesson_bodies: dict[str, str] = {}
    candidate_actions: list[dict[str, Any]] = []
    candidate_rejections: list[str] = []
    # Only outcomes that survived prompt budgeting can be cited. The store
    # then confirms the id is a completed prediction/episode, not merely a
    # journal-shaped string injected by another component.
    visible_outcome_ids = mem.completed_outcome_ids(_visible_outcome_ids(prompt_payload))

    def reject_candidate(reason: str, lesson: object) -> None:
        raw_ids = lesson.get("source_ids", []) if isinstance(lesson, dict) else []
        evidence_ids = (
            [str(value) for value in raw_ids[:12] if isinstance(value, str)]
            if isinstance(raw_ids, list)
            else []
        )
        candidate_rejections.append(reason)
        candidate_actions.append(
            {
                "lesson_id": None,
                "rank": None,
                "action": "candidate_rejected",
                "before_status": None,
                "after_status": None,
                "reason": reason,
                "evidence_ids": evidence_ids,
            }
        )

    status_before = {str(row["id"]): str(row["status"]) for row in mem.lessons()}
    known_fingerprints = {_lesson_fingerprint(str(row["statement"])) for row in mem.lessons()}
    for lesson in new_lessons[:MAX_NEW_LESSONS]:
        if not isinstance(lesson, dict):
            reject_candidate("Candidate rejected: lesson entry must be an object.", lesson)
            continue
        title = str(lesson.get("title", "")).strip()
        body = str(lesson.get("body", "")).strip()
        # Legacy fallback: LLM may still emit "statement" during migration.
        fallback = str(lesson.get("statement", "")).strip()
        if title and body:
            stmt = f"{title}\n\n{body}"
        elif title:
            stmt = title
        elif body:
            stmt = body
        else:
            stmt = fallback
        if len(stmt) < 20:  # garbage guard
            continue
        # Carry the scope into the stored statement. The conditions are
        # the difference between a lesson and a superstition, and the
        # statement is the ONLY field any agent ever reads — a condition
        # that lives in the JSON but not in the text is a condition the
        # desk will never see. Explicitly labelled so a reader can tell
        # the claim from its boundaries.
        for label, key in (
            ("Applies when", "applies_when"),
            ("Fails when", "fails_when"),
            ("Retire if", "invalidated_if"),
            ("Sample", "sample"),
        ):
            val = str(lesson.get(key, "")).strip()
            if val:
                stmt += f"\n{label}: {val[:300]}"
        scope = {
            key: str(lesson.get(key, "")).strip()[:300]
            for key in ("applies_when", "fails_when", "invalidated_if", "sample")
            if str(lesson.get(key, "")).strip()
        }
        raw_source_ids = lesson.get("source_ids")
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            reject_candidate("Candidate rejected: source_ids must be a non-empty array.", lesson)
            continue
        if len(raw_source_ids) > 12 or not all(isinstance(value, str) for value in raw_source_ids):
            reject_candidate(
                "Candidate rejected: source_ids must contain at most 12 string outcome ids.", lesson
            )
            continue
        source_ids = list(raw_source_ids)
        if len(set(source_ids)) != len(source_ids):
            reject_candidate("Candidate rejected: source_ids must be unique.", lesson)
            continue
        if not set(source_ids).issubset(visible_outcome_ids):
            reject_candidate(
                "Candidate rejected: every source id must be a verified outcome visible in this review.",
                lesson,
            )
            continue
        fingerprint = _lesson_fingerprint(stmt)
        if fingerprint in known_fingerprints:
            candidate_actions.append(
                {
                    "lesson_id": None,
                    "rank": None,
                    "action": "candidate_duplicate_suppressed",
                    "before_status": None,
                    "after_status": None,
                    "reason": "Candidate suppressed because an identical lesson already exists.",
                    "evidence_ids": source_ids,
                }
            )
            continue
        lid = mem.add_lesson(
            stmt,
            origin_episodes=source_ids,
            tags=_safe_historian_tags(lesson.get("tags", "")),
            conditions={"snapshot": current_conditions, "scope": scope},
            actor="learning_curator",
        )
        label = title or stmt[:80]
        created.append(f"{lid}: {label}")
        created_ids.add(lid)
        known_fingerprints.add(fingerprint)
        candidate_actions.append(
            {
                "lesson_id": lid,
                "rank": None,
                "action": "created",
                "before_status": None,
                "after_status": "candidate",
                "reason": "New candidate lesson proposed by the Learning Curator.",
                "evidence_ids": source_ids,
            }
        )
        if body:
            lesson_bodies[lid] = body

    # Voting is its own call. The lessons just created are excluded: a
    # candidate proposed ten seconds ago has no week of evidence behind
    # it, and letting it collect support on the day it was written would
    # start it a third of the way to promotion for free.
    vote_health: dict[str, Any] = {"ok": None, "reason": ""}
    voted = run_lesson_vote(
        mem,
        [le for le in lesson_book if le["id"] not in created_ids],
        evidence,
        llm=llm,
        health=vote_health,
    )
    # This is a rotation marker, not evidence or expiry. A candidate from a
    # quiet regime remains a candidate and comes back after its peers have
    # had their bounded review turn.
    mem.mark_lessons_reviewed([str(le["id"]) for le in lesson_book])

    # Promotion/challenge transitions are store-owned deterministic
    # consequences of measured evidence.  The LLM is never allowed to
    # manufacture either transition or archive a lesson directly.
    status_after = {str(row["id"]): str(row["status"]) for row in mem.lessons()}
    actions = [dict(action) for action in review["review_actions"]]
    action_by_lesson = {str(action["lesson_id"]): action for action in actions}
    for lesson_id, before_status in status_before.items():
        after_status = status_after.get(lesson_id, before_status)
        if after_status == before_status:
            continue
        action = action_by_lesson.get(lesson_id)
        if action is None:
            action = {"lesson_id": lesson_id, "rank": None, "evidence_ids": []}
            actions.append(action)
        transition = (
            "promoted"
            if after_status == "established"
            else "challenged"
            if after_status == "challenged"
            else "reviewed"
        )
        action.update(
            {
                "action": transition,
                "before_status": before_status,
                "after_status": after_status,
                "reason": f"Measured outcome rule moved the lesson from {before_status} to {after_status}.",
                "evidence_ids": [
                    item["id"]
                    for item in mem.lesson_evidence(lesson_id, limit=8)
                    if item["kind"] == "outcome"
                ],
            }
        )
    actions.extend(candidate_actions)
    refreshed_review = mem.lesson_review(current_conditions)
    for recommendation in refreshed_review["archive_recommendations"]:
        lesson_id = str(recommendation["lesson_id"])
        action = action_by_lesson.get(lesson_id)
        if action is None:
            action = {"lesson_id": lesson_id, "rank": None}
            actions.append(action)
        action.update(
            {
                "action": "archive_recommended",
                "before_status": "challenged",
                "after_status": "challenged",
                "reason": recommendation["reason"],
                "evidence_ids": [
                    item["id"]
                    for item in mem.lesson_evidence(lesson_id, limit=8)
                    if item["kind"] == "outcome"
                ],
            }
        )

    degraded_reasons = list(candidate_rejections)
    if vote_health["ok"] is False:
        degraded_reasons.append(str(vote_health["reason"]))
    run_status = "degraded" if degraded_reasons else "completed"
    run_reason = " ".join(degraded_reasons)[:1_000]
    curator_run_id = mem.record_curator_run(
        status=run_status,
        conditions=current_conditions,
        reviewed=len(review["queue"]),
        created=len(created),
        voted=voted,
        vote_ok=vote_health["ok"],
        archive_recommendations=len(refreshed_review["archive_recommendations"]),
        actions=actions,
        reason=run_reason,
        ts=now,
    )

    digest = {
        "ok": True,
        "ts": now.isoformat(),
        "created": created,
        "lesson_bodies": lesson_bodies,
        "voted": voted,
        # Compatibility for Telegram/tests that used the old field.  The
        # Curator only recommends archival; approval performs it later.
        "retired": 0,
        "reviewed": len(review["queue"]),
        "vote_ok": vote_health["ok"],
        "archive_recommendations": [
            item["lesson_id"] for item in refreshed_review["archive_recommendations"]
        ],
        "curator_run_id": curator_run_id,
        "degraded_reason": run_reason,
        "rejected_candidates": len(candidate_rejections),
    }
    mem.journal("historian", digest, actor="learning_curator")
    return digest


def format_historian_digest(digest: dict[str, Any]) -> str:
    if not digest.get("ok"):
        return f"🤖 Learning Curator skipped: {digest.get('reason', 'unknown')}"
    lines = [
        f"📚 *Learning Curator* — {digest.get('reviewed', 0)} lessons reviewed, "
        f"{digest['voted']} evidence votes"
    ]
    if digest.get("vote_ok") is False:
        lines[0] += " · ⚠️ voter degraded"
    if digest.get("rejected_candidates"):
        lines[0] += f" · ⚠️ {digest['rejected_candidates']} invalid candidate(s) rejected"
    lines[0] += "."
    if digest["created"]:
        lines.append("*New candidate lessons:*")
        for c in digest["created"]:
            # c is "{id}: {title}" — show the label, not the full body
            label = c[:160]
            lines.append(f"  • {label}")
            # If there's a stored body, show the first sentence as a teaser.
            body_hint = digest.get("lesson_bodies", {}).get(c.split(":")[0].strip(), "")
            if body_hint:
                first_sentence = body_hint.split(".")[0].strip()
                lines.append(f"    _{first_sentence}._")
    else:
        lines.append("_no new lessons this pass — the bar is high by design_")
    recommendations = list(digest.get("archive_recommendations", []))
    if recommendations:
        lines.append(
            f"*Archive review required:* {len(recommendations)} challenged machine lesson(s) "
            "meet the measured-evidence threshold."
        )
    return "\n".join(lines)
