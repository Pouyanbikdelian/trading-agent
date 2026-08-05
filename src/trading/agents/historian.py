"""The Historian — turns a week of evidence into at most two lessons.

The memory store has had the full lesson lifecycle since day one —
candidate -> established (3+ net supporting episodes) -> retired — but
nothing ever wrote to it. This is the librarian for that filing cabinet.

Weekly, after the Friday grading pass, the Historian reads the week's
journal (graded predictions, committee rulings, PM rebalances) plus the
current lesson book, and produces:

* **<=2 new candidate lessons** — durable, falsifiable market
  regularities ("sharp corrections inside uptrends resolved upward
  within N days"), never event recaps ("the Dow fell Wednesday").
* **evidence votes** on existing candidates — this week supported or
  contradicted them. Promotion to established (what the committee
  actually sees in context) happens only through the store's existing
  +3-net-support mechanic, so a lesson must survive ~a month of weekly
  scrutiny before any agent treats it as truth. That is the chaos
  filter: the LLM proposes, accumulated evidence disposes.
* **retirements** for established lessons the evidence has turned against.

Advisory infrastructure: journal + lesson cards + Telegram. No order path.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from trading.core.logging import logger
from trading.memory.store import MemoryStore

LlmFn = Callable[[str, str], dict[str, Any]]

MAX_NEW_LESSONS = 2

HISTORIAN_CHARTER = (
    "You are the Historian of a systematic trading desk. You will see one "
    "week of journal entries (graded predictions with outcomes, committee "
    "rulings, portfolio changes) and the current lesson book. Your job is "
    "distillation, not narration. Propose at most TWO new lessons, and "
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
    "The evidence block may contain a 'measured_edge' section: forward "
    "returns of names the desk PICKED versus names it PASSED on, net of "
    "SPY, sliced by ladder rank, market conditions and entry level. That "
    "section is measurement, not opinion — it outranks anything in the "
    "prose. Read 'n_symbols' (distinct names), NOT 'n' (rows): the same "
    "names are re-ranked every day, so rows overstate the sample many "
    "times over. Do not build a lesson on a slice with fewer than 20 "
    "distinct names; say the sample is too thin instead. If the measured "
    "edge contradicts the week's narrative, the measurement wins.\n\n"
    "You may also propose retiring an ESTABLISHED lesson the evidence has "
    "turned against. Respond ONLY with JSON:\n"
    '{"new_lessons": [{"title": "<5-8 word label>", '
    '"body": "<4-sentence elaboration>", '
    '"tags": "<comma,separated>", "evidence": "<what this week showed>"}], '
    '"retire": [{"lesson_id": "<id>", "why": "<1 sentence>"}]}'
)

# Voting is a separate call with a separate charter, and it never sees
# which lessons this desk's own historian authored. Asked in one breath to
# propose lessons and to judge them, a model leans towards confirming its
# own — and the +3-net-support promotion mechanic is only a chaos filter
# if the votes are something closer to independent evidence.
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
    "Where a 'measured_edge' section is present it is measurement rather "
    "than narrative and outranks the prose. Judge sample size by "
    "'n_symbols' (distinct names), not 'n' (rows, which repeat the same "
    "names daily), and ignore any slice under 20 distinct names.\n\n"
    "Respond ONLY with JSON. Omit claims you are not voting on:\n"
    '{"votes": [{"lesson_id": "<id>", "supports": true|false, '
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


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


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
    week_tag: str,
    llm: LlmFn | None = None,
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
        return 0
    llm = llm or _default_llm

    # Numbered, shuffled, and stripped. Order carries information too —
    # a book listed oldest-first invites the reviewer to treat the top of
    # the list as the settled ones.
    claims = [{"n": i, "claim": le["statement"]} for i, le in enumerate(lesson_book, start=1)]
    random.shuffle(claims)
    by_number = {i: le["id"] for i, le in enumerate(lesson_book, start=1)}

    prompt = json.dumps(
        {
            "week_journal": evidence.get("week_journal", {}),
            "measured_edge": evidence.get("measured_edge", {}),
            "claims": claims,
        },
        default=str,
    )[:24000]

    try:
        out = llm(VOTER_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="historian").warning(f"lesson vote failed: {e}")
        return 0

    voted = 0
    for vote in list(out.get("votes", []))[:10]:
        raw = vote.get("lesson_id")
        try:
            lid = by_number.get(int(raw))
        except (TypeError, ValueError):
            # Tolerate a reviewer that echoes the real id anyway.
            lid = str(raw) if any(le["id"] == str(raw) for le in lesson_book) else None
        if not lid:
            continue
        mem.add_evidence(lid, week_tag, supports=bool(vote.get("supports")))
        voted += 1
    return voted


def run_historian(mem: MemoryStore, *, llm: LlmFn | None = None) -> dict[str, Any]:
    """One weekly distillation pass. Returns a digest payload."""
    llm = llm or _default_llm
    week_tag = f"wk-{datetime.now(tz=timezone.utc).date().isoformat()}"

    lesson_book = [
        {
            "id": r["id"],
            "status": r["status"],
            "statement": r["statement"],
            "support": r["support"],
            "contradict": r["contradict"],
        }
        for r in mem.lessons()
        if r["status"] in ("candidate", "established")
    ]
    evidence = build_week_evidence(mem)
    prompt = json.dumps({**evidence, "lesson_book": lesson_book}, default=str)[:24000]

    try:
        out = llm(HISTORIAN_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="historian").warning(f"historian call failed: {e}")
        return {"ok": False, "reason": f"historian call failed: {e}"}

    created: list[str] = []
    lesson_bodies: dict[str, str] = {}
    for lesson in list(out.get("new_lessons", []))[:MAX_NEW_LESSONS]:
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
        lid = mem.add_lesson(
            stmt, origin_episodes=[week_tag], tags=str(lesson.get("tags", ""))[:120]
        )
        label = title or stmt[:80]
        created.append(f"{lid}: {label}")
        if body:
            lesson_bodies[lid] = body

    # Voting is its own call. The lessons just created are excluded: a
    # candidate proposed ten seconds ago has no week of evidence behind
    # it, and letting it collect support on the day it was written would
    # start it a third of the way to promotion for free.
    voted = run_lesson_vote(
        mem,
        [le for le in lesson_book if le["id"] not in {c.split(":")[0] for c in created}],
        evidence,
        week_tag=week_tag,
        llm=llm,
    )

    retired = 0
    established = {r["id"] for r in lesson_book if r["status"] == "established"}
    for r in list(out.get("retire", []))[:3]:
        lid = str(r.get("lesson_id", ""))
        if lid in established:
            mem.retire_lesson(lid, str(r.get("why", ""))[:200])
            retired += 1

    digest = {
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "created": created,
        "lesson_bodies": lesson_bodies,
        "voted": voted,
        "retired": retired,
    }
    mem.journal("historian", digest, actor="historian")
    return digest


def format_historian_digest(digest: dict[str, Any]) -> str:
    if not digest.get("ok"):
        return f"🤖 Historian skipped: {digest.get('reason', 'unknown')}"
    lines = [f"📜 *Historian* — weekly distillation ({digest['voted']} votes"]
    lines[0] += f", {digest['retired']} retired)" if digest.get("retired") else ")"
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
        lines.append("_no new lessons this week — the bar is high by design_")
    return "\n".join(lines)
