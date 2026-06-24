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
    "Separately, vote on EXISTING lessons: did this week's evidence support "
    "or contradict them? Only vote where the week actually bears on the "
    "lesson. Respond ONLY with JSON:\n"
    '{"new_lessons": [{"title": "<5-8 word label>", '
    '"body": "<4-sentence elaboration>", '
    '"tags": "<comma,separated>", "evidence": "<what this week showed>"}], '
    '"votes": [{"lesson_id": "<id>", "supports": true|false, '
    '"why": "<1 sentence>"}], '
    '"retire": [{"lesson_id": "<id>", "why": "<1 sentence>"}]}'
)


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


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
    prompt = json.dumps(
        {
            "week_journal": mem.journal_tail(80),
            "lesson_book": lesson_book,
        },
        default=str,
    )[:24000]

    try:
        out = llm(HISTORIAN_CHARTER, prompt)
    except Exception as e:
        logger.bind(component="historian").warning(f"historian call failed: {e}")
        return {"ok": False, "reason": f"historian call failed: {e}"}

    known = {r["id"] for r in lesson_book}
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

    voted = 0
    for vote in list(out.get("votes", []))[:10]:
        lid = str(vote.get("lesson_id", ""))
        if lid in known:
            mem.add_evidence(lid, week_tag, supports=bool(vote.get("supports")))
            voted += 1

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
