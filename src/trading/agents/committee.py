"""The committee — seven personas, one debate, one digest.

Flow (concept doc §debate protocol):

1. Specialists (quant, narrator, street, position_coach, risk_officer,
   trader) each produce a TAKE: stance + falsifiable prediction.
2. The Challenger reads all takes and attacks the two highest-conviction
   ones.
3. The Manager synthesizes: proposed sleeve posture, what would change
   its mind, disagreement index.
4. Everything is journaled; every prediction enters the scorecard (and
   thus, on grading, the source-trust ledger). The digest goes to
   Telegram. NOTHING here touches the order path.

Testability: ``run_committee(llm=...)`` accepts an injected completion
function so tests run hermetically without keys or network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from trading.core.logging import logger
from trading.memory.store import MemoryStore

LlmFn = Callable[[str, str], dict[str, Any]]  # (system, prompt) -> parsed JSON

_TAKE_SCHEMA = (
    'Respond ONLY with JSON: {"stance": "bullish|neutral|bearish", '
    '"take": "<2-3 sentences>", '
    '"prediction": {"subject": "<ticker or SPY>", "direction": "up|down|flat", '
    '"horizon_days": <int 1-30>, "confidence": <0.5-0.95>}, '
    '"sources": ["<source keys you relied on>"], "cited_lessons": ["<lesson ids>"]}'
)

CHARTERS: dict[str, str] = {
    "quant": (
        "You are the Quant on a small systematic trading desk. You trust the "
        "numbers in the context block (momentum ranks, regime, vol surface, "
        "macro dial) over any story. Be terse and specific. " + _TAKE_SCHEMA
    ),
    "narrator": (
        "You are the Narrator: you read geopolitics, central-bank psychology "
        "and crowd positioning. Use the World State dossiers in context; say "
        "what is priced in versus what is not. Weigh sources by their trust "
        "scores but do not ignore low-trust gossip — label it. " + _TAKE_SCHEMA
    ),
    "street": (
        "You are the Street analyst: you track sell-side ratings, price "
        "targets and revision momentum, and where consensus disagrees with "
        "price action. Crowded love and crowded hate both matter. " + _TAKE_SCHEMA
    ),
    "position_coach": (
        "You are the Position Coach: you care only about OUR book. For each "
        "position note where it was bought in its 52-week range (top vs dip), "
        "unrealized P&L and whether the original thesis still holds. " + _TAKE_SCHEMA
    ),
    "risk_officer": (
        "You are the Risk Officer: your job is what can hurt us this week. "
        "Read the vol surface, macro stress channels and concentration. You "
        "are allowed to say 'nothing actionable'. " + _TAKE_SCHEMA
    ),
    "trader": (
        "You are the Trader: tactical, this-week horizon. Liquidity, timing, "
        "what the tape is telling you. You dislike stale opinions. " + _TAKE_SCHEMA
    ),
}

CHALLENGER_CHARTER = (
    "You are the Challenger — professionally disagreeable. You will be shown "
    "the committee's takes. Attack the TWO highest-confidence takes: cite "
    "base rates, ways the thesis fails, and what evidence would falsify it. "
    "Steelman the opposite side. Respond ONLY with JSON: "
    '{"objections": [{"target_agent": "<name>", "objection": "<3-4 sentences>", '
    '"falsifier": "<what evidence would prove them wrong>"}]}'
)

MANAGER_CHARTER = (
    "You are the Fund Manager — neutral arbiter. You will be shown all takes, "
    "the Challenger's objections, and each agent's historical calibration. "
    "Weigh agents by track record, not eloquence. Respond ONLY with JSON: "
    '{"posture": "risk_on|neutral|risk_off", '
    '"proposal": "<3-5 sentences: what you would do and why>", '
    '"watch": "<the one thing that would change your mind>", '
    '"dissent_summary": "<1-2 sentences on where the committee disagrees>"}'
)

_STANCE_SCORE = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


def run_committee(
    context: dict[str, Any],
    mem: MemoryStore,
    *,
    llm: LlmFn | None = None,
    calibration: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One full committee cycle. Returns the digest payload (also journaled).

    ``context``: pre-gathered system state (positions, signals, monitors,
    dossiers, trust table, lessons). Built by the runner — this function
    does no I/O beyond the LLM and memory writes, so it stays testable.
    """
    llm = llm or _default_llm
    ctx_block = json.dumps(context, default=str, indent=1)[:8000]
    takes: dict[str, dict[str, Any]] = {}

    for name, charter in CHARTERS.items():
        try:
            out = llm(charter, f"Today's context:\n{ctx_block}")
            pred = out.get("prediction") or {}
            if not {"subject", "direction", "horizon_days", "confidence"} <= set(pred):
                raise ValueError("take missing falsifiable prediction")
            takes[name] = out
            pid = mem.add_prediction(
                agent=name,
                subject=str(pred["subject"]),
                direction=str(pred["direction"]),
                horizon_days=int(pred["horizon_days"]),
                confidence=float(pred["confidence"]),
                statement=str(out.get("take", ""))[:500],
                sources=[str(s) for s in out.get("sources", [])][:8],
            )
            mem.journal("take", {"agent": name, "prediction_id": pid, **out}, actor=name)
        except Exception as e:
            logger.bind(component="agents", agent=name).warning(f"take failed: {e}")

    if not takes:
        return {"ok": False, "reason": "no agent produced a valid take"}

    # --- Challenger round
    objections: list[dict[str, Any]] = []
    try:
        ranked = sorted(
            takes.items(), key=lambda kv: kv[1]["prediction"]["confidence"], reverse=True
        )
        target_block = json.dumps(dict(ranked[:4]), default=str)[:6000]
        ch = llm(
            CHALLENGER_CHARTER, f"Committee takes (attack the top-confidence two):\n{target_block}"
        )
        objections = list(ch.get("objections", []))[:3]
        mem.journal("debate", {"objections": objections}, actor="challenger")
    except Exception as e:
        logger.bind(component="agents", agent="challenger").warning(f"challenge failed: {e}")

    # --- Manager synthesis
    stances = [_STANCE_SCORE.get(t.get("stance", "neutral"), 0.0) for t in takes.values()]
    disagreement = float(max(stances) - min(stances)) / 2.0 if stances else 0.0
    ruling: dict[str, Any] = {}
    try:
        manager_prompt = json.dumps(
            {
                "takes": takes,
                "objections": objections,
                "calibration": calibration or [],
                "disagreement_index": disagreement,
            },
            default=str,
        )[:9000]
        ruling = llm(MANAGER_CHARTER, manager_prompt)
    except Exception as e:
        logger.bind(component="agents", agent="manager").warning(f"ruling failed: {e}")
        ruling = {
            "posture": "neutral",
            "proposal": "(manager unavailable)",
            "watch": "",
            "dissent_summary": "",
        }

    digest = {
        "ok": True,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "takes": takes,
        "objections": objections,
        "ruling": ruling,
        "disagreement_index": disagreement,
    }
    mem.journal("committee", {"ruling": ruling, "disagreement": disagreement}, actor="manager")
    return digest


def format_digest(digest: dict[str, Any]) -> str:
    """Telegram-ready rendering of a committee run."""
    if not digest.get("ok"):
        return f"🤖 Committee did not convene: {digest.get('reason', 'unknown')}"
    icons = {"bullish": "🟢", "neutral": "⚪", "bearish": "🔴"}
    lines = ["🏛 *Daily committee* — advisory only"]
    for name, t in digest["takes"].items():
        p = t["prediction"]
        lines.append(
            f"{icons.get(t.get('stance', 'neutral'), '⚪')} *{name}* "
            f"({p['subject']} {p['direction']} {p['horizon_days']}d, "
            f"{float(p['confidence']):.0%}): {str(t.get('take', ''))[:160]}"
        )
    if digest["objections"]:
        lines.append("")
        lines.append("⚔️ *Challenger:*")
        for o in digest["objections"]:
            lines.append(
                f"  vs *{o.get('target_agent', '?')}*: {str(o.get('objection', ''))[:180]}"
            )
    r = digest["ruling"]
    posture_icon = {"risk_on": "🟢", "neutral": "⚪", "risk_off": "🔴"}.get(
        r.get("posture", "neutral"), "⚪"
    )
    lines += [
        "",
        f"{posture_icon} *Manager — {r.get('posture', 'neutral').replace('_', ' ').upper()}*",
        str(r.get("proposal", ""))[:400],
        f"_Watching:_ {str(r.get('watch', ''))[:160]}",
        f"_Disagreement index: {digest['disagreement_index']:.2f} — "
        f"{str(r.get('dissent_summary', ''))[:160]}_",
    ]
    return "\n".join(lines)
