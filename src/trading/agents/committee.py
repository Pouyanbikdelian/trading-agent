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

from trading.agents.guards import run_guards
from trading.core.logging import logger
from trading.memory.store import MemoryStore

LlmFn = Callable[[str, str], dict[str, Any]]  # (system, prompt) -> parsed JSON

_TAKE_SCHEMA = (
    "Argue your case concretely: name specific tickers, levels, dates and "
    "numbers from the context — no hedged generalities. "
    'Respond ONLY with JSON: {"stance": "bullish|neutral|bearish", '
    '"take": "<4-6 sentences of specific, concrete reasoning>", '
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
    "scout": (
        "You are the Scout — equity research for the NEXT big theme, weeks "
        "to a few months out. Read the headlines block (gossip-grade: weigh "
        "buzz but label it) against sector_momentum_vs_spy_pct (the tape). "
        "SIGNAL HIERARCHY: announced capital commitments outrank commentary "
        "— a named investor deploying real money (fund launches, M&A, "
        "capex programs, strategic stakes; weight by dollar size and the "
        "allocator's credibility) reveals positioning; opinion pieces only "
        "reveal mood. Cite committed-capital headlines explicitly when "
        "they support a theme. "
        "The edge cases you hunt: buzz building BEFORE relative momentum "
        "confirms (early), and buzz peaking AFTER a big run (late, fade it). "
        "ROTATION LENS: off_52w_high_pct separates strong-and-extended "
        "(near the high, no cushion) from washed-out-and-turning (deep "
        "below the high with IMPROVING 1m momentum — a recovery candidate "
        "worth naming). Never pitch a laggard just because it is cheap; "
        "cheap with no catalyst and no momentum turn is a falling knife. "
        "STANDING OPERATOR DIRECTIVE: quantum computing is a permanent "
        "watch theme — track the pure-plays and big-tech quantum programs "
        "continuously and flag accumulation-worthy entries (post-correction, "
        "pre-catalyst) without waiting for momentum confirmation; the "
        "operator's stated conviction is that early-and-cheap beats "
        "late-and-confirmed for this one theme. The clues that matter most: "
        "GOVERNMENT and defense contracts, national-lab partnerships, and "
        "supply-chain/collaboration deals (test equipment, cryogenics, "
        "photonics suppliers) — committed institutional money reveals the "
        "real ecosystem before revenue does. Still report honestly when "
        "the space is overheated. "
        "Name the ONE stock or sector with the best setup. PREFER an "
        "individual stock over an ETF when a clear winner exists within the "
        "sector (e.g. LMT instead of ITA, LLY instead of XLV, JPM instead "
        "of XLF). Use an ETF only when the thesis is purely sector-wide with "
        "no individual standout, or as a diversified hedge. "
        "If nothing is compelling, say so with a flat call. " + _TAKE_SCHEMA
    ),
    "creative": (
        "You are the Creative — a position-blind contrarian wired directly "
        "to social media and gossip. You have NO visibility into the current "
        "book or existing positions. Your job: surface the idea the committee "
        "is missing because they are staring at their existing holdings.\n"
        "\n"
        "SOCIAL SIGNAL LAYER (highest priority input): Headlines tagged "
        "'reddit:u/<username>' or referencing named investors (Druckenmiller, "
        "Ackman, Burry, Chamath, etc.) carry source attribution. Check "
        "source_trust: users with score > 0.60 making specific, falsifiable "
        "calls ('$AMD breaks $180 by August') deserve real weight. Users "
        "below 0.40 are noise — label them but do not build a thesis on them. "
        "New usernames (no trust score yet) get provisional weight based on "
        "the specificity and plausibility of their claim, not their Reddit "
        "upvotes. When you rely on a social call, CITE the source key "
        "(e.g. 'reddit:u/TheCrux99') so the memory system can track it.\n"
        "\n"
        "MARKET STRUCTURE LAYER: Scan sector_momentum_vs_spy_pct for "
        "sectors with improving 1m momentum after a deep off_52w_high_pct "
        "drawdown — washed-out with a turn is a recovery play. Flag the "
        "dominant committee theme as a FADE when it is near its 52w high "
        "with >30% 3m outperformance vs SPY — that is a crowded late-cycle "
        "trade. Look for macro dislocations in the economy block (high rates, "
        "strong USD, credit spread moves) that structurally favor sectors "
        "the committee hasn't mentioned.\n"
        "\n"
        "STOCK OVER ETF: Predict a specific individual stock as your subject "
        "whenever a sector idea has a clear leader. Use an ETF only when "
        "no individual name stands out as the best expression of the thesis. "
        "If you cannot find a genuinely compelling fresh idea, say so with "
        "a flat call — do not manufacture conviction. " + _TAKE_SCHEMA
    ),
}

CHALLENGER_CHARTER = (
    "You are the Challenger — professionally disagreeable. You will be shown "
    "ALL committee takes plus today's market context. Your job: attack the "
    "committee's OVERALL direction and expose every material weakness — "
    "overconfident takes, crowding, stale theses, risks specific to the "
    "current market phase (late-cycle, post-rally, pre-event), and anything "
    "the committee is collectively ignoring. Cite base rates; steelman the "
    "opposite side. Use target_agent='committee' for direction-level "
    "objections. Respond ONLY with JSON: "
    '{"objections": [{"target_agent": "<name or committee>", '
    '"objection": "<3-4 sentences>", '
    '"falsifier": "<what evidence would prove them wrong>"}], '
    '"market_phase_caveat": "<1-2 sentences on what this market phase punishes>"}'
)

MANAGER_CHARTER = (
    "You are the Fund Manager — neutral arbiter. You will be shown all takes, "
    "the Challenger's objections, and each agent's historical calibration. "
    "Weigh agents by track record, not eloquence. Any guard_flags shown are "
    "deterministic mechanical checks (e.g. a name flagged at its 52-week high, "
    "book concentration) — treat them as verified facts, not opinions, and let "
    "them temper conviction. Respond ONLY with JSON: "
    '{"posture": "risk_on|neutral|risk_off", '
    '"proposal": "<3-5 sentences: what you would do and why>", '
    '"watch": "<the one thing that would change your mind>", '
    '"dissent_summary": "<1-2 sentences on where the committee disagrees>"}'
)

_STANCE_SCORE = {"bearish": -1.0, "neutral": 0.0, "bullish": 1.0}

# Specialist context slices. Identical context for everyone produced an
# echo chamber — six takes anchored on the same loudest number. Each
# specialist now sees only what its charter is FOR; the challenger and
# manager still see everything. Unknown agents fall back to the full view.
_VIEW_KEYS: dict[str, tuple[str, ...]] = {
    "quant": (
        "account",
        "positions",
        "macro_dial",
        "vol_surface",
        "style_leader",
        "sector_momentum_vs_spy_pct",
        "economy",
    ),
    "narrator": (
        "macro_dial",
        "dossiers",
        "source_trust",
        "established_lessons",
        "headlines",
        "economy",
    ),
    "street": ("positions", "style_leader", "sector_momentum_vs_spy_pct", "headlines"),
    "position_coach": ("account", "positions", "holds", "k_override", "established_lessons"),
    "risk_officer": (
        "account",
        "positions",
        "vol_surface",
        "macro_dial",
        "spy_vix_triggers",
        "holds",
        "economy",
    ),
    "trader": (
        "positions",
        "vol_surface",
        "spy_vix_triggers",
        "sector_momentum_vs_spy_pct",
        "headlines",
    ),
    "scout": ("sector_momentum_vs_spy_pct", "headlines", "source_trust", "established_lessons"),
    # Creative sees market structure + trust table but NOT the book —
    # position-blindness is the whole point; the trust table is needed
    # so it can weigh named social sources correctly.
    "creative": (
        "sector_momentum_vs_spy_pct",
        "headlines",
        "economy",
        "macro_dial",
        "source_trust",
    ),
}


def _agent_view(name: str, context: dict[str, Any]) -> dict[str, Any]:
    keys = _VIEW_KEYS.get(name)
    if not keys:
        return context
    return {k: context[k] for k in keys if k in context}


def _display(name: str) -> str:
    """Agent name safe for Telegram Markdown (underscores read as italics)."""
    return name.replace("_", " ")


def _clip(text: str, limit: int) -> str:
    """Truncate at a word boundary with an ellipsis — mid-word cuts made
    the Telegram digest read like a dropped call."""
    text = str(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]) + "…"


def _default_llm(system: str, prompt: str) -> dict[str, Any]:
    """Specialist (mid-tier) completion — the cheap, high-volume takes."""
    from trading.agents.llm import complete_json

    return complete_json(system, prompt)


def _frontier_llm(system: str, prompt: str) -> dict[str, Any]:
    """Decision-node completion — challenger + manager run on the frontier model
    (``tier='frontier'``). Their job is judgement and adversarial reasoning,
    where the stronger model earns its cost; ~2 calls per cycle."""
    from trading.agents.llm import complete_json

    return complete_json(system, prompt, tier="frontier")


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
    # Per-role model routing: specialists on the mid-tier model (high volume,
    # low stakes); the two decision nodes (challenger + manager) on the frontier
    # model. An injected ``llm`` (tests) overrides both and keeps runs hermetic.
    specialist_llm = llm or _default_llm
    decision_llm = llm or _frontier_llm
    ctx_block = json.dumps(context, default=str, indent=1)[:18000]
    takes: dict[str, dict[str, Any]] = {}

    for name, charter in CHARTERS.items():
        try:
            view = json.dumps(_agent_view(name, context), default=str, indent=1)[:18000]
            out = specialist_llm(charter, f"Today's context (your specialist slice):\n{view}")
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

    # Deterministic guards: mechanical checks the personas miss. Advisory
    # context for the manager + the digest; never an order gate.
    guard_flags = run_guards(takes, context)

    # --- Challenger round: sees ALL takes + market context
    objections: list[dict[str, Any]] = []
    market_caveat = ""
    try:
        target_block = json.dumps(takes, default=str)[:6000]
        ch = decision_llm(
            CHALLENGER_CHARTER,
            f"Market context:\n{ctx_block[:3000]}\n\nCommittee takes:\n{target_block}",
        )
        objections = list(ch.get("objections", []))[:5]
        market_caveat = str(ch.get("market_phase_caveat", ""))[:300]
        mem.journal(
            "debate",
            {"objections": objections, "market_caveat": market_caveat},
            actor="challenger",
        )
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
                "guard_flags": guard_flags,
            },
            default=str,
        )[:9000]
        ruling = decision_llm(MANAGER_CHARTER, manager_prompt)
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
        "market_caveat": market_caveat,
        "disagreement_index": disagreement,
        "guard_flags": guard_flags,
    }
    mem.journal("committee", {"ruling": ruling, "disagreement": disagreement}, actor="manager")
    return digest


def format_digest_compact(digest: dict[str, Any]) -> str:
    """Executive summary — a few bullets + conclusion. Full debate via /detail."""
    if not digest.get("ok"):
        return f"🤖 Committee did not convene: {digest.get('reason', 'unknown')}"
    icons = {"bullish": "🟢", "neutral": "⚪", "bearish": "🔴"}
    r = digest.get("ruling", {})
    posture = str(r.get("posture", "neutral")).replace("_", " ").upper()
    posture_icon = {"RISK ON": "🟢", "NEUTRAL": "⚪", "RISK OFF": "🔴"}.get(posture, "⚪")
    lines = [
        f"🏛 *Committee* — {posture_icon} *{posture}*  (dissent {digest['disagreement_index']:.1f})",
        "",
    ]
    # One bullet per non-neutral voice, the strongest first; max 5.
    voiced = [
        (n, t) for n, t in digest["takes"].items() if t.get("stance") in ("bullish", "bearish")
    ]
    voiced.sort(key=lambda kv: float(kv[1]["prediction"]["confidence"]), reverse=True)
    for name, t in voiced[:5]:
        lines.append(f"  {icons[t['stance']]} *{_display(name)}*: {_clip(t.get('take', ''), 160)}")
    if digest.get("objections"):
        o = digest["objections"][0]
        lines.append(f"  ⚔️ *challenger*: {_clip(o.get('objection', ''), 160)}")
    if digest.get("market_caveat"):
        lines.append(f"  ⚠️ {_clip(digest['market_caveat'], 160)}")
    lines += [
        "",
        f"*Conclusion:* {_clip(r.get('proposal', ''), 400)}",
        f"_Watching: {_clip(r.get('watch', ''), 160)} · `/detail` for the full debate_",
    ]
    return "\n".join(lines)


def format_digest(digest: dict[str, Any]) -> str:
    """Full Telegram rendering of a committee run (served by /detail)."""
    if not digest.get("ok"):
        return f"🤖 Committee did not convene: {digest.get('reason', 'unknown')}"
    icons = {"bullish": "🟢", "neutral": "⚪", "bearish": "🔴"}
    lines = ["🏛 *Daily committee* — advisory only"]
    for name, t in digest["takes"].items():
        p = t["prediction"]
        lines.append(
            f"{icons.get(t.get('stance', 'neutral'), '⚪')} *{_display(name)}* "
            f"({p['subject']} {p['direction']} {p['horizon_days']}d, "
            f"{float(p['confidence']):.0%}): {_clip(t.get('take', ''), 320)}"
        )
    if digest["objections"]:
        lines.append("")
        lines.append("⚔️ *Challenger:*")
        for o in digest["objections"]:
            lines.append(
                f"  vs *{_display(str(o.get('target_agent', '?')))}*: "
                f"{_clip(o.get('objection', ''), 320)}"
            )
    if digest.get("market_caveat"):
        lines.append("")
        lines.append(f"⚠️ *Market phase:* {digest['market_caveat']}")
    if digest.get("guard_flags"):
        lines.append("")
        lines.append("🛡 *Guards (mechanical):*")
        for g in digest["guard_flags"][:6]:
            lines.append(f"  • {_clip(g, 200)}")
    r = digest["ruling"]
    posture_icon = {"risk_on": "🟢", "neutral": "⚪", "risk_off": "🔴"}.get(
        r.get("posture", "neutral"), "⚪"
    )
    lines += [
        "",
        f"{posture_icon} *Manager — {r.get('posture', 'neutral').replace('_', ' ').upper()}*",
        _clip(r.get("proposal", ""), 600),
        f"_Watching:_ {_clip(r.get('watch', ''), 200)}",
        f"_Disagreement index: {digest['disagreement_index']:.2f} — "
        f"{_clip(r.get('dissent_summary', ''), 200)}_",
    ]
    return "\n".join(lines)
