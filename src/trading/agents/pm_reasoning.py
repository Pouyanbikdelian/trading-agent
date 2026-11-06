"""Evidence and review facts for the PM; never selects weights or creates orders."""

from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Any

from trading.core.text import clip

# A research thesis can outlive this window, but its event claims and votes
# need revalidation. Retain the history rather than silently treating it as new.
_REVALIDATE_AFTER_HOURS = 72


def _age_hours(value: object, now: datetime) -> float | None:
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError:
        try:
            stamp = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
    if stamp.tzinfo is None:
        return None
    age = (now - stamp).total_seconds() / 3600
    return age if age >= 0 else None


def _date_evidence(rows: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    dated = []
    for row in rows:
        age = _age_hours(row.get("ts"), now)
        dated.append(
            {
                **row,
                "age_hours": round(age, 1) if age is not None else None,
                "requires_revalidation": age is None or age > _REVALIDATE_AFTER_HOURS,
            }
        )
    return dated


def _fresh_headlines(context: dict[str, Any], now: datetime) -> dict[str, Any]:
    """A newly fetched RSS file can contain months-old articles."""
    result = dict(context)
    headlines = context.get("headlines")
    if not isinstance(headlines, list):
        return result
    fresh = []
    for row in headlines:
        age = _age_hours(row.get("published"), now) if isinstance(row, dict) else None
        if age is not None and age <= _REVALIDATE_AFTER_HOURS:
            fresh.append(row)
    result["headlines"] = fresh
    result["headline_freshness"] = {
        "max_age_hours": _REVALIDATE_AFTER_HOURS,
        "omitted_old_or_undated": len(headlines) - len(fresh),
        "note": "Publication age, not collection time. Missing news is unknown, not bearish.",
    }
    return result


def _candidate_review_plan(
    context: dict[str, Any],
    *,
    stocks: tuple[str, ...],
    held: frozenset[str],
    excluded: frozenset[str],
) -> dict[str, Any]:
    """Blocked mechanical picks must lead to alternative review, never an automatic buy."""
    ladder = context.get("candidate_ladder") or {}
    if not isinstance(ladder, dict):
        ladder = {}
    rows = ladder.get("ranked") or []
    allowed = set(stocks) - held - excluded
    ranked = list(
        dict.fromkeys(str(r.get("symbol", "")).upper().strip() for r in rows if isinstance(r, dict))
    )
    eligible = [s for s in ranked if s in allowed]
    selected = list(
        dict.fromkeys(str(s).upper().strip() for s in (ladder.get("selected_mechanical") or []))
    )
    blocked = {
        s: "operator exclusion"
        if s in excluded
        else "operator hold"
        if s in held
        else "not in PM stock universe"
        for s in selected
        if s not in allowed
    }
    mechanical = [s for s in selected if s in eligible]
    # Review several alternatives even when the PM rejects a permitted pick.
    alternatives = [s for s in eligible if s not in selected][: max(3, len(blocked))]
    return {
        "blocked_mechanical": blocked,
        "eligible_ranked": eligible,
        "mechanical_to_review": mechanical,
        "alternatives_to_review": alternatives,
        "required_symbols": mechanical + alternatives,
        "note": "Review in rank order; select only with a current thesis and risk budget. Cash needs an explicit comparison. These are review candidates, not replacement orders.",
    }


def _allocation_facts(
    prior_weights: dict[str, float], weights: dict[str, float], context: dict[str, Any]
) -> list[dict[str, Any]]:
    positions = context.get("positions")
    # Without an account snapshot, absence from positions is not evidence
    # that the broker holds zero shares (build_context also uses [] on failure).
    known = isinstance(positions, list) and bool(context.get("account"))
    quantities = {}
    if known and isinstance(positions, list):
        for row in positions:
            try:
                qty = float(row["qty"])
                if not isfinite(qty):
                    raise ValueError("non-finite quantity")
                quantities[str(row["symbol"]).upper()] = qty
            except (KeyError, TypeError, ValueError):
                known = False
                break
    facts = []
    for symbol in sorted(set(prior_weights) | set(weights)):
        before, after = prior_weights.get(symbol, 0.0), weights.get(symbol, 0.0)
        delta = after - before
        change = "unchanged" if abs(delta) < 0.0001 else "increase" if delta > 0 else "reduce"
        if before == 0 < after:
            change = "open"
        elif after == 0 < before:
            change = "exit"
        facts.append(
            {
                "symbol": symbol,
                "prior_sim_weight": round(before, 6),
                "target_weight": after,
                "change_pp": round(delta * 100, 3),
                "sim_change": change,
                "live_position_status": (
                    "unknown"
                    if not known
                    else "held_in_snapshot"
                    if quantities.get(symbol, 0) != 0
                    else "absent_in_snapshot"
                ),
            }
        )
    return facts


def _review_reasoning(
    response: dict[str, Any], plan: dict[str, Any], weights: dict[str, float]
) -> dict[str, Any]:
    """Expose omissions and contradictions; never invent the model's missing thesis."""
    reviews: dict[str, dict[str, Any]] = {}
    issues = []
    rows = response.get("candidate_reviews")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).upper().strip()
        if symbol in reviews:
            issues.append(f"duplicate review: {symbol}")
        reviews[symbol] = {**row, "symbol": symbol}
    required = list(dict.fromkeys([*plan["required_symbols"], *weights]))
    for symbol in required:
        row = reviews.get(symbol, {})
        expected = "allocate" if symbol in weights else "pass"
        reason = row.get("reason")
        evidence = row.get("evidence")
        if row.get("decision") != expected:
            issues.append(f"{symbol}: missing or inconsistent {expected} review")
        if not isinstance(reason, str) or not reason.strip():
            issues.append(f"{symbol}: missing reason")
        if not isinstance(evidence, list) or not any(
            isinstance(s, str) and s.strip() for s in evidence
        ):
            issues.append(f"{symbol}: missing evidence")
    cash_reason = response.get("cash_reason")
    if sum(weights.values()) < 0.9999 and (
        not isinstance(cash_reason, str) or not cash_reason.strip()
    ):
        issues.append("cash allocation has no explicit reason")
    return {
        "complete": not issues,
        "issues": issues,
        "required_symbols": required,
        "candidate_reviews": list(reviews.values()),
        "cash_reason": cash_reason if isinstance(cash_reason, str) else "",
        "scope": "Structural completeness only; evidence claims are not independently verified.",
    }


def _format_reasoning_details(result: dict[str, Any]) -> list[str]:
    """The same review omissions must be visible on the push digest and /pm."""
    review = result.get("reasoning_review") or {}
    lines = []
    if review.get("issues"):
        lines.append("⚠️ *Reasoning incomplete:* " + clip("; ".join(review["issues"]), 220))
    if result.get("requires_operator_review"):
        lines.append("⚠️ *Operator review required before execution eligibility.* Use `/pm review`.")
    positioning = result.get("positioning_shadow") or {}
    if positioning:
        coverage = positioning.get("coverage", {})
        lines.append(
            "*Positioning evidence:* "
            + ", ".join(f"{k}: {v}" for k, v in coverage.items())
            + "; experimental sizing only."
        )
        for change in positioning.get("adjustments", [])[:3]:
            lines.append(
                f"`{change['symbol']}` {change['source']} experiment {change['base']:.1%} → {change['shadow']:.1%}; live target unchanged."
            )
    freshness = result.get("evidence_freshness") or {}
    if freshness.get("historical_takes") and not freshness.get("current_agent_votes"):
        lines.append("⚠️ Committee views need revalidation; no current specialist votes.")
    for row in (result.get("allocation_facts") or [])[:4]:
        status = row["live_position_status"]
        live = (
            "absent in live snapshot"
            if status == "absent_in_snapshot"
            else "held in live snapshot"
            if status == "held_in_snapshot"
            else "live holding unknown"
        )
        lines.append(
            f"`{row['symbol']}` sim {row['prior_sim_weight']:.1%} → target {row['target_weight']:.1%}; {live}"
        )
    for row in review.get("candidate_reviews", []):
        if row.get("symbol") in (result.get("weights") or {}):
            lines.append(
                f"*{row['symbol']} thesis (model):* {clip(str(row.get('reason', '')), 140)}"
            )
            if sum("thesis (model)" in line for line in lines) >= 2:
                break
    if review.get("cash_reason"):
        lines.append("*Cash rationale (model):* " + clip(review["cash_reason"], 220))
    return lines


def _format_candidate_review(result: dict[str, Any]) -> str:
    review = result.get("reasoning_review")
    if not isinstance(review, dict):
        return "No structured candidate review in the last PM decision. It will be recorded on the next PM run."
    plan = result.get("candidate_review_plan") or {}
    lines = [f"*PM candidate review* · {result.get('ts', 'timestamp unknown')}"]
    if result.get("audit_id"):
        lines.append(f"Decision ID: `{result['audit_id']}`")
    if result.get("requires_operator_review"):
        lines.append(
            "Reasoning needs acknowledgement: `/pm accept <decision ID> <review reason>`. This is separate from order approval."
        )
    for symbol, reason in plan.get("blocked_mechanical", {}).items():
        lines.append(f"`{symbol}` blocked: {reason}")
    alternatives = plan.get("alternatives_to_review") or []
    lines.append("Alternatives requested: " + (", ".join(alternatives) or "none available"))
    lines.append(
        "*Review completeness:* " + ("complete" if review.get("complete") else "INCOMPLETE")
    )
    if review.get("issues"):
        lines.append(clip("; ".join(review["issues"]), 300))
    for row in review.get("candidate_reviews", [])[:12]:
        lines.append(
            f"`{row.get('symbol')}` {row.get('decision')}: {clip(str(row.get('reason', '')), 120)}"
        )
    if len(review.get("candidate_reviews", [])) > 12:
        lines.append("Showing first 12 reviews; full record retained in the decision journal.")
    lines.append("*Cash (model):* " + clip(review.get("cash_reason") or "not explained", 280))
    for row in review.get("claims", [])[:12]:
        refs = ", ".join(str(r) for r in (row.get("evidence_ids") or []))
        lines.append(
            f"Claim {row.get('id')}: {clip(str(row.get('statement', 'missing statement')), 140)} "
            f"[{row.get('assessment')}; sources: {clip(refs, 100)}; "
            f"numeric check: {'passed' if row.get('numeric_subclaim_verified') else 'not verified'}]"
        )
        if row.get("verification_needed"):
            lines.append("Verify: " + clip(str(row["verification_needed"]), 140))
    for row in review.get("sizing_reviews", [])[:12]:
        # Preserve malformed values as text: this view must also explain a
        # rejected response instead of crashing while formatting percentages.
        lines.append(
            f"`{row.get('symbol')}` sizing fractions: target={row.get('target_weight', '?')}, "
            f"downside={row.get('downside_fraction', '?')}, "
            f"portfolio loss={row.get('portfolio_loss_fraction', '?')}"
        )
        for field in (
            "downside_basis",
            "invalidation",
            "overlap",
            "why_not_smaller",
            "why_not_larger",
        ):
            lines.append(field.replace("_", " ") + ": " + clip(str(row.get(field, "missing")), 100))
    for row in review.get("positioning_assessments", [])[:18]:
        if isinstance(row, dict):
            lines.append(
                f"`{row.get('symbol')}` {row.get('source')}: {row.get('assessment')} / {row.get('effect')}; "
                + clip(str(row.get("reason", "")), 120)
            )
    cash = review.get("cash_plan") or {}
    if cash:
        lines.append(
            f"Cash fractions: safety floor={cash.get('safety_floor', '?')}, "
            f"discretionary={cash.get('discretionary_weight', '?')}; "
            f"review due: {cash.get('review_at', 'missing')}"
        )
        for field in ("trigger", "best_alternative", "reason"):
            lines.append(
                field.replace("_", " ") + ": " + clip(str(cash.get(field, "missing")), 160)
            )
    if any(len(review.get(k, [])) > 12 for k in ("claims", "sizing_reviews")):
        lines.append("Showing first 12 claims/sizes; full record retained in the decision archive.")
    lines.append(
        "Completeness checks structure, not investment merit or independent verification. This review does not execute orders."
    )
    return "\n".join(lines)
