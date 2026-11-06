"""Enforced evidence coverage and reduction-only research arms; no executable targets."""

from __future__ import annotations

from typing import Any


def check_positioning(
    response: dict[str, Any], symbols: set[str], catalog: dict[str, Any]
) -> list[str]:
    issues = []
    rows = response.get("positioning_assessments", [])
    indexed = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            issues.append("malformed positioning assessment")
            continue
        if not isinstance(row.get("symbol"), str) or not isinstance(row.get("source"), str):
            issues.append("malformed positioning symbol/source")
            continue
        key = (row.get("symbol"), row.get("source"))
        if key in indexed:
            issues.append("duplicate positioning assessment")
        indexed[key] = row
    sources = ("cot", "goldman_public", "gamma_estimate")
    for symbol in sorted(symbols):
        for source in sources:
            row = indexed.get((symbol, source), {})
            label = f"{symbol}/{source}"
            status = str(row.get("assessment"))
            refs = row.get("evidence_ids")
            if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
                refs = []
            applicable = {
                k
                for k, v in catalog.items()
                if v.get("positioning_source") == source
                and symbol in v.get("symbols", [])
                and not v.get("requires_revalidation")
            }
            macro = {
                k
                for k, v in catalog.items()
                if source in {"cot", "gamma_estimate"}
                and v.get("positioning_source") == source
                and "SPY" in v.get("symbols", [])
                and not v.get("requires_revalidation")
            }
            relevant = applicable | macro
            if status not in {"support", "conflict", "neutral", "unavailable", "not_applicable"}:
                issues.append(f"{label}: missing positioning assessment")
            elif status in {"support", "conflict", "neutral"}:
                if not refs or any(r not in relevant for r in refs):
                    issues.append(f"{label}: missing, stale or wrong-scope positioning evidence")
                if any(r not in applicable for r in refs) and (
                    row.get("scope") != "market_context"
                    or row.get("effect") not in ("none", "timing")
                ):
                    issues.append(
                        f"{label}: broad market evidence is timing context, not stock-specific sizing/ranking"
                    )
            elif relevant:
                issues.append(f"{label}: available applicable evidence was ignored")
            if str(row.get("effect")) not in {"none", "rank", "size", "timing"}:
                issues.append(f"{label}: missing positioning effect")
            if status in {"unavailable", "not_applicable"} and row.get("effect") != "none":
                issues.append(f"{label}: missing evidence cannot justify an adjustment")
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                issues.append(f"{label}: missing scope/uncertainty explanation")
    return issues


def shadow_positioning(weights: dict[str, float], context: dict[str, Any]) -> dict[str, Any]:
    """Predeclared hypotheses, capped at 10% relative reduction per source, never live."""
    observations = context.get("observations", [])
    arms = {}
    changes = []
    for source in ("cot", "gamma_estimate"):
        adjusted = dict(weights)
        for symbol, weight in weights.items():
            relevant = [o for o in observations if o["source"] == source and symbol in o["symbols"]]
            triggers = []
            for row in relevant:
                facts = row["facts"]
                # Explicit hypotheses: extreme long crowding / concentrated unsigned gamma.
                field, threshold = (
                    ("percentile", 0.9) if source == "cot" else ("largest_strike_share", 0.25)
                )
                value = facts.get(field)
                if value is not None and value >= threshold:
                    triggers.append(row["id"])
            if triggers:
                adjusted[symbol] = weight * 0.9
                changes.append(
                    {
                        "source": source,
                        "symbol": symbol,
                        "base": weight,
                        "shadow": adjusted[symbol],
                        "evidence_ids": triggers,
                    }
                )
        arms["experimental_positioning_" + source] = adjusted
    return {
        "version": 1,
        "mode": "shadow_only",
        "arms": arms,
        "adjustments": changes,
        "coverage": context.get("coverage", {}),
        "note": "Unvalidated crowding/concentration hypotheses, 10% relative reduction maximum per arm. Goldman narrative has no numeric overlay. Live targets unchanged.",
    }
