"""Check decision-changing claims and arithmetic; prose is never verified by a vote."""

from __future__ import annotations

import hashlib
from datetime import datetime
from math import isfinite
from typing import Any

from trading.agents.pm_reasoning import _age_hours, _review_reasoning


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError, OverflowError):
        return None
    return result if isfinite(result) else None


def _evidence_catalog(payload: dict[str, Any], now: datetime) -> dict[str, Any]:
    context = payload.get("today_context") or {}
    ladder = context.get("candidate_ladder") or {}
    catalog: dict[str, Any] = {}
    for row in ladder.get("ranked", []):
        symbol = str(row.get("symbol", "")).upper()
        # The date describes daily bars, not an intraday quote.
        stamp = str(ladder.get("as_of", ""))
        age = _age_hours(stamp + "T00:00:00+00:00" if len(stamp) == 10 else stamp, now)
        catalog[f"ladder:{symbol}"] = {
            "kind": "measured",
            "observed_at": stamp,
            "requires_revalidation": age is None
            or age > 96
            or bool(ladder.get("staleness_warning")),
            "independence_key": "market_prices",
            "facts": row,
        }
    for row in (context.get("headlines") or [])[:24]:
        title = str(row.get("title", ""))
        source = str(row.get("source", "unknown"))
        key = "news:" + hashlib.sha256((source + title).encode()).hexdigest()[:12]
        age = _age_hours(row.get("published"), now)
        catalog[key] = {
            "kind": "headline_only",
            "observed_at": row.get("published"),
            "requires_revalidation": age is None or age > 72,
            "independence_key": source.casefold(),
            "title": title,
            "note": "Headline observed; article claims are not independently verified.",
        }
    for i, take in enumerate((payload.get("agent_takes") or [])[:24]):
        catalog[f"take:{i}"] = {
            "kind": "opinion",
            "observed_at": take.get("ts"),
            "requires_revalidation": take.get("requires_revalidation", True),
            "independence_key": "committee_opinion",
            "agent": take.get("agent"),
            "text": take.get("take"),
        }
    catalog["sentinel"] = {
        "kind": "constraint",
        "facts": payload.get("sentinel_alert"),
        "requires_revalidation": False,
    }
    catalog["operator_constraints"] = {
        "kind": "constraint",
        "requires_revalidation": False,
        "facts": {
            "held": payload.get("operator_held_do_not_trade"),
            "excluded": payload.get("operator_excluded_never_buy"),
        },
    }
    for observation in (context.get("positioning") or {}).get("observations", []):
        catalog[observation["id"]] = {
            **observation,
            "positioning_source": observation["source"],
            "requires_revalidation": not (
                datetime.fromisoformat(observation["available_at"])
                <= now
                < datetime.fromisoformat(observation["expires_at"])
            ),
            "independence_key": observation["source"],
        }
    return catalog


def _sentinel_scope(state: dict[str, Any], held: frozenset[str]) -> dict[str, Any]:
    triggers = state.get("triggers") or []
    symbols = {str(t).split()[0].upper() for t in triggers if str(t).split()}
    market = bool(symbols & {"SPY", "VIX", "^VIX"})
    pinned_only = bool(symbols) and symbols <= held
    return {
        "scope": "market" if market else "pinned_only" if pinned_only else "managed_or_unknown",
        "symbols": sorted(symbols),
        "live_cap_unchanged": 0.70 if state.get("fired_within_24h") else 1.0,
        "experimental_market_only_cap": 1.0
        if pinned_only
        else 0.70
        if state.get("fired_within_24h")
        else 1.0,
        "experiment_only": True,
    }


def _check_contract(
    response: dict[str, Any],
    plan: dict[str, Any],
    weights: dict[str, float],
    catalog: dict[str, Any],
    now: datetime,
    sentinel_cap: float | None,
) -> dict[str, Any]:
    review = _review_reasoning(response, plan, weights)
    issues = list(review["issues"])
    claims: dict[str, Any] = {}
    checked = []
    rows = response.get("claims")
    for claim in rows if isinstance(rows, list) else []:
        if not isinstance(claim, dict):
            issues.append("malformed claim")
            continue
        cid = str(claim.get("id", ""))
        if not cid or cid in claims:
            issues.append("missing or duplicate claim ID")
        claims[cid] = claim
        refs = claim.get("evidence_ids")
        refs = refs if isinstance(refs, list) and all(isinstance(r, str) for r in refs) else []
        sources = [catalog[r] for r in refs if r in catalog]
        if not refs or len(sources) != len(refs):
            issues.append(f"{cid}: missing or unknown evidence IDs")
        assessment = claim.get("assessment")
        verified = False
        fact = claim.get("numeric_check")
        if isinstance(fact, dict):
            source = catalog.get(str(fact.get("evidence_id")), {})
            measured = _number((source.get("facts") or {}).get(str(fact.get("field"))))
            threshold = _number(fact.get("value"))
            if (
                source.get("kind") == "measured"
                and not source.get("requires_revalidation")
                and measured is not None
                and threshold is not None
                and fact.get("evidence_id") in refs
            ):
                op = fact.get("operator")
                verified = bool(
                    (op == "le" and measured <= threshold)
                    or (op == "ge" and measured >= threshold)
                    or (op == "eq" and abs(measured - threshold) <= 1e-6)
                )
        if assessment == "measured":
            if not verified:
                issues.append(f"{cid}: claimed measurement is unsupported or contradicted")
        elif assessment == "uncertain":
            if not str(claim.get("verification_needed") or "").strip():
                issues.append(f"{cid}: uncertainty needs a verification step")
        elif assessment == "constraint":
            if not sources or any(s.get("kind") != "constraint" for s in sources):
                issues.append(f"{cid}: constraint claim cites non-constraint evidence")
        else:
            issues.append(f"{cid}: use measured, uncertain or constraint assessment")
        if assessment != "uncertain" and any(s.get("requires_revalidation") for s in sources):
            issues.append(f"{cid}: stale evidence presented as current")
        checked.append(
            {
                **claim,
                "numeric_subclaim_verified": verified,
                # Opinions and headlines never become independent confirmation.
                "independent_measured_sources": len(
                    {s["independence_key"] for s in sources if s.get("kind") == "measured"}
                ),
                "free_text_claim_verified": False,
            }
        )
    for row in review["candidate_reviews"]:
        ids = row.get("claim_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) or i not in claims for i in ids)
        ):
            issues.append(f"{row['symbol']}: decision reason lacks linked claims")
    sizing = response.get("sizing_reviews")
    sized: dict[str, Any] = {}
    for row in sizing if isinstance(sizing, list) else []:
        if isinstance(row, dict):
            symbol = str(row.get("symbol", "")).upper()
            if symbol in sized:
                issues.append(f"{symbol}: duplicate sizing review")
            sized[symbol] = row
    for symbol, weight in weights.items():
        row = sized.get(symbol, {})
        downside = _number(row.get("downside_fraction"))
        loss = _number(row.get("portfolio_loss_fraction"))
        reported_weight = _number(row.get("target_weight"))
        if reported_weight is None or abs(reported_weight - weight) > 1e-6:
            issues.append(f"{symbol}: sizing does not match final target")
        if (
            downside is None
            or not 0 < downside <= 1
            or loss is None
            or abs(loss - weight * downside) > 1e-6
        ):
            issues.append(f"{symbol}: invalid downside/loss-budget arithmetic")
        for field in (
            "invalidation",
            "downside_basis",
            "overlap",
            "why_not_smaller",
            "why_not_larger",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                issues.append(f"{symbol}: missing sizing {field}")
    cash = max(0.0, 1.0 - sum(weights.values()))
    cash_plan = response.get("cash_plan")
    cash_plan = cash_plan if isinstance(cash_plan, dict) else {}
    if cash > 0.0001:
        required_floor = 1 - sentinel_cap if sentinel_cap is not None else 0.0
        floor = _number(cash_plan.get("safety_floor"))
        discretionary = _number(cash_plan.get("discretionary_weight"))
        if (
            floor is None
            or abs(floor - required_floor) > 1e-6
            or discretionary is None
            or abs(discretionary - max(0, cash - required_floor)) > 1e-6
        ):
            issues.append("cash plan does not reconcile safety floor and discretionary cash")
        try:
            due = datetime.fromisoformat(str(cash_plan.get("review_at")))
            hours = (due - now).total_seconds() / 3600
            if due.tzinfo is None or not 0 < hours <= 168:
                raise ValueError("outside review window")
        except (TypeError, ValueError):
            issues.append("cash plan needs an aware review_at within seven days")
        for field in ("trigger", "best_alternative", "reason"):
            if not isinstance(cash_plan.get(field), str) or not cash_plan[field].strip():
                issues.append(f"cash plan missing {field}")
    from trading.agents.positioning import check_positioning

    issues.extend(
        check_positioning(
            response, {r["symbol"] for r in review["candidate_reviews"]} | set(weights), catalog
        )
    )
    return {
        **review,
        "positioning_assessments": response.get("positioning_assessments", []),
        "complete": not issues,
        "issues": issues,
        "claims": checked,
        "sizing_reviews": list(sized.values()),
        "cash_plan": cash_plan,
        "scope": "Citations, numeric subclaims, sizing and cash arithmetic checked; model event claims and downside estimates are not independently verified.",
    }
