"""Exact-decision acknowledgement is independent of risk and exception approval."""

import copy
import json
import re
from datetime import timedelta

import pytest

from trading.agents.pm import run_agent_pm
from trading.agents.pm_audit import (
    _accept_review,
    _audit_dir,
    _read_record,
    _verify_decision,
    _write_record,
)
from trading.agents.pm_contract import _check_contract, _evidence_catalog, _sentinel_scope
from trading.agents.pm_signal import load_pm_signal
from trading.core.clock import UtcClock
from trading.memory import MemoryStore


def valid_response(now, weight=0.05):
    return {
        "positioning_assessments": [
            {
                "symbol": "AAA",
                "source": source,
                "assessment": "unavailable",
                "evidence_ids": [],
                "effect": "none",
                "reason": "No applicable positioning data supplied.",
            }
            for source in ("cot", "goldman_public", "gamma_estimate")
        ],
        "target_weights": {"AAA": weight},
        "rationale": "Small research allocation.",
        "candidate_reviews": [
            {
                "symbol": "AAA",
                "decision": "allocate",
                "reason": "Positive ranked score with uncertain event outlook",
                "evidence": ["ladder:AAA"],
                "claim_ids": ["c1"],
            }
        ],
        "claims": [
            {
                "id": "c1",
                "statement": "ranked score is nonnegative",
                "assessment": "measured",
                "evidence_ids": ["ladder:AAA"],
                "numeric_check": {
                    "evidence_id": "ladder:AAA",
                    "field": "score",
                    "operator": "ge",
                    "value": 0,
                },
            }
        ],
        "sizing_reviews": [
            {
                "symbol": "AAA",
                "target_weight": weight,
                "downside_fraction": 0.2,
                "portfolio_loss_fraction": weight * 0.2,
                "invalidation": "breaks entry thesis",
                "downside_basis": "scenario estimate, not a stop guarantee",
                "overlap": "one small sector allocation",
                "why_not_smaller": "bounded research budget",
                "why_not_larger": "event uncertainty",
            }
        ],
        "cash_reason": "Remaining alternatives do not justify exposure now.",
        "cash_plan": {
            "safety_floor": 0,
            "discretionary_weight": 1 - weight,
            "review_at": (now + timedelta(hours=1)).isoformat(),
            "trigger": "new verified event evidence",
            "best_alternative": "AAA larger allocation",
            "reason": "event uncertainty outweighs expected benefit",
        },
    }


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr("trading.agents.pm._stock_universe", lambda: ("AAA",))
    mem = MemoryStore(tmp_path / "memory")
    now = UtcClock().now()
    context = {
        "account": {"equity": 100000, "base_currency": "USD"},
        "positions": [],
        "candidate_ladder": {
            "as_of": now.date().isoformat(),
            "ranked": [{"symbol": "AAA", "rank": 1, "score": 0.5}],
            "selected_mechanical": ["AAA"],
        },
    }

    def run(llm):
        return run_agent_pm(context, mem, tmp_path, prices={"AAA": 100}, llm=llm)

    yield run, now, context
    mem.close()


def test_complete_decision_archives_exact_inputs_and_does_not_retry(desk, tmp_path):
    run, now, _ = desk
    calls = []

    def llm(system, prompt):
        calls.append((system, prompt))
        return valid_response(now)

    result = run(llm)
    assert len(calls) == 1
    assert not result["requires_operator_review"]
    directory, _ = _verify_decision(tmp_path, result)
    request, _ = _read_record(directory, "request.json")
    assert request["system"] == calls[0][0] and request["prompt"] == calls[0][1]
    assert request["observed_prices"] == {"AAA": 100}
    assert request["model"]["provider"] == "injected"
    assert "pm.py" in request["source_hashes"]
    assert load_pm_signal(tmp_path, sleeve_pct=1).ok
    with pytest.raises(FileExistsError):
        _write_record(directory, "request.json", {})


def test_following_documented_response_fields_needs_no_repair(desk):
    """Nov 9: the JSON example omitted mandatory positioning and claim links."""
    run, now, _ = desk
    calls = []

    def llm(system, prompt):
        calls.append(prompt)
        example = system.split("Respond ONLY with JSON: ", 1)[1].split("}. ", 1)[0]
        fields = set(re.findall(r'"([a-z_]+)":', example))
        response = {k: v for k, v in valid_response(now).items() if k in fields}
        response["candidate_reviews"] = [
            {k: v for k, v in row.items() if k in fields} for row in response["candidate_reviews"]
        ]
        return response

    result = run(llm)
    assert not result["requires_operator_review"], result["reasoning_review"]["issues"]
    assert len(calls) == 1


def test_misclassified_market_evidence_stays_blocked_and_explains_why(desk, tmp_path):
    """A coverage caveat is not an operator constraint, even after repair."""
    run, now, _ = desk
    response = valid_response(now)
    response["claims"].append(
        {
            "id": "c10",
            "statement": "Market observations cannot establish stock accumulation.",
            "assessment": "constraint",
            "evidence_ids": ["ladder:AAA"],
            "verification_needed": "Obtain a stock-level source.",
        }
    )
    result = run(lambda *_: copy.deepcopy(response))
    issue = "c10: constraint claim cites non-constraint evidence"
    assert result["reasoning_review"]["issues"] == [issue]
    loaded = load_pm_signal(tmp_path, sleeve_pct=1)
    assert not loaded.ok
    assert issue in loaded.reason
    assert "/pm review" in loaded.reason and result["audit_id"] in loaded.reason


def test_failed_model_call_replaces_previously_eligible_decision(desk, tmp_path):
    run, now, _ = desk
    first = run(lambda *_: valid_response(now))
    assert load_pm_signal(tmp_path, sleeve_pct=1).ok

    def failed(*_):
        raise RuntimeError("transport unavailable")

    result = run(failed)
    assert not result["ok"] and result["audit_id"] != first["audit_id"]
    directory, _ = _verify_decision(tmp_path, result)
    assert _read_record(directory, "attempt-1.json")[0]["error_type"] == "RuntimeError"
    assert not load_pm_signal(tmp_path, sleeve_pct=1).ok


def test_single_repair_can_fix_review_but_cannot_change_allocation(desk, tmp_path):
    run, now, _ = desk
    calls = []

    def llm(_system, prompt):
        calls.append(json.loads(prompt))
        if len(calls) == 1:
            return {"target_weights": {"AAA": 0.05}}
        response = valid_response(now)
        response["target_weights"] = {"AAA": 0.9}  # Must never be executed.
        return response

    result = run(llm)
    assert len(calls) == 2
    assert calls[1]["reasoning_repair"]["locked_final_weights"] == {"AAA": 0.05}
    assert result["weights"] == {"AAA": 0.05}
    assert not result["requires_operator_review"]
    directory, _ = _verify_decision(tmp_path, result)
    assert _read_record(directory, "attempt-2.json")[0]["response"]["target_weights"] == {
        "AAA": 0.9
    }


def test_unrepaired_decision_requires_exact_receipt_and_next_decision_does_not_inherit(
    desk, tmp_path
):
    run, _, _ = desk
    calls = []

    def llm(*_):
        calls.append(1)
        return {"target_weights": {"AAA": 0.05}}

    first = run(llm)
    assert len(calls) == 2 and first["requires_operator_review"]
    assert not load_pm_signal(tmp_path, sleeve_pct=1).ok
    assert "No order sent" in _accept_review(
        tmp_path, first["audit_id"], "Reviewed event uncertainty and sizing manually"
    )
    assert load_pm_signal(tmp_path, sleeve_pct=1).ok
    second = run(llm)
    assert second["audit_id"] != first["audit_id"]
    assert not load_pm_signal(tmp_path, sleeve_pct=1).ok
    assert "does not match" in _accept_review(
        tmp_path, first["audit_id"], "Reviewed the old decision again"
    )


@pytest.mark.parametrize("file", ["request.json", "attempt-1.json", "decision.json"])
def test_tampered_audit_cannot_be_accepted_or_traded(desk, tmp_path, file):
    run, now, _ = desk
    result = run(lambda *_: valid_response(now))
    path = _audit_dir(tmp_path, result["audit_id"]) / file
    envelope = json.loads(path.read_text())
    envelope["payload"]["tampered"] = True
    path.write_text(json.dumps(envelope))
    assert not load_pm_signal(tmp_path, sleeve_pct=1).ok
    assert "Cannot verify" in _accept_review(
        tmp_path, result["audit_id"], "Reviewed but audit no longer matches"
    )


def test_cash_review_deadline_remains_binding_after_acknowledgement(desk, tmp_path):
    run, now, _ = desk
    result = run(lambda *_: valid_response(now))
    assert (
        load_pm_signal(tmp_path, sleeve_pct=1, now=now + timedelta(hours=2)).reason
        == "PM cash thesis is due for reassessment; request /pm run"
    )
    assert result["reasoning_review"]["cash_plan"]["review_at"]


def test_removing_new_decision_version_does_not_fall_back_to_legacy(desk, tmp_path):
    run, now, _ = desk
    result = run(lambda *_: valid_response(now))
    del result["reasoning_contract_version"]
    (tmp_path / "agent_pm" / "last_run.json").write_text(json.dumps(result))
    loaded = load_pm_signal(tmp_path, sleeve_pct=1)
    assert not loaded.ok and "version is missing" in loaded.reason


def test_repair_exception_is_recorded_and_does_not_erase_original_intent(desk, tmp_path):
    run, _, _ = desk
    calls = []

    def llm(*_):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("transport failure")
        return {"target_weights": {"AAA": 0.05}}

    result = run(llm)
    assert result["weights"] == {"AAA": 0.05} and result["requires_operator_review"]
    assert (
        _read_record(_audit_dir(tmp_path, result["audit_id"]), "attempt-2.json")[0]["error_type"]
        == "RuntimeError"
    )


def test_numeric_claim_is_checked_and_repeated_opinions_cannot_verify_it():
    now = UtcClock().now()
    payload = {
        "today_context": {
            "candidate_ladder": {
                "as_of": now.date().isoformat(),
                "ranked": [{"symbol": "AAA", "score": 0.5}],
            }
        },
        "agent_takes": [
            {"agent": "a", "ts": now.isoformat(), "take": "rumor", "requires_revalidation": False},
            {
                "agent": "b",
                "ts": now.isoformat(),
                "take": "same rumor",
                "requires_revalidation": False,
            },
        ],
    }
    catalog = _evidence_catalog(payload, now)
    response = valid_response(now)
    okay = _check_contract(
        response, {"required_symbols": ["AAA"]}, {"AAA": 0.05}, catalog, now, None
    )
    assert okay["complete"] and okay["claims"][0]["numeric_subclaim_verified"]
    for refs in (["take:0", "take:1"], ["invented"]):
        bad = copy.deepcopy(response)
        bad["claims"][0]["evidence_ids"] = refs
        review = _check_contract(
            bad, {"required_symbols": ["AAA"]}, {"AAA": 0.05}, catalog, now, None
        )
        assert not review["complete"]
        assert review["claims"][0]["independent_measured_sources"] == 0


def test_uncertain_claim_needs_a_verification_step_and_downside_math_must_match():
    now = UtcClock().now()
    response = valid_response(now)
    response["claims"] = [
        {
            "id": "c1",
            "assessment": "uncertain",
            "evidence_ids": ["news:1"],
            "verification_needed": "check issuer filing",
        }
    ]
    catalog = {"news:1": {"kind": "headline_only", "requires_revalidation": False}}
    review = _check_contract(
        response, {"required_symbols": ["AAA"]}, {"AAA": 0.05}, catalog, now, None
    )
    assert review["complete"] and not review["claims"][0]["free_text_claim_verified"]
    response["sizing_reviews"][0]["portfolio_loss_fraction"] = 0.1
    assert not _check_contract(
        response, {"required_symbols": ["AAA"]}, {"AAA": 0.05}, catalog, now, None
    )["complete"]


def test_sentinel_scope_is_experimental_and_does_not_lift_live_cap():
    pinned = _sentinel_scope(
        {"fired_within_24h": True, "triggers": ["QNT -6.7% (held)"]}, frozenset({"QNT"})
    )
    assert pinned["scope"] == "pinned_only"
    assert pinned["live_cap_unchanged"] == 0.7 and pinned["experimental_market_only_cap"] == 1
    market = _sentinel_scope(
        {"fired_within_24h": True, "triggers": ["SPY -3%", "QNT -7%"]}, frozenset({"QNT"})
    )
    assert market["scope"] == "market" and market["experimental_market_only_cap"] == 0.7


@pytest.mark.parametrize(
    "gross_cap,sentinel_cap,expected_floor", [(0.71, None, 0.29), (0.71, 0.7, 0.3), (0.5, 0.7, 0.5)]
)
def test_cash_review_uses_resolved_gross_limit(gross_cap, sentinel_cap, expected_floor):
    now = UtcClock().now()
    payload = {
        "today_context": {
            "candidate_ladder": {
                "as_of": now.date().isoformat(),
                "ranked": [{"symbol": "AAA", "score": 0.5}],
                "shadow_risk_limits": {"max_gross_exposure": gross_cap},
            }
        }
    }
    catalog = _evidence_catalog(payload, now)
    response = valid_response(now, 0.14)
    response["cash_plan"].update(
        safety_floor=expected_floor, discretionary_weight=0.86 - expected_floor
    )
    review = _check_contract(
        response, {"required_symbols": ["AAA"]}, {"AAA": 0.14}, catalog, now, sentinel_cap
    )
    assert not any("cash plan" in issue for issue in review["issues"])
    response["cash_plan"]["safety_floor"] = 0
    assert any(
        "cash plan does not reconcile" in issue
        for issue in _check_contract(
            response, {"required_symbols": ["AAA"]}, {"AAA": 0.14}, catalog, now, sentinel_cap
        )["issues"]
    )


@pytest.mark.parametrize(
    "target",
    ["trading.agents.positioning.shadow_positioning", "trading.reporting.pm_shadow._shadow_plan"],
)
def test_a_failing_research_arm_cannot_cost_the_decision(desk, tmp_path, monkeypatch, target):
    """2026-11-10. The shadow arms run after the simulated book has been
    rebalanced and saved, and before the decision is written. Unguarded, a
    surprise in a positioning observation raised out of the run: the book
    had moved and there was no decision to trade or to audit."""
    run, now, _ = desk

    def boom(*_args, **_kwargs):
        raise KeyError("facts")

    monkeypatch.setattr(target, boom)
    result = run(lambda system, prompt: valid_response(now))
    assert result["ok"] and result["weights"] == {"AAA": 0.05}
    saved = json.loads((tmp_path / "agent_pm" / "last_run.json").read_text())
    assert saved["ok"] and saved["audit_id"] == result["audit_id"]
    _verify_decision(tmp_path, saved)
    assert load_pm_signal(tmp_path, sleeve_pct=1).ok


def test_a_malformed_cash_weight_is_a_refusal_not_a_crash(desk, tmp_path):
    """/pm calls load_pm_signal outside the cycle's catch-all."""
    run, now, _ = desk
    run(lambda system, prompt: valid_response(now))
    path = tmp_path / "agent_pm" / "last_run.json"
    raw = json.loads(path.read_text())
    raw["cash_weight"] = "most of it"
    path.write_text(json.dumps(raw))
    result = load_pm_signal(tmp_path, sleeve_pct=1)
    assert not result.ok and result.reason


# ---------------------------------------------------------------------------
# 2026-11-10 — the reasoning review warns where a human approves the basket
# ---------------------------------------------------------------------------


def _desk_settings(monkeypatch, *, approval: bool, strict: bool = False) -> None:
    from trading.core import config

    monkeypatch.setattr(
        config,
        "settings",
        config.settings.model_copy(
            update={"require_cycle_approval": approval, "agent_pm_reasoning_review_blocks": strict}
        ),
    )


def _incomplete(run):
    result = run(lambda *_: {"target_weights": {"AAA": 0.05}})
    assert result["requires_operator_review"]
    return result


def test_open_review_rides_on_the_approval_card_instead_of_stopping_the_desk(
    desk, tmp_path, monkeypatch
):
    """Paperwork failed the contract twice in four days and, with the PM as
    the whole desk, five cycles submitted nothing. The operator approves
    every basket anyway: tell them there, with the basket in front of them."""
    from trading.agents.pm_signal import format_bridge_note

    run, _, _ = desk
    first = _incomplete(run)
    _desk_settings(monkeypatch, approval=True)

    result = load_pm_signal(tmp_path, sleeve_pct=1)
    assert result.ok
    assert "reasoning review incomplete" in result.review_warning
    assert "issue(s)" in result.review_warning
    note = format_bridge_note(result)
    assert "reasoning review incomplete" in note and "/pm review" in note

    # Acknowledging the exact decision clears the warning; nothing else does.
    _accept_review(tmp_path, first["audit_id"], "Reviewed sizing and evidence manually")
    cleared = load_pm_signal(tmp_path, sleeve_pct=1)
    assert cleared.ok and cleared.review_warning == ""
    assert "reasoning review" not in format_bridge_note(cleared)


def test_a_complete_review_carries_no_warning(desk, tmp_path, monkeypatch):
    run, now, _ = desk
    run(lambda *_: valid_response(now))
    _desk_settings(monkeypatch, approval=True)
    result = load_pm_signal(tmp_path, sleeve_pct=1)
    assert result.ok and result.review_warning == ""


def test_without_basket_approval_an_open_review_still_withholds(desk, tmp_path, monkeypatch):
    """The warning is only safe because a human reads it before orders go.
    With REQUIRE_CYCLE_APPROVAL off nobody does."""
    run, _, _ = desk
    _incomplete(run)
    _desk_settings(monkeypatch, approval=False)
    result = load_pm_signal(tmp_path, sleeve_pct=1)
    assert not result.ok and "/pm accept" in result.reason


def test_the_strict_gate_can_be_switched_back_on(desk, tmp_path, monkeypatch):
    run, _, _ = desk
    _incomplete(run)
    _desk_settings(monkeypatch, approval=True, strict=True)
    result = load_pm_signal(tmp_path, sleeve_pct=1)
    assert not result.ok and "/pm accept" in result.reason


@pytest.mark.parametrize("file", ["request.json", "attempt-1.json", "decision.json"])
def test_warning_mode_never_relaxes_audit_integrity(desk, tmp_path, monkeypatch, file):
    run, _, _ = desk
    result = _incomplete(run)
    _desk_settings(monkeypatch, approval=True)
    path = _audit_dir(tmp_path, result["audit_id"]) / file
    envelope = json.loads(path.read_text())
    envelope["payload"]["tampered"] = True
    path.write_text(json.dumps(envelope))
    refused = load_pm_signal(tmp_path, sleeve_pct=1)
    assert not refused.ok and "verification failed" in refused.reason


def test_a_settings_failure_fails_closed(monkeypatch):
    from trading.agents.pm_audit import reasoning_review_blocks
    from trading.core import config

    monkeypatch.setattr(config, "settings", object())
    assert reasoning_review_blocks() is True


def test_the_digest_says_which_of_the_two_it_is(desk, monkeypatch):
    from trading.agents.pm import format_pm_digest

    run, _, _ = desk
    result = _incomplete(run)
    _desk_settings(monkeypatch, approval=True)
    card = format_pm_digest(result, None)
    assert "BLOCKED" not in card and "flagged on the approval card" in card
    _desk_settings(monkeypatch, approval=False)
    assert "BLOCKED pending reasoning review" in format_pm_digest(result, None)
