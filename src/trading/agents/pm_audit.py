"""Append-only, content-addressed PM inputs, answers and exact-decision review receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

from trading.core.clock import UtcClock
from trading.core.file_lock import file_lock
from trading.core.text import clip


def _encoded(value: object) -> bytes:
    def safe(item: object) -> object:
        if isinstance(item, float) and not isfinite(item):
            return {"nonfinite_number": repr(item)}
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        return item

    return json.dumps(
        safe(value), sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    ).encode()


def _write_record(directory: Path, name: str, payload: dict[str, Any]) -> str:
    body = _encoded(payload)
    digest = hashlib.sha256(body).hexdigest()
    with (directory / name).open("xb") as f:
        f.write(_encoded({"sha256": digest, "payload": json.loads(body)}))
        f.flush()
        os.fsync(f.fileno())
    return digest


def _read_record(directory: Path, name: str) -> tuple[dict[str, Any], str]:
    envelope = json.loads((directory / name).read_text())
    payload = envelope["payload"]
    digest = hashlib.sha256(_encoded(payload)).hexdigest()
    if digest != envelope["sha256"] or not isinstance(payload, dict):
        raise ValueError("PM audit content hash mismatch")
    return payload, digest


def _audit_dir(state_dir: Path, audit_id: str) -> Path:
    if not re.fullmatch(r"pm-[0-9a-f]{32}", audit_id):
        raise ValueError("invalid PM decision ID")
    return state_dir / "agent_pm" / "decisions" / audit_id


def _start_audit(state_dir: Path, request: dict[str, Any]) -> tuple[str, str]:
    audit_id = "pm-" + uuid4().hex
    directory = _audit_dir(state_dir, audit_id)
    directory.mkdir(parents=True, exist_ok=False)
    digest = _write_record(directory, "request.json", request)
    return audit_id, digest


def _model_metadata(injected: bool) -> dict[str, str]:
    if injected:
        return {"provider": "injected", "model": "injected callable"}
    from trading.agents import llm

    anthropic = bool(llm._anthropic_key())
    return {
        "provider": "anthropic" if anthropic else "openai",
        "model": llm._resolve_anthropic_model("frontier")
        if anthropic
        else llm._resolve_openai_model("frontier"),
        "effort": llm._anthropic_effort("frontier") if anthropic else "provider default",
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    paths = [
        root / n
        for n in (
            "pm.py",
            "pm_audit.py",
            "pm_reasoning.py",
            "pm_contract.py",
            "pm_shadow.py",
            "llm.py",
        )
    ]
    paths.append(root.parent / "reporting" / "pm_shadow.py")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    for relative in (
        "agents/positioning.py",
        "data/positioning.py",
        "runtime/positioning_monitor.py",
        "agents/context.py",
        "core/types.py",
    ):
        path = root.parent / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _archive_decision(state_dir: Path, result: dict[str, Any]) -> str:
    directory = _audit_dir(state_dir, result["audit_id"])
    # Link each attempt, including the exact repair prompt, to the decision.
    attempts = {}
    for name in ("attempt-1.json", "attempt-2.json"):
        if (directory / name).exists():
            _, attempts[name] = _read_record(directory, name)
    result["audit_attempt_hashes"] = attempts
    return _write_record(directory, "decision.json", result)


def _verify_decision(state_dir: Path, raw: dict[str, Any]) -> tuple[Path, str]:
    directory = _audit_dir(state_dir, str(raw.get("audit_id", "")))
    saved, digest = _read_record(directory, "decision.json")
    current = {k: v for k, v in raw.items() if k != "audit_sha256"}
    if digest != raw.get("audit_sha256") or _encoded(saved) != _encoded(current):
        raise ValueError("PM decision differs from its archived content")
    _, request_hash = _read_record(directory, "request.json")
    if request_hash != saved.get("audit_request_sha256"):
        raise ValueError("PM request hash mismatch")
    for name, expected in saved.get("audit_attempt_hashes", {}).items():
        if name not in {"attempt-1.json", "attempt-2.json"}:
            raise ValueError("invalid attempt name")
        if _read_record(directory, name)[1] != expected:
            raise ValueError("PM attempt hash mismatch")
    return directory, digest


def reasoning_review_blocks() -> bool:
    """Does an unacknowledged reasoning review withhold the PM's targets?

    Why this is a setting and not a constant (2026-11-10 review). From
    4 November every decision whose paperwork failed the contract — sizing
    arithmetic to six decimals, three positioning rows per reviewed stock,
    typed claims — was withheld until the operator typed ``/pm accept`` with
    a 35-character ID. The contract failed on the 6th and again on the 9th,
    the PM sleeve is the whole desk, and the incident log records five
    consecutive cycles with zero orders. Nothing unsafe was being stopped:
    every one of those baskets still had to pass the risk manager and the
    operator's own /approve.

    So the review blocks only where it is the last human check:

    * ``REQUIRE_CYCLE_APPROVAL`` off — no basket approval exists, so an
      unreviewed decision must not trade itself. Blocks.
    * ``AGENT_PM_REASONING_REVIEW_BLOCKS=true`` — the operator asked for
      the strict gate back. Blocks.
    * otherwise — the issues are printed on the approval card and the
      operator decides there, once, with the basket in front of them.

    Audit integrity (hash, version, archive) is not part of this choice and
    always blocks. A settings failure fails closed.
    """
    try:
        from trading.core.config import settings

        if not bool(settings.require_cycle_approval):
            return True
        return bool(settings.agent_pm_reasoning_review_blocks)
    except Exception:
        return True


def _integrity_block_reason(state_dir: Path, raw: dict[str, Any]) -> str | None:
    """A decision that cannot be tied to its archive never trades."""
    # Legacy decisions retain existing bridge semantics. New producers always
    # write the version; a malformed new audit never falls back to legacy.
    if not any(k in raw for k in ("reasoning_contract_version", "audit_id", "audit_sha256")):
        return None
    if raw.get("reasoning_contract_version") != 1:
        return "PM audit version is missing or unsupported; request /pm run to create a new verifiable decision."
    try:
        _verify_decision(state_dir, raw)
    except (OSError, ValueError, KeyError, TypeError):
        return "PM audit verification failed; request /pm run to create a new verifiable decision."
    return None


def _review_issues(raw: dict[str, Any]) -> list[str]:
    review = raw.get("reasoning_review")
    issues = review.get("issues") if isinstance(review, dict) else None
    return [str(issue) for issue in issues] if isinstance(issues, list) else []


def _outstanding_review(state_dir: Path, raw: dict[str, Any]) -> bool:
    """True while a verified decision's reasoning review is unacknowledged."""
    if not raw.get("requires_operator_review"):
        return False
    try:
        directory, digest = _verify_decision(state_dir, raw)
        receipt, _ = _read_record(directory, "operator-review.json")
        if receipt.get("decision_sha256") == digest and receipt.get("audit_id") == raw["audit_id"]:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return True


def _review_block_reason(state_dir: Path, raw: dict[str, Any]) -> str | None:
    """The strict gate: integrity failure or an unacknowledged review."""
    integrity = _integrity_block_reason(state_dir, raw)
    if integrity:
        return integrity
    if not any(k in raw for k in ("reasoning_contract_version", "audit_id", "audit_sha256")):
        return None
    if not _outstanding_review(state_dir, raw):
        return None
    # Nov 9: a completed cycle looked stuck because its card hid the actual
    # failed claim behind a generic request to acknowledge the decision.
    issues = _review_issues(raw)
    detail = " Validation issues: " + clip("; ".join(issues), 240) + "." if issues else ""
    return (
        f"PM evidence/reasoning requires exact-decision review: /pm review then /pm accept "
        f"{raw.get('audit_id', 'unknown')} <reason>. Risk and order approval still apply.{detail}"
    )


def _review_warning(state_dir: Path, raw: dict[str, Any]) -> str:
    """The approval-card line for a decision trading with an open review.

    Names the count and the first failures, not the whole list: a card that
    prints twenty paperwork lines every cycle teaches its reader to skip
    the one that mattered. ``/pm review`` has the rest.
    """
    if not any(k in raw for k in ("reasoning_contract_version", "audit_id", "audit_sha256")):
        return ""
    if not _outstanding_review(state_dir, raw):
        return ""
    issues = _review_issues(raw)
    count = f"{len(issues)} issue(s)" if issues else "issues not recorded"
    shown = ": " + clip("; ".join(issues[:2]), 200) if issues else ""
    more = f" (+{len(issues) - 2} more)" if len(issues) > 2 else ""
    return (
        f"⚠️ *PM reasoning review incomplete* — {count}{shown}{more}. "
        "The targets below are the PM's; its paperwork did not pass the contract and "
        "has not been acknowledged. `/pm review` shows all of it. "
        "Approving this basket is your decision on it."
    )


def _accept_review(state_dir: Path, audit_id: str, reason: str) -> str:
    if len(reason.strip()) < 12:
        return "Include a meaningful review reason (at least 12 characters). No approval recorded."
    path = state_dir / "agent_pm" / "last_run.json"
    with file_lock(path):
        try:
            raw = json.loads(path.read_text())
            if raw.get("audit_id") != audit_id:
                return "Decision changed or ID does not match. Read /pm review again."
            directory, digest = _verify_decision(state_dir, raw)
            if not raw.get("requires_operator_review"):
                return "This decision has no outstanding reasoning review."
            if (directory / "operator-review.json").exists():
                receipt, _ = _read_record(directory, "operator-review.json")
                if receipt.get("decision_sha256") != digest:
                    return "Review receipt mismatch; no approval recorded."
                return "This exact decision was already reviewed; no order was sent."
            _write_record(
                directory,
                "operator-review.json",
                {
                    "audit_id": audit_id,
                    "decision_sha256": digest,
                    "reviewed_at": UtcClock().now().isoformat(),
                    "reason": reason.strip(),
                    "scope": "Reasoning acknowledgement only; not order approval or evidence verification.",
                },
            )
            return "Reasoning review recorded for this exact decision. No order sent; freshness, risk, exceptions and cycle approval still apply."
        except (OSError, ValueError, KeyError, TypeError):
            return "Cannot verify the saved decision; no approval recorded."
