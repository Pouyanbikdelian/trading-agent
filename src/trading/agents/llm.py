"""Minimal LLM client — provider-agnostic, JSON-strict, no SDK deps.

Uses httpx (already a project dependency) against the Anthropic or
OpenAI HTTP APIs, chosen by which key is present in the environment:

* ``ANTHROPIC_API_KEY``  -> Anthropic Messages API
* ``OPENAI_API_KEY``     -> OpenAI Chat Completions

Model is ``AGENTS_MODEL`` (default Sonnet 5 for specialist research) or
``AGENTS_MODEL_FRONTIER`` (default Opus 5 for decision nodes).  Adaptive
thinking is bounded: specialists use medium effort and the manager, PM,
challenger and Curator use high effort.  Every response records model,
effort, tokens and latency to a local telemetry journal — never its prompt
or response body.

``complete_json`` extracts the first JSON object from the response and
retries once with a terse "ONLY JSON" nudge — LLM output that can't be
parsed is treated as no take at all, never as a guess.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading.core.logging import logger

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Frontier tier for the committee's decision nodes (challenger, manager).
# Overridable via the AGENTS_MODEL_FRONTIER env var without a code change.
FRONTIER_ANTHROPIC_MODEL = "claude-opus-5"
FRONTIER_OPENAI_MODEL = "gpt-4o"
DEFAULT_TIMEOUT_S = 60.0
FRONTIER_TIMEOUT_S = 180.0
DEFAULT_MAX_TOKENS = 4_000
FRONTIER_MAX_TOKENS = 8_000


class AgentsDisabledError(RuntimeError):
    """No API key configured — the committee cannot run."""


def _raise_with_body(resp: Any) -> None:
    """4xx/5xx with the provider's actual error message, not just the
    status line — '400 Bad Request' alone hides 'credit balance too low'
    vs 'model not found', which need opposite fixes."""
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM API {resp.status_code}: {resp.text[:300]}")


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : i + 1])
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("JSON completion is not an object")
    raise ValueError("unbalanced JSON in response")


def _anthropic_key() -> str | None:
    from trading.core.config import settings

    return settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")


def _openai_key() -> str | None:
    from trading.core.config import settings

    return settings.openai_api_key or os.getenv("OPENAI_API_KEY")


def _agents_model() -> str:
    from trading.core.config import settings

    return str(settings.agents_model or os.getenv("AGENTS_MODEL", "") or "")


def _token_budget(tier: str | None, requested: int | None) -> int:
    """Resolve a bounded completion budget at call time.

    Anthropic's adaptive thinking consumes part of the response budget, so
    retaining the old 1,200-token cap made a stronger model less useful: it
    could reason, or it could return the structured decision, but rarely
    both.  These are ceilings, not targets, and remain operator-overridable.
    """
    if requested is not None:
        return requested
    try:
        from trading.core.config import settings

        return (
            int(settings.agents_frontier_max_tokens)
            if tier == "frontier"
            else int(settings.agents_max_tokens)
        )
    except Exception:
        return FRONTIER_MAX_TOKENS if tier == "frontier" else DEFAULT_MAX_TOKENS


def _anthropic_effort(tier: str | None) -> str:
    """The reasoning effort is explicit rather than a model-default gamble."""
    try:
        from trading.core.config import settings

        return str(
            settings.agents_frontier_effort if tier == "frontier" else settings.agents_effort
        )
    except Exception:
        return "high" if tier == "frontier" else "medium"


def _timeout_s(tier: str | None) -> float:
    """Keep high-effort decision calls bounded without cutting them short.

    A single 60-second transport limit was suitable for short specialist
    observations, but it turns a valid longer Opus reasoning pass into a
    failed Curator/PM run.  This is still a finite operator-configurable
    bound; it is not an unbounded wait in a live scheduler.
    """
    try:
        from trading.core.config import settings

        return float(
            settings.agents_frontier_timeout_s if tier == "frontier" else settings.agents_timeout_s
        )
    except Exception:
        return FRONTIER_TIMEOUT_S if tier == "frontier" else DEFAULT_TIMEOUT_S


def _record_telemetry(
    *,
    provider: str,
    model: str,
    tier: str | None,
    effort: str | None,
    max_tokens: int,
    latency_ms: float,
    usage: dict[str, Any] | None,
    stop_reason: str | None,
    timeout_s: float,
    error_type: str | None = None,
) -> None:
    """Persist non-sensitive LLM cost/latency evidence for operations.

    A model upgrade without this record is faith: a timeout, runaway token
    use, or cache miss gets misdiagnosed as an agent-quality problem.  This
    intentionally stores no prompts, completions, API keys or account data.
    A failed telemetry write is never allowed to break a trading review.
    """
    row = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "tier": tier or "standard",
        "effort": effort,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": (usage or {}).get("input_tokens"),
        "output_tokens": (usage or {}).get("output_tokens"),
        "cache_creation_input_tokens": (usage or {}).get("cache_creation_input_tokens"),
        "cache_read_input_tokens": (usage or {}).get("cache_read_input_tokens"),
        "stop_reason": stop_reason,
        "error_type": error_type,
    }
    logger.bind(component="agents.llm", **row).info("LLM completion")
    try:
        from trading.core.config import settings

        path = Path(settings.state_dir) / "llm_usage.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND gives each small line a single write; no prompt material
        # means this diagnostic file is safe to retain alongside state.
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())
        finally:
            os.close(fd)
    except Exception:
        logger.bind(component="agents.llm").warning("could not write LLM telemetry")


def _call_anthropic(
    system: str, prompt: str, *, model: str, max_tokens: int, tier: str | None
) -> str:
    import httpx

    effort = _anthropic_effort(tier)
    timeout_s = _timeout_s(tier)
    started = time.monotonic()
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": _anthropic_key() or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
                # Claude 5's adaptive thinking lets the API decide how much
                # scratch work a particular decision needs; effort keeps that
                # freedom finite.  Do not use a fixed thinking-token budget —
                # it is brittle across straightforward and adversarial prompts.
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            },
            timeout=timeout_s,
        )
    except Exception as exc:
        _record_telemetry(
            provider="anthropic",
            model=model,
            tier=tier,
            effort=effort,
            max_tokens=max_tokens,
            latency_ms=(time.monotonic() - started) * 1_000,
            usage=None,
            stop_reason=None,
            timeout_s=timeout_s,
            error_type=type(exc).__name__,
        )
        raise
    _raise_with_body(resp)
    body = resp.json()
    _record_telemetry(
        provider="anthropic",
        model=model,
        tier=tier,
        effort=effort,
        max_tokens=max_tokens,
        latency_ms=(time.monotonic() - started) * 1_000,
        usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
        stop_reason=str(body.get("stop_reason")) if body.get("stop_reason") else None,
        timeout_s=timeout_s,
    )
    return "".join(b.get("text", "") for b in body.get("content", []))


def _call_openai(system: str, prompt: str, *, model: str, max_tokens: int, tier: str | None) -> str:
    import httpx

    timeout_s = _timeout_s(tier)
    started = time.monotonic()
    try:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {_openai_key() or ''}"},
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=timeout_s,
        )
    except Exception as exc:
        _record_telemetry(
            provider="openai",
            model=model,
            tier=tier,
            effort=None,
            max_tokens=max_tokens,
            latency_ms=(time.monotonic() - started) * 1_000,
            usage=None,
            stop_reason=None,
            timeout_s=timeout_s,
            error_type=type(exc).__name__,
        )
        raise
    _raise_with_body(resp)
    body = resp.json()
    choice = body["choices"][0]
    _record_telemetry(
        provider="openai",
        model=model,
        tier=tier,
        effort=None,
        max_tokens=max_tokens,
        latency_ms=(time.monotonic() - started) * 1_000,
        usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
        stop_reason=str(choice.get("finish_reason")) if choice.get("finish_reason") else None,
        timeout_s=timeout_s,
    )
    return str(choice["message"]["content"])


def _frontier_model_override() -> str | None:
    """Operator-set frontier model from settings or env, if any."""
    from trading.core.config import settings

    return getattr(settings, "agents_model_frontier", None) or os.getenv("AGENTS_MODEL_FRONTIER")


def _resolve_anthropic_model(tier: str | None) -> str:
    if tier == "frontier":
        return _frontier_model_override() or FRONTIER_ANTHROPIC_MODEL
    return _agents_model() or DEFAULT_ANTHROPIC_MODEL


def _resolve_openai_model(tier: str | None) -> str:
    if tier == "frontier":
        return FRONTIER_OPENAI_MODEL
    return _agents_model() or DEFAULT_OPENAI_MODEL


def complete_text(
    system: str, prompt: str, *, max_tokens: int | None = None, tier: str | None = None
) -> str:
    """One text completion. ``tier='frontier'`` routes the decision nodes
    (challenger, manager) to the stronger model; the default (None) keeps the
    mid-tier committee model. Provider is chosen by which API key is present."""
    budget = _token_budget(tier, max_tokens)
    if _anthropic_key():
        return _call_anthropic(
            system,
            prompt,
            model=_resolve_anthropic_model(tier),
            max_tokens=budget,
            tier=tier,
        )
    if _openai_key():
        return _call_openai(
            system, prompt, model=_resolve_openai_model(tier), max_tokens=budget, tier=tier
        )
    raise AgentsDisabledError("set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable agents")


def complete_json(
    system: str, prompt: str, *, max_tokens: int | None = None, tier: str | None = None
) -> dict[str, Any]:
    """One completion, parsed as JSON; one retry on parse failure."""
    text = complete_text(system, prompt, max_tokens=max_tokens, tier=tier)
    try:
        return _extract_json(text)
    except Exception:
        logger.bind(component="agents").warning("unparseable LLM output; retrying once")
        text = complete_text(
            system,
            prompt + "\n\nRespond with ONLY a valid JSON object. No prose.",
            max_tokens=max_tokens,
            tier=tier,
        )
        return _extract_json(text)
