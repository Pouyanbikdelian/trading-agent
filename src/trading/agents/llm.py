"""Minimal LLM client — provider-agnostic, JSON-strict, no SDK deps.

Uses httpx (already a project dependency) against the Anthropic or
OpenAI HTTP APIs, chosen by which key is present in the environment:

* ``ANTHROPIC_API_KEY``  -> Anthropic Messages API
* ``OPENAI_API_KEY``     -> OpenAI Chat Completions

Model is ``AGENTS_MODEL`` (default a mid-tier model — committee chatter
doesn't need a frontier model; see concept doc's cost-discipline note).

``complete_json`` extracts the first JSON object from the response and
retries once with a terse "ONLY JSON" nudge — LLM output that can't be
parsed is treated as no take at all, never as a guess.
"""

from __future__ import annotations

import json
import os
from typing import Any

from trading.core.logging import logger

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
TIMEOUT_S = 60.0


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
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in response")


def _anthropic_key() -> str | None:
    from trading.core.config import settings

    return settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")


def _openai_key() -> str | None:
    from trading.core.config import settings

    return settings.openai_api_key or os.getenv("OPENAI_API_KEY")


def _agents_model() -> str:
    from trading.core.config import settings

    return settings.agents_model or os.getenv("AGENTS_MODEL", "")


def _call_anthropic(system: str, prompt: str, *, model: str, max_tokens: int) -> str:
    import httpx

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
        },
        timeout=TIMEOUT_S,
    )
    _raise_with_body(resp)
    return "".join(b.get("text", "") for b in resp.json().get("content", []))


def _call_openai(system: str, prompt: str, *, model: str, max_tokens: int) -> str:
    import httpx

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
        timeout=TIMEOUT_S,
    )
    _raise_with_body(resp)
    return resp.json()["choices"][0]["message"]["content"]


def complete_text(system: str, prompt: str, *, max_tokens: int = 1200) -> str:
    model = _agents_model()
    if _anthropic_key():
        return _call_anthropic(
            system, prompt, model=model or DEFAULT_ANTHROPIC_MODEL, max_tokens=max_tokens
        )
    if _openai_key():
        return _call_openai(
            system, prompt, model=model or DEFAULT_OPENAI_MODEL, max_tokens=max_tokens
        )
    raise AgentsDisabledError("set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable agents")


def complete_json(system: str, prompt: str, *, max_tokens: int = 1200) -> dict[str, Any]:
    """One completion, parsed as JSON; one retry on parse failure."""
    text = complete_text(system, prompt, max_tokens=max_tokens)
    try:
        return _extract_json(text)
    except Exception:
        logger.bind(component="agents").warning("unparseable LLM output; retrying once")
        text = complete_text(
            system,
            prompt + "\n\nRespond with ONLY a valid JSON object. No prose.",
            max_tokens=max_tokens,
        )
        return _extract_json(text)
