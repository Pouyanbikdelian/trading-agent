"""Anthropic Message Batches: the same requests at half the price, later.

A daily shadow forecast has no use for an answer in seconds, and the batch
API bills input and output at 50%. The trade is latency (usually under an
hour, at most 24) and a second step to collect, so the job submits tonight
and whichever run comes next collects.

Shapes follow platform.claude.com/docs/en/build-with-claude/batch-processing
as read on 2026-11-11: ``POST /v1/messages/batches`` with
``{"requests": [{"custom_id", "params"}]}``; poll the batch until
``processing_status == "ended"``; fetch ``results_url`` as JSONL and match
lines by ``custom_id`` — results arrive in any order.
"""

from __future__ import annotations

import json
import re
from typing import Any

API = "https://api.anthropic.com/v1/messages/batches"
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
TIMEOUT_S = 60.0


def _headers() -> dict[str, str]:
    from trading.agents import llm

    key = llm._anthropic_key()
    if not key:
        raise llm.AgentsDisabledError("ANTHROPIC_API_KEY is not set")
    return {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}


def message_params(system: str, prompt: str, *, model: str, max_tokens: int) -> dict[str, Any]:
    """One request body, identical to the synchronous call's so a forecast
    does not depend on which route produced it."""
    from trading.agents import llm

    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": llm._anthropic_effort("shadow")},
    }


def submit(requests: list[dict[str, Any]]) -> str:
    import httpx

    for request in requests:
        if not CUSTOM_ID.match(str(request.get("custom_id", ""))):
            raise ValueError(f"invalid batch custom_id: {request.get('custom_id')!r}")
    response = httpx.post(API, headers=_headers(), json={"requests": requests}, timeout=TIMEOUT_S)
    response.raise_for_status()
    return str(response.json()["id"])


def poll(batch_id: str) -> dict[str, Any]:
    import httpx

    response = httpx.get(f"{API}/{batch_id}", headers=_headers(), timeout=TIMEOUT_S)
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}


def results(results_url: str) -> list[dict[str, Any]]:
    import httpx

    response = httpx.get(results_url, headers=_headers(), timeout=TIMEOUT_S)
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def message_text(line: dict[str, Any]) -> tuple[str | None, dict[str, Any], str]:
    """(text, usage, result type) for one results line. Text is None unless
    the request succeeded; thinking blocks carry no ``text`` and are skipped."""
    raw_result = line.get("result")
    result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
    kind = str(result.get("type") or "unknown")
    if kind != "succeeded":
        return None, {}, kind
    raw_message = result.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    raw_blocks = message.get("content")
    blocks: list[Any] = raw_blocks if isinstance(raw_blocks, list) else []
    text = "".join(str(b.get("text", "")) for b in blocks if isinstance(b, dict))
    raw_usage = message.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    return text, usage, kind
