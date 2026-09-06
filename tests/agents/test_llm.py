"""LLM transport contracts: model routing, adaptive effort and safe telemetry."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from trading.agents import llm


class _Response:
    status_code = 200
    text = ""

    def json(self) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "usage": {"input_tokens": 123, "output_tokens": 45},
            "stop_reason": "end_turn",
        }


def test_anthropic_frontier_request_uses_adaptive_high_effort_and_telemetry(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    telemetry: dict[str, Any] = {}

    def post(*args: Any, **kwargs: Any) -> _Response:
        seen.update(kwargs)
        return _Response()

    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(llm, "_record_telemetry", lambda **row: telemetry.update(row))
    monkeypatch.setattr(llm, "_anthropic_effort", lambda tier: "high")

    out = llm._call_anthropic(
        "system", "prompt", model="claude-opus-5", max_tokens=8000, tier="frontier"
    )

    assert out == '{"ok": true}'
    assert seen["json"]["thinking"] == {"type": "adaptive"}
    assert seen["json"]["output_config"] == {"effort": "high"}
    assert seen["json"]["max_tokens"] == 8000
    assert seen["timeout"] == llm.FRONTIER_TIMEOUT_S
    assert telemetry["provider"] == "anthropic"
    assert telemetry["model"] == "claude-opus-5"
    assert telemetry["tier"] == "frontier"
    assert telemetry["usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert telemetry["timeout_s"] == llm.FRONTIER_TIMEOUT_S


def test_frontier_timeout_is_telemetried_without_a_completion(monkeypatch) -> None:
    telemetry: dict[str, Any] = {}

    def post(*args: Any, **kwargs: Any) -> _Response:
        raise httpx.ReadTimeout("slow response")

    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(llm, "_record_telemetry", lambda **row: telemetry.update(row))

    with pytest.raises(httpx.ReadTimeout):
        llm._call_anthropic(
            "system", "prompt", model="claude-opus-5", max_tokens=8000, tier="frontier"
        )

    assert telemetry["tier"] == "frontier"
    assert telemetry["timeout_s"] == llm.FRONTIER_TIMEOUT_S
    assert telemetry["error_type"] == "ReadTimeout"


def test_explicit_token_budget_wins_over_tier_default() -> None:
    assert llm._token_budget("frontier", 1234) == 1234
    assert llm.DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-5"
    assert llm.FRONTIER_ANTHROPIC_MODEL == "claude-opus-5"
