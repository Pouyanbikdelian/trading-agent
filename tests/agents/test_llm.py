"""LLM transport contracts: model routing, adaptive effort and safe telemetry."""

from __future__ import annotations

import json
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
    assert llm._token_budget("standard", None) == 12_000
    assert llm.DEFAULT_ANTHROPIC_MODEL == "claude-opus-5-5"
    assert llm.FRONTIER_ANTHROPIC_MODEL == "claude-opus-5-5"


@pytest.mark.parametrize(
    "statement",
    [
        "A literal } does not close the response",
        "A literal { is not a new object",
        'Escaped quote: " and slash \\ followed by }',
        'Nested-looking text: {"key": {"nested": true}}',
    ],
)
def test_json_decoder_treats_braces_in_strings_as_text(statement: str) -> None:
    expected = {"take": statement, "prediction": {"confidence": 0.7}}
    response = "Here is the result:\n```json\n" + json.dumps(expected) + "\n```"
    assert llm._extract_json(response) == expected


def test_valid_quoted_braces_do_not_trigger_a_second_paid_call(monkeypatch) -> None:
    calls = []

    def complete(*args, **kwargs):
        calls.append(args)
        return '{"take":"Closing brace } in prose", "ok":true}'

    monkeypatch.setattr(llm, "complete_text", complete)
    assert llm.complete_json("system", "prompt")["ok"]
    assert len(calls) == 1


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_http_failures_are_telemetried_once_without_sensitive_bodies(monkeypatch, provider) -> None:
    rows = []

    def post(*args, **kwargs):
        return httpx.Response(429, text="private provider diagnostic")

    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(llm, "_record_telemetry", lambda **row: rows.append(row))
    call = llm._call_anthropic if provider == "anthropic" else llm._call_openai

    with pytest.raises(RuntimeError, match="429"):
        call("private system", "private prompt", model="test", max_tokens=100, tier=None)

    assert len(rows) == 1
    assert rows[0]["http_status"] == 429
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["usage"] is None
    assert "private" not in json.dumps(rows)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_malformed_provider_body_is_telemetried(monkeypatch, provider) -> None:
    rows = []
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: httpx.Response(200, text="broken"))
    monkeypatch.setattr(llm, "_record_telemetry", lambda **row: rows.append(row))
    call = llm._call_anthropic if provider == "anthropic" else llm._call_openai

    with pytest.raises(ValueError):
        call("system", "prompt", model="test", max_tokens=100, tier=None)

    assert len(rows) == 1
    assert rows[0]["http_status"] == 200
    assert rows[0]["error_type"] == "JSONDecodeError"


def test_openai_token_usage_is_normalized_for_telemetry(monkeypatch) -> None:
    telemetry = {}
    monkeypatch.setattr(
        "httpx.post",
        lambda *args, **kwargs: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45},
            },
        ),
    )
    monkeypatch.setattr(llm, "_record_telemetry", lambda **row: telemetry.update(row))

    result = llm._call_openai("system", "prompt", model="test", max_tokens=100, tier=None)

    assert json.loads(result) == {"ok": True}
    assert telemetry["usage"] == {"input_tokens": 123, "output_tokens": 45}


def test_a_cut_off_answer_is_retried_once_with_twice_the_room(monkeypatch) -> None:
    """2026-09-26: a max_tokens stop used to be retried with the SAME
    ceiling, so it failed again and the voice sat the meeting out."""
    calls: list[int] = []

    def complete(system: str, prompt: str, *, max_tokens=None, tier=None) -> str:
        calls.append(max_tokens)
        if len(calls) == 1:
            llm._set_stop("max_tokens")
            return '{"take": "half an ans'
        llm._set_stop("end_turn")
        return '{"take": "whole answer"}'

    monkeypatch.setattr(llm, "complete_text", complete)
    out = llm.complete_json("s", "p", max_tokens=12_000, tier="standard")
    assert out == {"take": "whole answer"}
    assert calls == [12_000, 24_000]


def test_still_cut_off_after_the_retry_is_an_error_not_a_fragment(monkeypatch) -> None:
    def complete(system: str, prompt: str, *, max_tokens=None, tier=None) -> str:
        llm._set_stop("max_tokens")
        return '{"take": "cut'

    monkeypatch.setattr(llm, "complete_text", complete)
    with pytest.raises(ValueError, match="cut off at max_tokens=48000"):
        llm.complete_json("s", "p", max_tokens=24_000, tier="frontier")


def test_the_anthropic_stop_reason_is_recorded(monkeypatch) -> None:
    class Cut(_Response):
        def json(self) -> dict[str, Any]:
            return {"content": [], "usage": {}, "stop_reason": "max_tokens"}

    monkeypatch.setattr("httpx.post", lambda *a, **k: Cut())
    monkeypatch.setattr(llm, "_record_telemetry", lambda **row: None)
    llm._call_anthropic("s", "p", model="claude-opus-5-5", max_tokens=100, tier=None)
    assert llm.last_stop_reason() == "max_tokens"
