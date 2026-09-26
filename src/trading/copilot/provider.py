"""LLM provider abstraction for the copilot.

Three providers, one interface. Default is **Anthropic Claude Opus 5.5**
(Haiku until 2026-09-26; the operator moved every agent to one model), and
the API key already exists in the deploy's .env. Qwen (Alibaba Model Studio / DashScope) and DeepSeek ride their
OpenAI-compatible endpoints so they share one code path.

Configuration is env-only (never code):

* ``COPILOT_PROVIDER``  — anthropic | qwen | deepseek (default anthropic)
* ``COPILOT_MODEL``     — override the per-provider default model
* ``COPILOT_BASE_URL``  — override the endpoint (e.g. DashScope region)
* keys: ``ANTHROPIC_API_KEY`` / ``DASHSCOPE_API_KEY`` / ``DEEPSEEK_API_KEY``

Security: the ONLY things ever sent to the provider are the system
charter and the evidence prompt the engine builds. No broker
credentials, tokens, or account identifiers pass through here — and
nothing in this module has access to them anyway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

TIMEOUT_S = 30.0
MAX_TOKENS = 900
# Opus/Sonnet 5.x always think, and max_tokens caps thinking PLUS the
# answer: at 900 a medium-effort think could leave nothing to say. Low
# effort keeps a chat reply quick; the larger ceiling and timeout give the
# thinking somewhere to go. Haiku 4.5 rejects the effort field entirely.
THINKING_MODEL_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5")
THINKING_TIMEOUT_S = 90.0
THINKING_MAX_TOKENS = 4_000
THINKING_EFFORT = "low"

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "qwen": "qwen-plus",
    "deepseek": "deepseek-chat",
}
_DEFAULT_BASE_URLS = {
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
}
_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


class ProviderError(RuntimeError):
    """Configuration or transport failure — the bot shows it verbatim."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    base_url: str | None
    api_key: str

    @staticmethod
    def from_env() -> ProviderConfig:
        name = (os.getenv("COPILOT_PROVIDER") or "anthropic").strip().lower()
        if name not in _DEFAULT_MODELS:
            raise ProviderError(
                f"COPILOT_PROVIDER={name!r} — expected one of {sorted(_DEFAULT_MODELS)}"
            )
        key = os.getenv(_KEY_ENVS[name], "")
        if not key:
            raise ProviderError(
                f"{_KEY_ENVS[name]} is not set — the copilot needs it for provider {name!r}"
            )
        return ProviderConfig(
            name=name,
            model=os.getenv("COPILOT_MODEL") or _DEFAULT_MODELS[name],
            base_url=os.getenv("COPILOT_BASE_URL") or _DEFAULT_BASE_URLS.get(name),
            api_key=key,
        )


def complete(system: str, prompt: str, *, config: ProviderConfig | None = None) -> str:
    """One synchronous completion. Raises ProviderError on any failure —
    the caller degrades gracefully, never retries into a spend loop."""
    cfg = config or ProviderConfig.from_env()
    try:
        if cfg.name == "anthropic":
            return _anthropic(system, prompt, cfg)
        return _openai_compatible(system, prompt, cfg)
    except httpx.HTTPError as e:
        raise ProviderError(f"{cfg.name} request failed: {type(e).__name__}: {e}") from e


def _anthropic_body(system: str, prompt: str, model: str) -> tuple[dict[str, Any], float]:
    """Request body and timeout for ``model``: thinking models get an
    explicit low effort and room to think; others keep the old budget."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    if model.startswith(THINKING_MODEL_PREFIXES):
        body["max_tokens"] = THINKING_MAX_TOKENS
        body["output_config"] = {"effort": THINKING_EFFORT}
        return body, THINKING_TIMEOUT_S
    return body, TIMEOUT_S


def _anthropic(system: str, prompt: str, cfg: ProviderConfig) -> str:
    body, timeout_s = _anthropic_body(system, prompt, cfg.model)
    r = httpx.post(
        (cfg.base_url or "https://api.anthropic.com") + "/v1/messages",
        headers={
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise ProviderError(f"anthropic HTTP {r.status_code}: {r.text[:200]}")
    data: dict[str, Any] = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []))


def _openai_compatible(system: str, prompt: str, cfg: ProviderConfig) -> str:
    if not cfg.base_url:
        raise ProviderError(f"no base_url for provider {cfg.name!r}")
    r = httpx.post(
        cfg.base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {cfg.api_key}", "content-type": "application/json"},
        json={
            "model": cfg.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=TIMEOUT_S,
    )
    if r.status_code != 200:
        raise ProviderError(f"{cfg.name} HTTP {r.status_code}: {r.text[:200]}")
    data: dict[str, Any] = r.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as e:
        raise ProviderError(f"{cfg.name} malformed response: {e}") from e
