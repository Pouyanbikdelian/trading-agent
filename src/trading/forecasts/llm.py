"""The LLM forecaster: same numbers as the baseline, plus headlines."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"
#: Two forecasters from one model, differing only in what they are shown.
#: Not personas — an experiment. "numbers" sees exactly what the mechanical
#: baseline sees, so it asks whether an LLM reads the same figures any
#: better. "news" adds headlines, so the gap between the two is what the
#: text was worth. A model that needs a costume to disagree with itself
#: would tell us nothing.
LENSES = ("numbers", "news")
BATCH_SIZE = 10
#: Hard ceiling on calls per run, whatever TOP_N says. Ten calls of ten.
MAX_CALLS = 10
MAX_TOKENS = 6_000

SYSTEM = (
    "You are a probability forecaster for US stocks. For each stock you are given, "
    "estimate the probability that its total return over the next 5 trading sessions, "
    "and separately over the next 21, will be HIGHER than SPY's over the same sessions.\n"
    "\n"
    "You are scored with the Brier score against what actually happens, and compared "
    "with a simple statistical model that sees the same price numbers. Honest "
    "probabilities win; confident ones lose. Across large-cap stocks about half beat "
    "the index in any window, and short-horizon relative returns are close to "
    "unpredictable, so most answers belong between 0.45 and 0.55. Move further only "
    "for a specific reason in the data or headlines you were given, and say what it is. "
    "A headline is something that was reported, not something you have verified, and "
    "the market has usually seen it too. Do not use anything you believe you know about "
    "prices after the as-of date.\n"
    "\n"
    'Respond ONLY with JSON: {"forecasts": [{"symbol": "AAPL", "p5": 0.51, '
    '"p21": 0.53, "why": "<one short clause, or \'no specific reason\'>"}]} '
    "with exactly one entry per stock given, probabilities between 0.05 and 0.95."
)


def build_prompt(asof: str, cards: list[dict[str, Any]]) -> str:
    return json.dumps({"as_of_close": asof, "benchmark": "SPY", "stocks": cards}, default=str)


def source_name(model: str, lens: str) -> str:
    return f"llm:{model}:{lens}"


def lens_cards(cards: list[dict[str, Any]], lens: str) -> list[dict[str, Any]]:
    """The cards as this lens may see them."""
    if lens == "news":
        return cards
    return [{k: v for k, v in card.items() if k != "recent_headlines"} for card in cards]


def batches_for(cards: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [cards[i : i + BATCH_SIZE] for i in range(0, len(cards), BATCH_SIZE)][:MAX_CALLS]


def parse(response: dict[str, Any], symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Keep only well-formed entries for stocks that were actually asked about.

    A missing or malformed entry is dropped, never filled in: an invented
    0.5 would be graded as if the model had said it.
    """
    wanted = {s.upper() for s in symbols}
    out: dict[str, dict[str, Any]] = {}
    rows = response.get("forecasts") if isinstance(response, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).upper().strip()
        if symbol not in wanted or symbol in out:
            continue
        probs = {}
        for horizon, key in ((5, "p5"), (21, "p21")):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value = float(value)
            if math.isfinite(value) and 0.01 <= value <= 0.99:
                probs[horizon] = value
        if probs:
            out[symbol] = {"probs": probs, "why": str(row.get("why") or "")[:200]}
    return out


def _default_call(model: str) -> Callable[[str, str], dict[str, Any]]:
    from trading.agents import llm

    def call(system: str, prompt: str) -> dict[str, Any]:
        # tier="shadow" is only a telemetry label here: it keeps these calls
        # on their own line in llm_usage.jsonl, at the standard effort.
        text = llm._call_anthropic(
            system, prompt, model=model, max_tokens=MAX_TOKENS, tier="shadow"
        )
        return llm._extract_json(text)

    return call


def forecast(
    asof: str,
    cards: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    call: Callable[[str, str], dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Ask in batches. One failed batch loses ten stocks, not the run."""
    call = call or _default_call(model)
    results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for batch in batches_for(cards):
        symbols = [str(c["symbol"]) for c in batch]
        try:
            results.update(parse(call(SYSTEM, build_prompt(asof, batch)), symbols))
        except Exception as exc:
            errors.append(f"{symbols[0]}..{symbols[-1]}: {type(exc).__name__}: {str(exc)[:120]}")
    return results, errors
