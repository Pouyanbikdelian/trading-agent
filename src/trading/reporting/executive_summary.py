r"""Executive summary — LLM-generated prose given the day's structured data.

Behaviour
---------
* If ``ANTHROPIC_API_KEY`` is set in the environment, this module calls
  the Anthropic API (Claude Sonnet by default) and returns a 3-paragraph
  narrative covering: portfolio performance, market regime,
  position-by-position takeaways with cited news headlines.
* If the API key is *not* set (or the call fails), a deterministic
  bullet-point fallback is generated from the same payload. The report
  is never blocked by a missing or flaky LLM.

The prompt is designed for finance: the model is told to be
quantitative, cite the headlines we pulled, flag anything that looks
inconsistent with the data, and to refuse to invent numbers that aren't
in the payload.  No emoji, no marketing prose.

Cost note
---------
At Sonnet 4.6 pricing ($3 / 1M input tokens, $15 / 1M output), a daily
report of ~3,000 input tokens (positions + news headlines) and ~600
output tokens costs ~$0.02 / day or ~$5 / year. Cheap enough to run
unconditionally; we'll only flip a per-day budget cap if usage drifts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from trading.core.logging import logger
from trading.reporting.daily import DailyReport
from trading.reporting.news import Headline

SYSTEM_PROMPT = """\
You are the analyst writing a one-paragraph daily summary for a quant
trader's automated trading system.

Strict rules:
- Use only numbers and facts present in the JSON payload provided.
  Never invent figures, prices, or events.
- Cite news headlines explicitly when relevant: 'Reuters reported that
  ...'. Refer to headlines by their source and a short paraphrase.
- Be quantitative. If the daily PnL is given, use the actual percentage.
- Tone: factual, terse, sober. No marketing language, no emoji, no
  rhetorical flourishes.
- If the data shows anything alarming (drawdown approaching the halt
  threshold, position concentration over 20%, stale heartbeat,
  contradictory signals across sources), flag it explicitly in a
  closing line.
- 3 paragraphs maximum, ~150 words total.

Structure:
  Paragraph 1: portfolio performance (PnL, equity, top contributors).
  Paragraph 2: market regime + risk state.
  Paragraph 3: position-level takeaways with headline citations.
"""


def summarise(
    report: DailyReport,
    *,
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    max_tokens: int = 800,
) -> str:
    """Return a narrative executive summary for the given daily report.

    Falls back to a deterministic bullet-point summary when no API key
    or when the API call fails for any reason.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _fallback_summary(report)

    payload = _serialise_payload(report)
    user_message = (
        "Here is the daily report payload as JSON. Write the summary.\n\n"
        f"```json\n{json.dumps(payload, default=str)}\n```"
    )

    try:
        from anthropic import Anthropic  # lazy import — only loaded if used

        client = Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        # response.content is a list of content blocks; take text from text blocks.
        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", ""))
        narrative = "\n".join(text_parts).strip()
        return narrative or _fallback_summary(report)
    except Exception as e:
        logger.bind(component="exec_summary").exception(f"LLM call failed: {e!r}")
        return _fallback_summary(report)


def _serialise_payload(report: DailyReport) -> dict[str, Any]:
    """Pick a small, JSON-friendly subset of the DailyReport that gives
    the model enough context without blowing the token budget."""
    payload: dict[str, Any] = {
        "as_of": report.as_of.isoformat(),
        "cash": round(report.cash, 2),
        "equity": round(report.equity, 2),
        "daily_pnl_pct": round(report.daily_pnl_pct, 4),
        "week_pnl_pct": round(report.week_pnl_pct, 4),
        "month_pnl_pct": round(report.month_pnl_pct, 4),
        "ytd_pnl_pct": round(report.ytd_pnl_pct, 4),
        "vix_level": report.vix_level,
        "vix_regime": report.vix_regime,
        "halted": report.halted,
        "halt_reason": report.halt_reason,
        "n_positions": len(report.positions),
        "positions": [
            {
                "symbol": sym,
                "weight": round(d["weight"], 4),
                "market_value": round(d["market_value"], 2),
                "unrealized_pnl": round(d["unrealized_pnl"], 2),
            }
            for sym, d in sorted(
                report.positions.items(),
                key=lambda kv: -kv[1]["market_value"],
            )[:10]
        ],
        "news_by_symbol": {
            sym: [
                {"title": h.title, "source": h.source}
                for h in (items if not is_dataclass(items) else [asdict(items)])
            ][:3]
            for sym, items in report.news_by_symbol.items()
        },
    }
    return payload


def _fallback_summary(r: DailyReport) -> str:
    """Deterministic summary used when the LLM isn't available."""
    parts: list[str] = []

    parts.append(
        f"- Portfolio equity ${r.equity:,.2f}, cash ${r.cash:,.2f}. "
        f"Daily PnL {r.daily_pnl_pct:+.2%} (${r.daily_pnl:+,.2f}), "
        f"week {r.week_pnl_pct:+.2%}, month {r.month_pnl_pct:+.2%}, "
        f"YTD {r.ytd_pnl_pct:+.2%}."
    )

    if r.vix_level is not None:
        parts.append(f"- VIX = {r.vix_level:.2f}, regime = {r.vix_regime or 'unknown'}.")
    if r.halted:
        parts.append(f"- RISK MANAGER HALTED: {r.halt_reason}.")
    elif r.heartbeat_age_seconds is not None and r.heartbeat_age_seconds > 86400:
        parts.append(
            f"- Heartbeat stale ({r.heartbeat_age_seconds:.0f}s); "
            "runner may have stopped reporting."
        )

    if r.positions:
        top = sorted(
            r.positions.items(),
            key=lambda kv: -kv[1]["market_value"],
        )[:5]
        for sym, d in top:
            line = (
                f"  - {sym}: {d['weight']:+.2%} of equity (unrealized {d['unrealized_pnl']:+,.2f})"
            )
            if r.news_by_symbol.get(sym):
                first = r.news_by_symbol[sym][0]
                title = first.title if isinstance(first, Headline) else first.get("title", "")
                if title:
                    line += f"  | latest: '{title[:80]}'"
            parts.append(line)

    return "\n".join(parts)
