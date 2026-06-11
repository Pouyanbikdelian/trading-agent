"""LLM agent committee — Phase 1 (advisory only).

See docs/concept_multiagent_memory.md. Agents read the live system
state + permanent memory, write gradeable takes back into memory, and
publish a daily digest to Telegram. They NEVER construct orders; the
RiskManager remains the only path to the broker.
"""

from trading.agents.committee import run_committee

__all__ = ["run_committee"]
