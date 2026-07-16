"""Investment Committee Copilot — Phase 1, strictly READ-ONLY.

Answers "why did we buy/sell/hold X?" questions in Telegram by
retrieving what the committee actually said (journaled rulings and
per-agent takes), what actually happened (orders, fills, positions),
and what is true now — then asking a cheap LLM to synthesize with
mandatory citations.

Hard boundary, stated once for the whole package: nothing in here
imports ``trading.execution`` or constructs an ``Order``. Every SQLite
file is opened ``mode=ro``. The copilot explains; it never acts. Order
drafting/execution stays a separate, deterministic, explicitly
confirmed system (rule #4 of CLAUDE.md extends to this assistant).
A test (tests/copilot/test_no_execution.py) enforces the import ban.
"""

from trading.copilot.engine import answer

__all__ = ["answer"]
