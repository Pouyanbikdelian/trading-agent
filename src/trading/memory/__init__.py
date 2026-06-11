"""Permanent memory spine — the system's lived experience.

See docs/concept_multiagent_memory.md. Five stores, one principle:
append-only, versioned, nothing deleted — only superseded, with links.

* Journal     — every event, raw, forever (SQLite, append-only)
* Episodes    — closed trades + full context at the time
* Lessons     — distilled beliefs with provenance + evidence counters
                (markdown cards, Obsidian-compatible, indexed in SQLite)
* World State — living narrative dossiers (markdown, timestamped appends)
* Scorecard   — every prediction logged with a horizon, auto-graded later
* Trust       — per-source credibility ledger (Beta prior, never reset)
"""

from trading.memory.store import MemoryStore

__all__ = ["MemoryStore"]
