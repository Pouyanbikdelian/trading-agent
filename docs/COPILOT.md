# Investment Committee Copilot — Phase 1 (read-only)

Ask the Telegram bot *why* the system did things, and whether old theses
still hold. The copilot retrieves what the committee actually said
(journaled rulings + per-agent takes), what actually happened (orders,
fills, positions), and what is true now — then a cheap LLM synthesizes
an answer with mandatory citations.

**It cannot trade.** No module under `src/trading/copilot/` imports the
broker; every SQLite file is opened read-only; the LLM has no tools,
only quoted evidence. A test (`tests/copilot/`) fails the build if
anyone ever wires an execution import in. Order drafting/execution
remains a separate, deterministic, explicitly confirmed system.

## Commands

| Command | What it does |
|---|---|
| `/ask <question>` | Free-form: "why are we so heavy in semis?", "did the committee ever discuss energy?" |
| `/why SYM` | Thesis, votes, dissent, execution, and aftermath for a symbol |
| `/thesis SYM` | Latest thesis + invalidation conditions + is it still valid now |
| `/committee SYM` | Decision history for a symbol (bare `/committee` still convenes the live committee) |

Answers are structured **THEN** (what the committee believed, cited as
`D<id>`/`T<id>`) / **NOW** (positions, orders `trd-…`, data timestamps)
/ **CHANGED · UNCERTAIN**. If the journal has no matching decision, the
copilot says so — it never invents a rationale, and in that case it
doesn't even call the LLM.

## Setup

Default provider is **Anthropic Haiku** and reuses the
`ANTHROPIC_API_KEY` already in the VPS `.env` — zero setup. To switch:

```
COPILOT_PROVIDER=qwen        # or deepseek | anthropic
DASHSCOPE_API_KEY=...        # for qwen (Alibaba Model Studio)
DEEPSEEK_API_KEY=...         # for deepseek
# COPILOT_MODEL=qwen-plus    # optional override
# COPILOT_BASE_URL=...       # optional endpoint override (e.g. DashScope CN region)
```

Deploy = normal image rebuild (`docker compose build trader && docker
compose up -d --force-recreate bot`). No schema migrations: the copilot
derives `state/copilot.db` from the memory journal on first use and
keeps it current incrementally.

## Security & reliability properties

- Authorized chat only (existing bot gate: `TELEGRAM_CHAT_ID`).
- Rate limit: one LLM call / 15s; floods get "cooling down", not spend.
- Context cap ~14k chars; request timeout 30s; provider failure returns
  a plain error message, never crashes the poll loop.
- Audit log: every question, its evidence ids and sizes append to
  `state/copilot_audit.jsonl`.
- Nothing secret goes to the model: evidence is journal text, order
  rows, position numbers and price timestamps — no tokens, keys,
  usernames or account ids.
- Transcripts are untrusted DATA: the charter instructs the model that
  instruction-shaped sentences inside past agent chatter are quotes,
  never commands — and even a fully jailbroken copilot has no
  order-capable tool to misuse.

## Known Phase-1 limits (by design)

- It only knows what's journaled: committee rulings, agent takes, PM
  runs. Pure momentum-cycle rebalances are mechanical and have no
  thesis on record — the copilot says exactly that.
- Retrieval is SQLite FTS5 + symbol/date filters. No vector DB until
  the corpus outgrows keyword search (revisit if answers start missing
  obviously-relevant decisions).
- Market data comes from the local parquet cache (no live quotes) and
  is cited with its data timestamp.
