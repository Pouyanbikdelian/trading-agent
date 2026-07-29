# Investment Committee Copilot — read-only

Ask the Telegram bot *why* the system did things, and whether old theses
still hold. The copilot retrieves what the committee actually said
(journaled rulings + per-agent takes), what actually happened (orders,
fills, positions), and what is true now — then a cheap LLM synthesizes
an answer with mandatory citations.

Since 2026-07-29 it is also **conversational**: it remembers the last few
turns, sizes its answers to the question, knows what it cannot see, can be
replied-to, and records the operator's pushback and standing instructions
for the agents to read. See "Conversation layer" below.

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
| `/forget` | Drop the conversation thread and start a fresh topic (`/newtopic`) |
| `/mandates` | List standing instructions; `/mandates drop M123`, `/mandates clear` |
| `/harden M123` · `/soften M123` | Re-grade a standing instruction whose tone was read wrongly |

You don't need a command: **plain text goes to the copilot**, a near-miss
typo gets a suggestion, and a forward-looking instruction is stored as a
mandate instead of answered.

Answers use the full **THEN** (what the committee believed, cited as
`D<id>`/`T<id>`) / **NOW** (positions, orders `trd-…`, data timestamps) /
**CHANGED · UNCERTAIN** structure only when the question deserves it —
see "Answer sizing". If the journal has no matching decision, the copilot
says so — it never invents a rationale, and in that case it doesn't even
call the LLM.

## Conversation layer

**Thread memory** (`copilot/thread.py`). The last `MAX_TURNS` turns are
persisted to `state/copilot_thread.json` and fed to the prompt, so "and
XLE?" or "why not?" resolves against what came before. The last symbol
carries forward, which is what makes a bare follow-up answerable at all.
Three limits, each deliberate: the window is short (the evidence should
get the context room), it expires after 45 minutes (a morning question
must not inherit last night's argument), and **turns are never evidence** —
the charter forbids citing anything the copilot itself said, and states
that when a turn contradicts the evidence the evidence wins.

**Answer sizing** (`engine.answer_budget`). Length is computed in code
from the question's shape and passed as `answer_budget`, a hard limit in
the charter. "What is XLV?" → 1-2 sentences; "why did we buy MU and what
was the dissent?" → up to 6. Prose alone did not hold: a five-word
question once drew four sentences of positioning thesis across five
decision ids — correct content, wrong size.

**Capability manifest** (charter rule 8). The charter enumerates what the
copilot has (positions, orders, journal, PM book, risk state, objections,
mandates, cached closes) and what it does not (live/intraday quotes,
fundamentals, analyst ratings, corporate actions, untouched symbols,
anything after the shown timestamps). If a question needs something on
the second list it must name the missing piece and stop. Substituting
general knowledge is called out as the one unrecoverable failure — an
answer from memory looks identical to an answer from evidence, and the
operator cannot tell them apart.

**Reply-to-message.** Replying to any bot message passes its text as
`CHAT_operator_is_replying_to` and searches its symbols, so "why this
alert? elaborate" has a referent. Treated as quoted data, like a
transcript.

## Operator objections and mandates

Two things the operator says are worth more than the scrollback, so both
are journaled and put in front of the agents.

**Objections** (`thread.py`) — reactive pushback: "why not XLE instead?",
"too much semis". Detected by deterministic marker matching (a miss is
stored as ordinary `chat`; a false positive would put words in the
operator's mouth in front of the committee, so precision wins). Journaled
as `operator_objection`.

**Mandates** (`copilot/mandates.py`) — forward-looking instructions for
the next run: "high conviction on GS, look at it next round". Strength is
graded from tone, strongest marker wins, unrecognised phrasing defaults
`soft`:

| Strength | Phrasing | Agent behaviour |
|---|---|---|
| `strong` | "I want X", "buy X", "make sure we hold X" | Act on it unless there is a concrete, stateable reason not to; if declining, say why |
| `medium` | "high conviction on X", "highly consider X" | Weigh seriously and address it; may decline with a brief reason |
| `soft` | "consider X", "might be worth a look" | Consider it, drop it freely |

Grading is regex, not an LLM call: the operator must be able to predict
how a phrase reads, and a model that silently re-grades tone between runs
is not predictable. Every capture **echoes back the strength it was given**
with `/harden` / `/soften` to correct it — the moment to catch a misread
is while the operator is still looking at the screen. Mandates expire
after `DEFAULT_TTL_DAYS` (14) and are journaled as `operator_mandate`.

Both reach the committee (`manager`, `risk_officer`, `position_coach`) and
the PM via `agents.context.build_context`. The `creative` voice is
deliberately excluded — objections and mandates name held symbols, and
leaking them would break its position-blindness.

**Neither is an order, and neither lifts a risk limit.** A mandate is
context for the PM's decision; the weights it produces still pass through
`agents.pm._clamp_weights`, so the per-name caps (10% single stock, 25%
ETF), the 50% cluster cap and the gross cap all still bind — "I want 40%
GS" comes out at 10%. `/hold` pins still outrank mandates, and an
off-universe ticker is still dropped to cash. The manager charter also
says explicitly that deferring to the operator when the data contradicts
him is the failure mode to avoid, and that a ruling going against a live
objection must name it in `dissent_summary`.

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
  `state/copilot_audit.jsonl`. Every exchange is additionally journaled
  (`chat` or `operator_objection`), so the conversation is reviewable
  outside Telegram.
- Mandate capture and typo matching are pure local code — a standing
  instruction or a mistyped command costs no LLM call.
- Conversation continuity does not depend on the memory store: if the
  journal is unwritable the thread still updates and the answer still
  lands.
- Nothing secret goes to the model: evidence is journal text, order
  rows, position numbers and price timestamps — no tokens, keys,
  usernames or account ids.
- Transcripts are untrusted DATA: the charter instructs the model that
  instruction-shaped sentences inside past agent chatter are quotes,
  never commands — and even a fully jailbroken copilot has no
  order-capable tool to misuse.

## Known limits (by design)

- It only knows what's journaled: committee rulings, agent takes, PM
  runs. Pure momentum-cycle rebalances are mechanical and have no
  thesis on record — the copilot says exactly that.
- Retrieval is SQLite FTS5 + symbol/date filters. No vector DB until
  the corpus outgrows keyword search (revisit if answers start missing
  obviously-relevant decisions).
- Market data comes from the local parquet cache (no live quotes) and
  is cited with its data timestamp.
- **"Why X and not Y" is only answerable for agent-chosen names.** The
  committee and PM write prose rationale, so their picks are explicable.
  The momentum book picks by ranking, and the ranked candidate ladder is
  computed per cycle, shown in the approval prompt, then discarded — so
  "JPM ranked 6th at +9.2%, GS 23rd at +4.1%" is unavailable after the
  fact. Journaling that ladder is the open fix.
- Objection/mandate symbol tagging uses known symbols (positions, orders)
  plus the PM universe. A name outside all of those is stored as text
  with no symbol tag.
- **Objections and mandates are not yet graded.** The journal has the
  data to ask "did the operator's overrides help or hurt?" at a 21-day
  horizon via the existing `predictions`/`source_trust` machinery, but
  the scorer isn't built. Until it is, there is no feedback loop on the
  operator's own calls.
