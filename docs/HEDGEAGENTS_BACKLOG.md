# HedgeAgents-inspired backlog (added 2026-07-11)

Source: "HedgeAgents: A Balanced-aware Multi-agent Financial Trading System"
(Li et al., WWW Companion '25, arXiv:2502.13165). Their headline numbers are
contamination-inflated (GPT-4 backtested on its own training window) — we are
NOT chasing their returns. We're stealing two coordination mechanics that fit
our architecture. Both are advisory-layer work: neither touches the order path
(rule #4 stands).

Do AFTER go-live wave 1 (GO_LIVE.md), not before.

---

## Item 1 — Extreme Market Review (their "EMC")

**What they do:** hard rule-based tripwire (>5% daily amplitude or >10%
cumulative 3-day amplitude on any managed asset) convenes an emergency
conference. The agent holding the losing book must present (a) current
holdings, (b) why it's losing, (c) a proposed plan. The other agents +
manager critique and the blended suggestions steer the next actions. Their
ablation: removing it raised MDD by ~72% and cut Sharpe ~20%.

**What we already have:** `runtime/sentinel.py` — 15-min RTH tripwires
(SPY ≤ −1.5% day, VIX ≥ +20% vs prior close, any held name ≤ −5%), LLM
severity judge, "alarm" auto-convenes committee, 2h debounce. Advisory only.

**The delta to build:**

1. **Per-position cumulative trigger.** Add a 3-day cumulative amplitude wire:
   any held symbol with |3-day return| ≥ 10% (their threshold; make it
   `SENTINEL_CUM3D_PCT`, default 0.10) trips even if no single day hit −5%.
   Catches slow bleeds the daily wire misses. Data: last 4 closes from the
   parquet cache — no new fetch path.
2. **Structured loss-review protocol** (the real value). Today an alarm
   convenes the committee with generic context. Instead, when the trigger is a
   *held position* (not SPY/VIX macro), build a "crisis brief" and run a
   dedicated review flow:
   - Brief = holdings + entry dates/prices, PnL of the crisis position,
     which sleeve owns it, the original committee ruling that led to the
     entry (from the journal), and the sentinel trigger details.
   - Round 1: position_coach (or PM if it's the PM sleeve) must answer three
     fixed questions: why is it down, is the entry thesis broken or intact,
     what's the plan (hold / trim / exit / hedge) with a size.
   - Round 2: risk_officer + challenger critique the plan; quant adds the
     numbers (vol, correlation to rest of book, distance to stop).
   - Output: a single structured verdict `{action, size, reason, dissent}`
     journaled + sent to Telegram. Advisory — Yan executes via existing
     commands (`/hold`, manual close) or ignores.
3. **Outcome tagging.** 5 trading days after each review, mark what happened
   (position PnL since verdict, was advice followed) into the journal so the
   historian can score whether crisis reviews add value. This is the feedback
   loop the paper doesn't have.

**Files touched:** `runtime/sentinel.py` (new wire + crisis brief),
`agents/committee.py` (review flow / prompt), journal schema (verdict +
outcome fields), `.env.example` (new threshold — additive, no defaults
changed), tests.

**Acceptance:** unit test with a synthetic 3-day −10% series trips the wire;
replay of a journaled historical alarm produces a verdict with all four
fields; Telegram message renders; no code path can emit an Order.

**Effort:** ~1 session. Cost: ~3–5 extra LLM calls per crisis event (rare —
their EMC fired 36 times in 3 quarters on crypto-heavy books; ours will be
far fewer on equities).

---

## Item 2 — Three-tier memory with distillation + retrieval (their MI/IR/GE)

**What they do:** three memory tiers — Market Information (raw daily
context), Investment Reflection (per-decision: what was decided, why, and
what happened), General Experience (distilled reusable lessons). At decision
time they embed a summarized query and retrieve top-k=5 similar past cases
across all tiers into the prompt. A periodic "Experience Sharing Conference"
forces each agent to nominate a representative case from its reflections and
the group distills it into a general lesson. Their ablation: removing
retrieval cost ~58% of ARR.

**What we already have:** memory store + historian (`agents/historian.py`,
Fridays 22:45 UTC, ≤2 candidate lessons/week, promotion to "established"
needs +3 net support votes). Committee journal of rulings.

**The delta to build:**

1. **Reflection tier (the missing middle).** Today lessons are distilled from
   the journal weekly, but there's no per-decision reflection record with an
   outcome attached. Add: every committee ruling and PM rebalance writes a
   `Reflection` row — `{ts, sleeve, decision_summary, thesis, market_snapshot,
   outcome: null}`. A weekday job (piggyback the 21:15 UTC mark-to-market)
   fills `outcome` after N=5 and N=21 trading days (PnL vs SPY of what was
   decided). SQLite alongside the existing store.
2. **Embedding retrieval into decision prompts.** Before each committee run /
   PM run, build a one-paragraph query from the current context (regime,
   top movers, pending question), embed it, retrieve top-k=5 nearest items
   from {reflections with outcomes, established lessons}, and inject as a
   "relevant past cases" block in the prompt — each case: situation → decision
   → outcome. Embeddings: local sentence-transformers or Anthropic-adjacent
   embedding via API — pick cheapest; store vectors in the same SQLite (a few
   thousand rows, brute-force cosine is fine, no vector DB).
3. **Upgrade historian into an ESC-style pass.** Keep cadence and promotion
   voting. Change the input: instead of scanning the raw journal, each sleeve
   nominates its single most instructive *closed* reflection of the week
   (biggest |outcome| or biggest surprise vs thesis), the historian runs one
   distillation call over the nominated cases + their outcomes, and writes
   candidate lessons that cite the source reflections (`source_ids`), so every
   lesson is traceable to real trades.
4. **Hygiene:** cap retrieval block at ~600 tokens; never retrieve
   open-outcome reflections into prompts (only closed cases and established
   lessons); log which memories were retrieved per run so we can audit
   whether they helped (compare committee calibration with/without — we
   already track calibration).

**Files touched:** `memory/` (reflection store + embeddings), `agents/
committee.py` + `agents/pm.py` (query build + retrieval block),
`agents/historian.py` (nomination flow), runner schedule (outcome filler),
tests with a fixture store.

**Acceptance:** reflection rows created on every ruling; outcomes filled after
the window; retrieval returns deterministic top-k on a fixture; prompts show
the block; historian lessons carry `source_ids`; full run adds ≤1 embedding
call + 0 extra LLM calls per cycle (retrieval is free; only distillation pays).

**Effort:** ~2 sessions (1: reflection tier + outcome filler; 2: retrieval +
historian rework).

---

## Explicitly NOT adopting

- Their per-asset "analyst personas" and 23-tool inventory — our 8-voice
  committee with sliced views already does this better.
- 30-day Budget Allocation Conference — arbitrary cadence; our combiner +
  vol-targeting is the principled version. (Only idea worth a look later:
  adding a CVaR term to the combiner objective. Parked.)
- Their performance claims as a benchmark — see contamination note above.
