# Concept: Multi-Agent Committee with Permanent Memory

*Drafted 2026-06-11 from Yan's requirements + the HedgeAgents paper
(Li et al., WWW '25). Status: concept — Phase 1 build pending.*

## Requirements (Yan, verbatim intent)

1. Multi-agent "portfolio manager" committee in the spirit of HedgeAgents.
2. **Permanent memory**: never forgets, learns from experience, can compare
   today against anything it lived through — years ahead.
3. **Human-psychology sense**: live awareness of geopolitics (wars, rate
   hikes), central-banker personalities and tendencies, what the crowd
   anticipates / what is already priced in.
4. Analyst ratings as an additional input.
5. **Position awareness**: knows where each position was bought on the
   chart (top vs dip, entry vs 52-week range) and reasons about it.
6. Every new learning recorded and *mapped* to prior learnings.

## Non-negotiable constraints (from CLAUDE.md)

* LLM agents NEVER construct orders. They emit takes/weights; the existing
  RiskManager remains the only path to orders. Kill switches unchanged.
* Phased autonomy: advisory → approval-gated proposals → (maybe) autonomy.
  Promotion requires a scored track record, not vibes.

## The agents

| Agent | Inputs | Daily output |
|---|---|---|
| **Quant** | existing signals: momentum ranks, regime, vol surface, macro dial | structured take: bullish/bearish per sleeve + confidence |
| **Narrator** | news feed, macro calendar, World State dossiers | what changed in the story; what's priced in vs not; positioning read |
| **Street** | analyst ratings, price targets, revision momentum (yfinance) | where consensus disagrees with price; crowded vs hated names |
| **Position Coach** | our entries vs 52w range/ATH/drawdown, unrealized P&L, holding age | per-position read: "bought 8% off ATH, now -12%, thesis intact?" |
| **Risk Officer** | existing monitors (VIX, options surface, macro channels) | veto-voice: what could hurt us this week |
| **Fund Manager** | all takes + each agent's historical calibration | proposed sleeve weights + reasons → approval gate |

Every output is **structured JSON with an explicit, gradeable prediction**
(direction, horizon, confidence). Prose without a falsifiable claim is
rejected at parse time — this is what makes memory scoreable.

## The memory spine (the heart of this concept)

Five stores, one principle: **append-only, versioned, backed up; nothing
is ever deleted — things are only superseded, with links.**

1. **Journal** (`memory/journal.db`, SQLite, append-only)
   Every event: agent takes, manager proposals, approvals, orders, fills,
   halts, news digests, conference transcripts. Raw and immutable.
   This is the system's "lived life". Nightly backup off-VPS.

2. **Episodes** — one record per closed trade or notable market event.
   Snapshot of full context AT THE TIME: entry/exit price + percentile in
   52w range (the "did we buy the top or the dip" field), regime label,
   macro dial, VIX, which agents argued what, holding period, outcome in
   R-multiples. Episodes are the unit of comparison across years:
   "find me past episodes most similar to today" is an embedding +
   structured filter query.

3. **Lessons** — distilled beliefs. Each lesson is a card:
   - statement ("buying semis >90th percentile of 52w range after a 3σ
     rates shock has been -EV for us")
   - provenance: links to the episodes that created it
   - confidence + support/contradict counters, auto-updated whenever a
     new episode matches the lesson's pattern
   - status: candidate → established → retired (never deleted; a retired
     lesson keeps its history and the reason it died)
   Lessons are embedded for retrieval; agents must cite lesson IDs when
   they use them, so every decision is traceable to experience.

4. **World State** — living dossiers, maintained by the Narrator:
   one file per storyline (e.g. `iran_conflict.md`, `fomc_chair.md`,
   `japan_rates.md`), each with: current status, timeline of updates,
   market sensitivity notes, and "what the crowd expects" — updated when
   news shifts, every change timestamped and kept. This is the
   psychology/anticipation layer: the gap between dossier expectations
   and outcomes is itself a gradeable prediction.

5. **Scorecard** — every prediction by every agent is logged with its
   horizon and auto-graded when the horizon arrives. Produces per-agent
   and per-lesson calibration (hit rate, Brier score). The Fund Manager
   weighs agents by THIS, not by eloquence. This is how "skills are
   never lost": skill is a number attached to a memory, not a vibe.

**Consolidation (the learning loop).** Weekly job — the Experience
Sharing Conference: cluster the week's episodes, match against existing
lessons (embedding similarity), update counters, propose new candidate
lessons, flag contradicted ones for retirement review. Monthly: the
Manager's allocation conference reads the freshest calibration table.
Quarterly: a "memory audit" digest to Telegram — top lessons by evidence,
recently retired, biggest open contradictions.

**Durability**: journal + memory stores live in `state/memory/` (volume),
nightly `sqlite .backup` shipped off-box (restic → any S3-compatible
bucket, ~$1/mo). Schema versioned with idempotent migrations like
runner.db. Embeddings are recomputable from raw text, so the *text* is
the canonical asset.

## Decision flow (one trading day)

1. Morning: each agent reads inputs + retrieves K similar episodes/lessons
   → writes its take (logged to Journal + Scorecard).
2. Manager synthesizes → proposed sleeve weights + cited lessons →
   Telegram with `/approve` workflow (Phase 2+).
3. Approved weights → existing RiskManager → orders → fills logged.
4. Evening: outcomes appended; closed positions become Episodes.
5. Weekly/monthly/quarterly consolidation as above.

## Extreme Market Conference (from the paper, kept)

Trigger: portfolio -2% day (the existing kill-switch alert) or index
±3σ move. All agents convene immediately: Position Coach lays out the
book, Narrator explains the story, Risk Officer proposes de-risking
options, Manager sends you ONE message with 2-3 options and the lessons
that apply. You decide via Telegram.

## Phases & promotion criteria

* **Phase 1 (build ~2-3 weeks, run ≥6 weeks)**: agents + memory spine,
  advisory only. Cost ~$10-30/mo LLM. Promotion gate: Manager's daily
  directional take beats coin-flip with ≥60% calibration-weighted hit
  rate AND the memory retrieval demonstrably surfaces relevant episodes.
* **Phase 2**: Manager proposals enter the REQUIRE_CYCLE_APPROVAL flow;
  paper only. Promotion gate: ≥3 months where approved-as-proposed
  beats the mechanical baseline after costs.
* **Phase 3**: limited autonomy on a capped sleeve (e.g. 20% of equity),
  paper → live only after the standard 30-day paper rule.

## Honest risks

* LLM verbosity ≠ insight; the Scorecard exists to catch this.
* Narrative agents are most useful in macro-driven markets and nearly
  useless in quiet tapes — expect long stretches of "no edge".
* Memory can entrench bad lessons learned from small samples; the
  support/contradict counters and retirement reviews are the immune
  system, but the operator should read the quarterly audits.
* Cost discipline: frontier model for the Manager + conferences, cheap
  model for daily agent chatter.

---

## Addendum (2026-06-11, second pass with Yan)

### Source-trust ledger ("learn who to trust, still hear the gossip")

Every ingested news/claim is logged with its **source**. When a dossier
expectation or agent prediction is later graded, the sources that
informed it get credited or debited. Each source accumulates a trust
score (Bayesian beta prior, so new sources start neutral and converge
with evidence). Retrieval never drops low-trust sources — it *labels*
them: agents see `[trust 0.81] Reuters: ...` next to `[gossip 0.34]
@fintwit_anon: ...` and must weigh accordingly. Gossip that keeps being
early-and-right earns its way up; prestige outlets that lag decay down.
The trust table is itself memory: inspectable, versioned, never reset.

### Debate protocol (agents challenge each other)

Two roles added; two requests already covered:

* **Trader** (new) — tactical voice: entries/exits, what the tape says
  this week, liquidity and timing. Short horizon by charter.
* **Challenger** (new) — professionally disagreeable red team. Charter:
  attack the strongest current consensus take, surface base rates and
  past episodes where similar confidence failed, steelman the opposite
  position. Scored not on direction but on *whether its objections
  predicted the misses*.
* Pure strategist → covered by Quant + Narrator. Neutral arbiter →
  covered by the Fund Manager (explicitly neutral charter).

Protocol per cycle: (1) specialists post takes → (2) Challenger files
objections against the 2 highest-conviction takes → (3) authors reply
once → (4) Manager rules, recording the **disagreement index** (spread
of agent views). Disagreement is stored with the episode: over years we
learn whether committee discord is itself a risk signal. Full debate
transcripts go to the Journal.

### Tooling decisions

* **Memory format**: SQLite (journal/scorecard) + **markdown dossiers
  and lesson cards** in a folder — deliberately Obsidian-compatible.
  Yan can open `state/memory/` as an Obsidian vault and browse lessons
  with backlinks; no Obsidian dependency in code.
* Embeddings: sqlite-vec or plain numpy over local files; no external
  vector DB until the corpus outgrows it (years away).
* Orchestration: plain Python in this repo (matches house style); no
  LangChain/LangGraph dependency.

### Live dashboard (personal website)

Host on the EXISTING VPS — Caddy is already terminating TLS for n8n,
so this is a new docker service + one Caddyfile route + a subdomain
(only new cost: a domain if Yan wants a custom one). FastAPI serving a
static page + small JSON API that reads runner.db/state read-only.
Basic-auth via Caddy.

Panels (v1 → v2):
1. Equity P&L line (from `equity_curve`) + drawdown ribbon  [small]
2. Top positions with entry-vs-52w-range markers            [small]
3. Macro dial gauges (rates/dollar/energy/composite z)
4. Vol-surface tiles (ATM IV, skew, term slope, P/C OI)
5. Regime strip (HMM state + VIX bucket over time)
6. Agent room: latest takes, Challenger objections, Manager ruling,
   disagreement index sparkline
7. Scorecard leaderboard: per-agent calibration, per-source trust
8. Lessons feed: newest + recently reinforced/retired
9. Next events: rebalance countdown, FOMC/CPI calendar
