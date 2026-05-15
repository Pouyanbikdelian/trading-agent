# Equity research agent — architecture (not yet built)

This is the design for the autonomous Claude-driven research layer you
asked about. **Nothing here is implemented yet** — it's the blueprint
so you can decide what to build first.

## What it would do

The research agent runs on its own schedule (hourly, daily — separate
from the trading cycle). Its job is to surface trade *ideas*, not to
trade. Each idea is a structured proposal the trading cycle reads on
its next bar:

```python
TradeIdea(
    symbol="NVDA",
    direction="long",
    horizon_days=21,
    confidence=0.72,
    rationale="...analyst-quality reasoning here...",
    source_evidence=[...]  # news headlines, filings, technical signals
)
```

The trading cycle then decides whether to **modulate** an existing
algorithmic signal (boost a name we already hold, fade one we don't)
or **inject** a new position into the satellite sleeve. The risk
manager still gates everything.

## Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  watcher                — pulls news / filings continuously     │
│  filters                — drops noise, dedupes, ranks novelty   │
│  contextualizer         — assembles per-symbol dossiers         │
│  llm                    — Claude reads dossier → TradeIdea      │
│  scorer                 — historical hit-rate per idea bucket   │
│  publisher              — writes ideas to runner_store          │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Watcher

Sources (in order of cost / value):

* **yfinance.Ticker.news** — free, basic. Already wired in
  `reporting/news.py`.
* **SEC EDGAR full-text + RSS** — free, official. 8-Ks land within
  minutes of filing. Use `requests` + `feedparser`.
* **Reuters / FT / Bloomberg RSS** — free for headlines; the actual
  articles require subscriptions.
* **X firehose / Bloomberg Terminal API** — paid, real money.
* **Earnings calendars** (Tiingo / IEX) — $10-30/mo paid.

### 2. Filters

* **Novelty**: keep only items not already in the dossier from previous runs.
* **Relevance**: regex match against the active universe + the core
  sleeve names. Drop everything else.
* **Source quality**: weight by historical signal quality (Reuters >
  random aggregator).
* **De-duplication**: same-event coverage from different outlets;
  cluster headlines by 24-hour windows + symbol intersection.

### 3. Contextualizer

For each shortlisted symbol, assemble:

* Last 5 quarterly EPS surprises + revenue
* Trailing 30-day price action vs sector ETF
* Recent options-implied vol (if we have IBKR options data)
* Top 3 news clusters in the last 7 days, with sentiment
* Insider transaction summary (EDGAR Form 4)

That's the *dossier* — a few hundred lines of structured data the LLM
will read.

### 4. LLM

Claude Sonnet 4.6 by default. Prompt structure:

```
SYSTEM: You are a sceptical equity analyst. You are given a dossier
on ONE company. Decide whether there is an actionable trade idea in
the next 1-30 days, and at what confidence (0-1). You may decline.
You may NOT invent facts; cite the dossier line numbers.

USER: <dossier-as-markdown>

Respond as a JSON object with fields:
{
  "direction": "long" | "short" | "none",
  "horizon_days": int,
  "confidence": float,
  "rationale": str,  // <= 300 words
  "cited_evidence": [list of dossier line refs]
}
```

Cost back-of-envelope: 50 names per day, 5K tokens per dossier (Sonnet
$3/M in, $15/M out, 1K out) = $0.95/day = $350/year. Cheap.

### 5. Scorer

Track every idea's **realized** subsequent return (compare to the
benchmark) and build a rolling hit-rate per idea bucket
(by confidence band, by source, by sector). Use this to:

* Auto-throttle low-quality buckets (e.g. if "long-confidence-0.6"
  ideas have a 40% hit rate over 60 days, weight them at half).
* Surface meta-stats in the daily report ("you've taken 14 LLM ideas
  this month; hit rate 0.57, average alpha +1.2%").

### 6. Publisher

Writes ideas to a new SQLite table `trade_ideas` next to the existing
runner.db. The runner cycle reads this at each tick and integrates
into the satellite signal as a *modulator* (multiplier on the
algorithm's weight) — never as a unilateral order.

## What I'd build first

In priority order:

1. **Watcher v0** — yfinance + SEC EDGAR + Reuters RSS, dumping
   structured items into `state/news.db`. ~2 days.
2. **Contextualizer + LLM caller** — assemble dossiers for the held
   symbols and emit `TradeIdea` records. Use Claude API. ~3 days.
3. **Publisher → cycle integration as a confidence modulator** —
   wire ideas into the cycle's weight pipeline. ~1 day.
4. **Scorer** — needs ≥ 60 days of historical ideas to be meaningful;
   build the table earlier, the metric layer later.

Total: ~1 week to a usable v0. The scorer matures over 2-3 months as
real ideas accumulate.

## Hard rules (CLAUDE.md still applies)

* The research agent **cannot** flip live-trading flags or bypass the
  risk manager.
* Every LLM-generated idea must be **persisted** before action so a
  post-mortem can reconstruct what we knew when.
* The runner stays the source of truth on whether to trade — the
  agent is a *signal*, not a switch.
