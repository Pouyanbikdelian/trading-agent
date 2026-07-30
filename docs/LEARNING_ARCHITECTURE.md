# Continuous learning architecture — proposal

*Drafted 2026-07-30 from Yan's brief: "learn continuously from decisions,
like a real trader — including the ideas we didn't act on; recognise when
a setting we've seen before comes back." Status: proposal. Nothing here is
built. Advisory layer only — none of it touches the order path.*

---

## The honest diagnosis first

The desk already grades three things: agent predictions (Brier, per-agent
calibration), source trust, and closed episodes. The historian distils
those into lessons weekly. That machinery is real and it works.

Four things are missing, and they are the reason the system does not
actually get smarter month over month:

1. **It only grades what it did.** Every candidate the PM ranked below the
   cut, every name the committee argued about and dropped, every mandate
   the desk declined — none of it is scored. The system therefore cannot
   answer the only question that establishes whether the selection process
   has edge: *did our picks beat our rejections?* Right now the ranked
   candidate ladder is written into `cycle_approval_pending.json` and
   deleted ~10 minutes later (already noted in TODO.md). That is the most
   valuable discarded data in the system.

2. **Lessons are unconditioned.** A lesson is a bare statement with tags.
   There is no record of *what the world looked like when it was true*, so
   retrieval is "show all established lessons" rather than "show the
   lessons that applied the last time conditions looked like today". Yan's
   own examples are all conditional statements — "in **this setting**,
   markets were selective of high-quality names", "with **this macro and
   geopolitics**, crowded trades were punished". An unconditioned lesson
   book cannot express those, so the historian is structurally forced to
   produce vaguer lessons than the operator wants.

3. **A trade is graded as one number.** `pnl_pct` conflates three
   independent judgements: was the thesis right, was the timing right, was
   the size right. Most losing trades in a trend book are right-thesis /
   wrong-timing, and most disappointing winners are right-thesis /
   right-timing / too small. Collapsed into one P&L figure, none of that
   is learnable.

4. **Calibration is measured but not used.** `mem.calibration()` produces a
   per-agent Brier score, the manager is shown it, and then nothing
   mechanical happens. Agent influence is unrelated to agent accuracy.
   **Adding more agents before fixing this makes the system worse, not
   better** — an unweighted committee converts extra voices into noise.

---

## 1. The shadow book (counterfactual ledger)

**The single highest-value addition.** Free alpha measurement: no capital
at risk, no new data source, and it is the only way to find out whether
any of the rest of this machinery is working.

New table:

```sql
CREATE TABLE IF NOT EXISTS shadow (
    id           TEXT PRIMARY KEY,       -- sh-<uuid8>
    ts           REAL NOT NULL,
    symbol       TEXT NOT NULL,
    origin       TEXT NOT NULL,          -- ladder|committee|mandate|scout|operator
    disposition  TEXT NOT NULL,          -- taken|passed|declined|cut_by_risk|cut_by_cap
    rank         INTEGER,                -- position in the ranked ladder, if any
    score        REAL,                   -- the ranking score at the time
    why          TEXT NOT NULL DEFAULT '',
    conditions   TEXT NOT NULL DEFAULT '{}',  -- JSON regime fingerprint (see §2)
    px_at        REAL,
    r5           REAL, r21 REAL, r63 REAL,    -- forward returns, filled by the grader
    bench5       REAL, bench21 REAL, bench63 REAL,  -- SPY over the same window
    graded_ts    REAL
);
```

Written at three points, all cheap:

- **The ladder** — `_compute_top_candidates` already builds the ranked
  list with scores. Write every row, not just the ones above the cut,
  with `disposition = taken|passed`.
- **The committee** — every symbol named in a take that did not end up in
  the book, `disposition = passed`, `why` = the manager's one-line reason.
- **Risk cuts** — when `_clamp_weights` or the cluster cap trims a
  position, log what was wanted vs what was allowed. This is how you find
  out what the caps cost, which is the only honest way to argue for or
  against changing them.

Graded by the existing nightly grader, which already reads cached closes.
The output is a monthly scorecard the desk has never had:

```
Selection edge, trailing 63d      picks   passes   spread
  momentum ladder (n=118)        +4.1%    +1.2%    +2.9%   ✓ works
  committee names  (n=31)        +2.0%    +3.8%    -1.8%   ✗ negative
  risk-cut names   (n=9)         +1.1%    +9.4%    -8.3%   caps cost 8.3%
  operator mandates(n=6)         +6.2%    n/a              (see below)
```

The third line is the one that changes behaviour. So is the fourth: it
closes the loop on Yan's own calls, currently the one participant exempt
from calibration (also in TODO.md).

**Design honesty note:** forward return alone is not edge — a shadow book
of passed names will look great in a rising market. Every comparison must
be against SPY over the identical window (`bench*` columns) and reported
with n. A spread with n=9 is a story, not a result.

---

## 2. Regime fingerprints, and lessons that are indexed by them

This is what makes "I've seen this setting before" mechanically possible
rather than a phrase in a prompt.

Define one **fingerprint** function, computed once per cycle from data the
system already has, and stamped onto every episode, shadow row, decision
and lesson:

```python
{
  "hmm": "bear|chop|bull",              # regime/hmm.py, already exists
  "vix_bucket": "low|normal|elevated|stress",
  "term": "contango|flat|backwardation",  # VIX term structure
  "breadth": "narrow|mixed|broad",      # % of universe above 200d
  "dispersion": "low|normal|high",      # cross-sectional return stdev
  "rates": "easing|neutral|tightening", # macro dial, already exists
  "trend_age_d": 143,                   # days since the last regime flip
  "crowding": 0.62,                     # book ENB / naive N, already computed
}
```

Everything here is already available in `agents/context.py` or
`regime/`. This is a ~60-line assembly function, not new research.

Then:

- **`lessons` gains `conditions TEXT`** — the fingerprint at the time the
  lesson was written, plus a `applies_when` clause the historian must fill.
- **Retrieval changes from "all established" to "established, ranked by
  fingerprint distance from today"**. The committee prompt gets the five
  most *relevant* lessons instead of the N most recent.
- **The historian's charter changes** to require a condition clause:
  a lesson without one is rejected at parse time, exactly as a take
  without a falsifiable prediction is rejected today.

The lesson body already demands four sentences, one of which is "under
what market conditions it applies" — but that sentence is prose nobody
can query. The fingerprint makes it a key.

**Analogue retrieval** falls out for free: nearest-neighbour over stored
fingerprints answers "the closest 20 historical windows to today, and
what worked in them" — requirement #2 in `concept_multiagent_memory.md`,
still unbuilt after a year.

---

## 3. Decision reviews — three verdicts, not one

New pass, run at 5d and 21d after every decision (not just closed
positions), producing a structured `reflection` row:

| verdict | question | why separate |
|---|---|---|
| **thesis** | did the stated reason turn out to be true? | a right thesis that lost money is a timing/sizing lesson, not a research lesson |
| **timing** | would waiting/acting earlier have been better? | the only way to learn entry discipline |
| **sizing** | was the position big enough given how it went? | the most common silent error in a capped book |

Cheap model, one call per decision, strictly structured output. These are
the historian's input — it distils from *reflections*, not from raw
journal rows. That is the three-tier memory already specced in
`docs/HEDGEAGENTS_BACKLOG.md`; this section is the concrete shape of it.

The pre-mortem is the mirror image and belongs at decision time — see §4.

---

## 4. Agents: what to add, and in what order

**Fix weighting before adding voices.** Weight each take in the manager's
synthesis by that agent's Brier score on a shrunk estimate (an agent with
n<20 graded predictions gets prior weight, not its raw score). Half a day
of work; it makes every subsequent agent addition strictly beneficial
instead of dilutive.

Then, in value order:

1. **Pre-mortem** (decision time, cheap model). "It is three months from
   now and this position is down 20%. Write the post-mortem." Forces the
   failure mode into the record *before* the trade, which makes the later
   reflection gradeable against something. Cheapest high-value voice in
   the literature and it is not in the committee today.

2. **Analogue finder** (needs §2). Not an opinion — a retrieval: "today's
   fingerprint is closest to 2018-11, 2015-08, 2021-09; in those windows
   momentum-continuation failed and quality outperformed". Answers Yan's
   "in this setting" examples directly.

3. **Crowding / positioning** (deterministic first). Yan's "crowded
   trades were punished" example. Start as scanners writing to
   `state/anomalies.json` — the demoted-Mathematician approach already
   agreed in TODO Phase 12 — and only promote it to a voice if the
   journaled findings grade well over ~2 months.

4. **Timing/trigger scanner** (deterministic, never an LLM). "Markets
   bottomed when this signal popped" is a quantitative claim and must be
   measured, not narrated: breadth thrusts, VIX term inversions clearing,
   put/call extremes. Every finding reports effect size, n, **and how many
   things were tested** — a scanner that hides its search breadth is a
   p-hacking machine.

Explicitly **not** recommended: more generalist commentary voices. The
committee has eight and the marginal one adds correlated prose.

---

## 5. Swarm agents and cheap models — where they actually belong

The honest read: swarm architectures are good at **breadth** and bad at
**judgement**. N cheap models voting is a confidence amplifier, not an
accuracy amplifier — they share training data, they share failure modes,
and their agreement is close to meaningless. So the rule is:

> **Cheap models fan out over data. Frontier models decide. Never let a
> cheap model be a voting voice in the committee.**

Concretely, good uses for a cheap tier (Kimi K2, DeepSeek, Haiku) — all
map-stage, all embarrassingly parallel, all currently not done at all:

| job | volume | why cheap is right |
|---|---|---|
| per-symbol screen across the full universe | ~500/day | mechanical extraction, verifiable output |
| news/headline triage into the narrator's slice | ~200/day | relevance filtering, not judgement |
| first-draft reflections (§3) | ~20/day | structured template, frontier reviews only outliers |
| transcript / filing summarisation into dossiers | bursty | quoted-data summarisation |
| shadow-book "why passed" one-liners | ~100/day | restating an existing reason |

The plumbing is already there: `agents/llm.py` has tier routing
(`tier='frontier'` vs default). Adding a `tier='cheap'` with its own
`AGENTS_MODEL_CHEAP` env var and base URL is a small change.

Two guards this needs on day one, because fan-out is how LLM bills go
non-linear: a **daily token budget** with a hard stop, and **skip-on-
budget-exhausted** degrading to no screen rather than to a partial one.

**Efficiency win, unrelated to swarms:** the eight specialists each get
their own slice today, but the shared context block is resent every call.
Prompt caching on the common prefix is a straightforward ~40–60% cut in
committee token cost.

---

## 6. Suggested build order

Each item is independently useful and independently abandonable.

| # | item | effort | why this order |
|---|---|---|---|
| 1 | Shadow book: ladder rows + nightly grading + `/edge` command | ~1 session | measures everything else; without it the rest is faith |
| 2 | Regime fingerprint function + stamp it on episodes/shadow/decisions | ~0.5 session | pure plumbing, no behaviour change, unblocks 3 and 5 |
| 3 | Calibration-weighted manager synthesis | ~0.5 session | makes later agents additive rather than dilutive |
| 4 | Reflections at 5d/21d, cheap tier | ~1 session | gives the historian real input |
| 5 | Conditioned lessons + analogue retrieval | ~1 session | the "I've seen this before" capability |
| 6 | Pre-mortem voice | ~0.5 session | cheap, and now gradeable against 4 |
| 7 | Cheap tier + universe screen, with budget cap | ~1 session | breadth; only worth it once 1 can measure whether it helps |

---

## What this will and will not do

It will make the system **explainable and measurable** — "our rejections
beat our picks by 1.8% over the last quarter" is a sentence the desk
cannot currently say, and it is the sentence that drives every real
improvement.

It will **not**, by itself, improve returns. Learning machinery is
instrumentation. The likely outcome of building §1 is discovering that
one or two parts of the current process have no edge, and the value is in
switching those off. That is a good outcome and it should be framed as
the expected one, so that a negative result does not get rationalised
away when it arrives.
