# Roadmap: waves 2–4 (agreed with Yan, 2026-09-23)

Wave 1 (robustness) is on branch `claude/robustness-wave-1`. This file is the
plan for the rest. Every item that can change exposure follows hard rule 3:
backtest → walk-forward OOS → paper ≥ 30 days → live, sized down first.
Nothing here gives an LLM an order path (hard rule 8).

---

## Wave 2 — Faster only when needed, meetings only when needed

**Principle.** Speed goes to the mechanical and human paths, never to the LLM.
For a weekly/monthly momentum book, reacting faster to noise costs more than it
saves; speed pays in exits and in not missing a broken position.

### 2.1 Escalation ladder (`runtime/escalation.py`, new)

One state machine fed by monitors that already exist (sentinel wires,
options_monitor VIX term structure, macro_monitor dial, market_watch breadth,
held-name day/3-day moves) plus an earnings calendar for holdings.

| Level | Example triggers | What unlocks |
|---|---|---|
| CALM | default | weekly cycle; committee only if its inputs changed |
| WATCH | held name ±7% day or −10% over 3 days; VIX curve inverts; SPY −3% day; earnings ≤ 2 days for a holding | a *focused* committee with one agenda question; cooldown per trigger; daily token cap |
| ALERT | several WATCH triggers, or SPY −5% over 3 days | off-cycle **review-only** cycle + approval request to Yan |
| ACT | — | only pre-approved mechanical rules (guards, a pre-set mode) run without a human |

State in `state/escalation.json`; every transition journaled and graded later
("did escalating help?") so the ladder earns its thresholds.

### 2.2 Committee on demand

- **Agenda-driven convening.** A trigger produces a question ("NVDA −9% on
  guidance: hold, trim or exit?"), not a generic full review. Cheaper and less
  repetitive than the Mon/Fri full debate.
- **Skip when nothing changed.** Fingerprint the committee's inputs (book,
  ladder, monitor states, headlines hash); an unchanged fingerprint skips the
  scheduled meeting and says so.
- **Pre-earnings briefings** for held names (yfinance earnings dates).
- **Sunday check-in with Yan**: what changed, what awaits approval, agent
  scorecard, decisions needed. This is the "meeting" that includes the human.
- **Daily LLM budget.** Hard token/$ stop in `agents/llm.py`
  (`state/llm_usage.jsonl` already records usage; nothing reads it yet).

---

## Wave 3 — Pick the selector on evidence; make the backtest honest

### 3.1 Matched selection race (D1)

Three shadow books, same decision dates, same universe, CHF accounting, the
same costs and the same risk manager:

1. mechanical momentum (as configured live),
2. the LLM PM as today,
3. momentum with a PM **veto** (the PM may remove names with a stated reason,
   never add).

Daily mark, weekly scorecard, `/race` in Telegram. Measure net excess return,
turnover, drawdown, slippage, and the contribution of PM additions/removals.

### 3.2 PM veto mode (D1)

`AGENT_PM_MODE=select|veto`. In veto mode the PM's output is a list of
removals from the mechanical top-k, each with a falsifiable reason; weight is
redistributed by the mechanical rule. The charter rule "creative/scout ≥ 0.70
confidence ⇒ allocate ≥ 5%" is removed (the agents are currently worse
calibrated than a coin flip). Switching live from `select` to `veto` is Yan's
call, after veto mode has run on paper alongside the race.

### 3.3 Honest backtest (D4)

- **Point-in-time S&P 500 membership** from Wikipedia's change history (free).
  Delisted tickers have no free price history, so residual survivorship is
  measured and stated, not hidden.
- **Split/dividend-safe cache**: detect a re-based history (cached overlap ≠
  fresh overlap) and refresh the whole series.
- **Engine realism**: next-session fills, weight drift between rebalances, cash
  yield, CHF base + FX, the live constraints (10% name cap, 30% sector cap,
  long-only, pinned-slot reservation).
- **In-fold parameter selection** in walk-forward, and DSR with the true
  number of trials.
- **Re-verify the "winning" 126/21/63 config** under all of the above. Also
  settle what is live: compose passes `rebalance=${REBALANCE:-5}` (weekly).

---

## Wave 4 — Better ranking, a market-stretch cash dial, better signals

### 4.1 Ranker fixes (D2)

- **Monthly calendar rebalance** (`calendar_rebalance_months=1`) instead of a
  bar count that drifts with the rolling window.
- **Rank buffer**: buy the top k, hold until a name falls below rank ~20.
- **Commission-aware trading**: skip trades below a minimum size or whose
  expected benefit is below round-trip cost (dust filter).
- **Consistent sizing**: k and the per-name cap must fill the book (k=8 with a
  10% cap leaves ≥20% cash by accident); redistribute capped weight.
- **Sector cap picks the next-best name** instead of leaving the slot in cash.
- **Long-term winners**: Yan holds these manually; `/hold` already takes a
  position out of the desk's book, so the desk never sells it.

### 4.2 Market-stretch cash dial (Yan's "discount season" rule)

Yan's rule: hold more cash when the market is stretched at all-time highs
(typically 30–50% cash when really at the top, less if the book is well
diversified or hedged); be fully invested after a prolonged sell-off, once it
has stabilised (corrections −10–15%, bear markets −20%+, long grinds down).

The dial is `invested = f(stretch, discount, stabilisation) × book_risk_adj`,
capped at 1.0 (no leverage), applied as an overlay after the strategy weights:

- **Stretch** (raises cash): SPY distance above its 200-day average (z-score),
  breadth divergence (index at a high while % of members above their 200-day
  falls), VIX complacency. Being at an all-time high *alone* does not raise
  cash — markets spend long stretches at new highs, and a flat 30–50% cash
  rule at every high costs heavily in bull runs. Stretch + high does.
- **Discount** (lowers cash): drawdown depth from the high and its duration.
- **Stabilisation gate** (required before buying the discount): no new 20-day
  low for N days and the VIX curve back in contango. Buying while it is still
  falling is how discount buying goes wrong in 2008-style declines.
- **Book-risk adjustment**: a book with high effective number of bets, low
  average correlation or low beta needs less cash; a concentrated, correlated
  book needs more.

Levels are chosen on the honest backtest (wave 3), within Yan's range
(roughly 50–70% invested at a stretched top, 100% in a stabilised discount),
and shown to Yan as a trade-off before paper.

### 4.3 Better signals (D3)

Each is a research item through the full gate:

- **Residual momentum** (Blitz, Huij & Martens 2011): momentum of returns after
  removing market/sector exposure.
- **Smooth vs jumpy momentum** ("frog in the pan", Da, Gurun & Warachka 2014).
- **52-week-high proximity** (George & Hwang 2004).
- **Earnings awareness**: avoid buying just before earnings; post-earnings
  drift as a tiebreaker.
- **Momentum volatility scaling** (Barroso & Santa-Clara 2015).

---

## Open decisions for Yan

1. Approve deploying wave 1 (maintenance sequence, `docs/deploy.md` §15–16).
2. Confirm the live `REBALANCE` value in the VPS `.env`.
3. After the race and a paper run of veto mode: switch the PM to veto?
4. The cash-dial levels, once the backtest trade-off is in front of you.
