# Trading agent — system map

A to-the-point tour of how the system fits together and how a single
trading cycle flows from data to orders.

## The 30-second elevator pitch

```
   Market data ──► Strategy ──► Overlays ──► Risk manager ──► Broker (IBKR)
                  (weights)    (cash/derisk)   (orders)        (fills)
                                                                  │
   Operator ◄── Telegram bot ◄── Runner state (sqlite + halt.json)│
       │                                ▲                          │
       └────── /halt /resume ───────────┘                          │
                                                                   ▼
                                                              Account snapshot
```

The whole system is built around the rule **strategies never construct
orders**. They emit *target weights*. The risk manager is the only
component allowed to turn weights into orders, and it is hard-blocking
— if it refuses, nothing happens.

## Architecture, by package

```
src/trading/
├── core/         types, settings, clock, logging
├── data/         DataSource Protocol + adapters (yfinance, ccxt, ibkr) + Parquet cache
├── backtest/     vectorized engine, metrics, walk-forward harness
├── strategies/   Strategy interface + library
├── regime/       VIX classifier + HMM regime classifier
├── selection/    portfolio combination + overlays (vol_target, regime_derisk, dip_buy)
├── portfolio/    core-satellite framework
├── execution/    Broker Protocol + IBKR adapter + simulator
├── risk/         pre-trade limits + kill switches
├── runner/       APScheduler-based live loop
├── reporting/    daily/weekly markdown report + optional LLM summary
├── bot/          Telegram long-poll command bot + outbound notifier
└── cli.py        single Typer CLI; subcommands per phase
```

## What each piece does

| Package | Role | Reads | Writes |
|---|---|---|---|
| **data** | Cache market bars from yfinance/ccxt/IBKR. Parquet, partitioned by asset class + symbol + frequency. | Network APIs | `data/parquet/...` |
| **strategies** | Pure functions of prices → target weights. No state, no orders. | Prices DataFrame | — |
| **regime** | Classify the current market regime (e.g. VIX percentile, HMM state). | VIX series | — |
| **selection** | Combine multi-strategy weights, then apply overlays. Overlays only *scale* weights, never invent new positions. | Prices, VIX | — |
| **portfolio** | Core-satellite split (long-term core sleeve + tactical satellite). | YAML config | — |
| **execution** | Broker Protocol with two implementations: IBKR (`ib-async`) and a Simulator. | Market state | Orders to broker |
| **risk** | Pre-trade limit checks. Halts persist to disk. Halts the only way to flatten. | Snapshot, weights | `state/halt.json` |
| **runner** | The orchestrator. APScheduler-driven cron. One cycle = (fetch data → strategy → overlays → risk → submit → record). | Everything | `state/runner.db`, `state/orders.db`, `state/heartbeat.json` |
| **reporting** | Read the SQLite stores, produce a Markdown report. Optional LLM summary. | Stores | Markdown to stdout/file |
| **bot** | Long-poll Telegram bot. Reads state, writes halt.json. Sends alerts. | Token + chat ID + state | `state/halt.json` |
| **cli** | Typer entry point: `trading data fetch`, `trading backtest`, `trading paper`, `trading live`, `trading halt`, `trading bot run`, etc. | — | — |

## A single cycle, step by step

When APScheduler fires (e.g. every Friday at market close):

```
┌─────────────────────────────────────────────────────────────┐
│  1. Refresh data                                            │
│     • Read prices from ParquetCache                         │
│     • If a refresh hook is configured, pull latest bars     │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Generate target weights                                 │
│     • Each strategy.generate(prices) → wide DataFrame       │
│     • If N strategies, combiner aggregates                  │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Apply overlays (in order)                               │
│     • regime_derisk (cash conversion when SPY broken)       │
│     • vol_target    (scale to annualized vol target)        │
│     • dip_buy       (boost into pullbacks of held names)    │
│  Each overlay only *scales* weights; never adds positions.  │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Risk manager evaluates                                  │
│     • Refresh AccountSnapshot from broker                   │
│     • Check halt.json — if halted, refuse + optionally      │
│       force-flatten existing positions                      │
│     • Auto-halt if daily-loss or drawdown cap breached      │
│     • Convert target weights → Orders, sized within         │
│       max_position_pct and max_gross_exposure               │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Broker submission                                       │
│     • broker.submit(orders)                                 │
│     • Wait for fills, record to OrderStore (orders.db)      │
└───────────────────────┬─────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Persist + heartbeat                                     │
│     • Write AccountSnapshot to RunnerStore (runner.db)      │
│     • Atomic-write state/heartbeat.json                     │
│     • Optionally push a Telegram alert if anything notable  │
└─────────────────────────────────────────────────────────────┘
```

## Where things live on disk

```
/opt/trading-agent/
├── .env                 # secrets — gitignored, never leaves the VPS
├── docker-compose.yml   # IB Gateway + trader + Telegram bot containers
├── src/trading/         # source
├── config/              # universes.yaml, risk.yaml (editable)
├── data/parquet/        # cached market data
├── logs/                # runner + bot log files
└── state/
    ├── halt.json        # ← the kill switch
    ├── heartbeat.json   # ← runner liveness
    ├── runner.db        # ← account snapshots, cycle outcomes
    └── orders.db        # ← orders + fills history
```

## The hard kill — three paths in

```
                       ┌────────────────────┐
                       │  state/halt.json   │  ← the only thing the runner reads
                       │ {"halted": true,…} │
                       └────────▲───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
   ┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐
   │ Telegram /halt  │ │ trading halt CLI│ │  Auto-trigger   │
   │ (from phone)    │ │ (from terminal) │ │  (risk limits)  │
   └─────────────────┘ └─────────────────┘ └─────────────────┘
```

Since 2026-08-18 the auto-trigger is **continuous**: the runner's
60-second account snapshot evaluates the daily-loss and drawdown limits
on every tick, not just inside a cycle. Before that, a limit crossed on a
Tuesday afternoon waited for the Friday cron.

Once `halt.json` says halted, the next cycle:
1. Refuses any order that would **add or rotate** exposure
2. Still builds the real CHF basket, and if part of it provably *reduces*
   exposure, offers that reduce-only subset for approval
   (`ALLOW_DEFENSIVE_CYCLE_WHEN_HALTED`, default true)
3. Stays halted until you explicitly `/resume` or `trading resume` —
   a defensive reduction never clears the halt

## Halting is asymmetric

A halt blocks risk going **up**. It must never block risk coming **down**
— on 2026-08-18 it did, and two trailing-stop exits were refused while
the book sat fully exposed for four hours.

`trading/risk/reduce_only.py` holds the single proof of "this order
lowers exposure": there must be a live position of the opposite sign, the
side must be that position's exit side, no working broker order may point
the other way, and the quantity is clamped to the remaining headroom so
nothing can cross flat. Three callers share it — `/close`, `/flatten`,
and the cycle's defensive basket — so there is exactly one definition of
safe, and a fresh broker snapshot re-proves it immediately before submit.

What a halt still will not do is close anything by itself. Use `/close`
or `/flatten`, or approve the defensive basket.

**A halt does NOT close positions.** Step 2 above used to read "If
`flatten_on_next_cycle: true`, force-closes existing positions" — that
key was written by `/halt` and read by nothing, so the flatten never
happened while `/halt` replied "Next cycle will force-flatten positions".
Removed 2026-08-07; halting is reversible and liquidating is not, so one
word should not market-sell a book. Use `/flatten` to exit.

Note what that means: halting stops *new* trading, it does not reduce
exposure on its own. Neither do the daily-loss and drawdown kill
switches — they halt too. The position guards are the only thing that
exits automatically, and they are off by default; everything else that
lowers risk needs you to type `/close`, `/flatten`, or `/approve` on a
defensive basket.

**The daily baseline now has provenance.** `RiskManager` only accepts a
daily-loss baseline captured inside a five-minute window at the real NYSE
open (`runtime/nyse_session.py` reads the exchange calendar, so holidays
and early closes are handled). A baseline from the wrong session,
currency or time of day is refused, and cycles stay review/reduce-only
until the next open rather than measuring against a figure nobody can
vouch for. Practical consequence: **the runner has to be up at 13:30 UTC**
for that day to be executable.

The risk manager itself **never auto-unhalts**. By design — an
automatic recovery on a partially-understood failure is how money
disappears.

## Standing operator preferences — what binds and what merely advises

Three mechanisms, and the difference between them is the whole point.

| | scope | enforced by | expires |
|---|---|---|---|
| `/hold SYM` | freezes a name **both ways** — no buys, no sells | code | never |
| `/exclude SYM` | blocks every **automatic buy**; selling always works | code | never |
| a mandate ("avoid X from now on") | context for the agents | the model's judgement | positives after 14 days; **prohibitions never** |

`/exclude` was added 2026-08-19 after tracing what became of the
operator's *"I don't like PM (Philip Morris), don't buy it in the
future"*. It had been captured as a mandate: free text in an LLM prompt,
graded `strong` by a matcher that saw `\bbuy\b` and ignored the "don't"
in front of it, then pruned from the store fourteen days later with no
notice. Nothing in code ever consulted it, and the PM went on holding
`PM` at 7%.

`runner/exclusions.py` is the filter. It binds in `agents/pm.py`
(`_clamp_weights(blocked=held | excluded)`), on the cycle's order path,
on the custom-approval rebuild, and on the candidate scoreboard so
`/pick` cannot reintroduce a banned name. A manual `/buy` still works —
the operator typing an order now outranks a preference from last month —
and the bot's reply says the symbol is excluded so the contradiction is
visible rather than silent.

Mandates keep their advisory role, with the two defects fixed: polarity
is graded off its own ladder (so "don't buy X" is a prohibition, not a
strong buy), and a prohibition never expires. When one names a symbol the
bot now offers `/exclude SYM` in the same reply — the shortest path from
"I said it" to "it binds".

## `/baseline` — the drawdown switch's acknowledgement path

`equity_high_watermark` only ratchets upward; nothing lowers it. With a
tight `MAX_DRAWDOWN_PCT` that made the drawdown switch a **one-way
trapdoor**: once equity is far enough under the peak, every evaluation
halts, `/resume` re-halts on the next 60-second tick, and the only
documented escape — a new equity high — is unreachable, because a halted
book may only be reduced. That is a lockout, not a risk control.

`/baseline` shows the reference points. `/baseline reset [reason]`
re-stamps the high-water mark and the daily open to live equity. It is
deliberately manual, deliberately separate from `/resume` (accepting a
loss and restarting trading are two judgements), refuses a stale or
wrong-currency snapshot, and appends to `state/baseline_resets.jsonl`.
The daily baseline it writes is marked `operator_reset:<actor>` so it is
never mistaken for an opening-bell capture.

The intended sequence after a breach is therefore: look at the loss →
`/cycle` to see what the desk would do about it (reduce-only) →
`/baseline reset` to accept the new starting point → `/resume` → a normal
cycle with buys.

## The strategy that's actually shipping

Default config (in `.env.example`):

| Setting | Value |
|---|---|
| Universe | sp500 |
| Strategy | top_k_momentum |
| Rebalance | quarterly (cron: 4:05pm ET Friday, but only every 63 bars) |
| k (held names) | 15 |
| Lookback | 126 bars (~6 months) |
| Skip | 21 bars (excludes last month — short-term reversal noise) |
| Sizing | inverse-volatility weighted within the top-k |
| Position cap | 20% per name |
| Gross cap | 100% (no leverage) |

Backtest results (in-sample 2018-2026, then walk-forward OOS):

| Metric | In-sample | OOS |
|---|---|---|
| CAGR | 32.5% | 33.8% |
| Sharpe | 1.12 | 1.10 |
| MaxDD | -35% | -35% |
| vs QQQ | 2.4× the dollars, same drawdown |

## Defensive overlays

Three layers, applied to weights *after* the strategy decides:

| Overlay | Triggers when | Effect |
|---|---|---|
| **regime_derisk** | SPY closes below SMA(200) for 5 consecutive days | Scales gross to 30% (or 10% on death-cross) |
| **vol_target** | Realized portfolio vol exceeds target | Scales gross down toward target |
| **dip_buy** | Held position drops 5%+ from peak, still above SMA(200) | Boosts the weight (capped at max_per_position) |

None of these add new positions or short anything. They only adjust the
size of what the strategy already chose. This keeps the no-leverage
rule intact.

## Operator's daily / weekly loop

```
┌─ Friday 4:05pm ET ─────────────────────────────┐
│  Runner fires, executes one cycle              │
│  Sends weekly report to Telegram               │
└────────────────────────────────────────────────┘
        │
        ▼
┌─ You, anytime ────────────────────────────────┐
│  Telegram → /status      → see state          │
│             /positions   → see book           │
│             /report      → fresh weekly       │
│             /halt        → flatten everything │
│             /resume      → clear halt         │
└───────────────────────────────────────────────┘
```

The default report cadence is weekly — daily reports on a
quarterly-rebalance strategy are mostly noise.

## Deployment topology

```
                          ┌─────────────────────┐
                          │   Your iPhone       │
                          │   Telegram app      │
                          └──────────┬──────────┘
                                     │ long-poll over TLS
                                     ▼
┌──────────────────── VPS: 157.245.x.x ──────────────────┐
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Docker Compose                                  │    │
│  │  ┌────────────┐  ┌────────────┐  ┌───────────┐  │    │
│  │  │ ib-gateway │  │   trader   │  │    bot    │  │    │
│  │  │ (IBKR API) │◄─┤  (runner)  │  │ (telegram)│  │    │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘  │    │
│  │        │               │               │        │    │
│  │        └───── shared volumes ──────────┘        │    │
│  │          state/, logs/, data/                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  UFW firewall: only port 22 (SSH) open inbound          │
└──────────────────────────────┬───────────────────────────┘
                               │ outbound only
                               ▼
                  ┌────────────────────────┐
                  │   IBKR servers         │
                  │   (paper or live)      │
                  └────────────────────────┘
```

## Promotion path: research → paper → live

```
   research          paper                   live
   (no broker)       (simulated $)           (real $)
        │                 │                       │
        │  ≥ 30 days OOS  │                       │
        │  ─────────────► │                       │
        │                 │   only after          │
        │                 │   30 days clean       │
        │                 │   ──────────────────► │
```

The live runner is *only* enabled when both:
- `.env` has `TRADING_ENV=live`
- `.env` has `ALLOW_LIVE_TRADING=true`

`Settings.is_live_armed()` enforces this. The CLI refuses to launch
`trading live` otherwise.

## Convert this doc to a PDF

If you want a one-pager you can print or save:

```bash
# On macOS, requires pandoc + a LaTeX engine like basictex
brew install pandoc basictex
pandoc docs/system_map.md -o system_map.pdf

# Or simpler: open in a browser and print → "Save as PDF"
open -a "Google Chrome" docs/system_map.md
```

Or just view it on GitHub — it renders cleanly there.
