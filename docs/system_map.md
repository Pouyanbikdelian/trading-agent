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

Once `halt.json` says halted, the next cycle:
1. Refuses any new orders
2. If `flatten_on_next_cycle: true`, force-closes existing positions
3. Stays halted until you explicitly `/resume` or `trading resume`

The risk manager itself **never auto-unhalts**. By design — an
automatic recovery on a partially-understood failure is how money
disappears.

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
