# Configuration — every knob that matters

## The gates that stop live trading

Live requires **both**, and `Settings.is_live_armed()` enforces the AND:

```bash
TRADING_ENV=live
ALLOW_LIVE_TRADING=true
```

Either one alone does nothing. `trading live` refuses to start otherwise.
This is the single most important invariant in the repo and nothing
should ever weaken it — not a flag, not a "temporary" override, not a
test fixture that patches it globally.

## Where each setting actually lives

There are three config surfaces and they do **not** overlap the way the
file names suggest.

| Surface | Read by | Contains |
|---|---|---|
| `.env` | `core/config.Settings` (pydantic-settings) | **All risk limits the risk manager enforces**, secrets, broker, universe/strategy selection |
| `config/risk.yaml` | backtester + simulator only | Slippage model, documented intent. **The portfolio/position/drawdown numbers here are NOT loaded by the risk manager.** |
| `config/universes.yaml` + `universes.generated.yaml` | `core/universes.load_universe` | Symbol sets. Hand-curated file wins on key collision. |

That second row is a trap and the file's own header now warns about it:
editing `max_position_pct` in `risk.yaml` changes nothing about live
risk. `.env` is the source of truth.

### Risk limits (`.env`)

```bash
MAX_GROSS_EXPOSURE=1.0        # long-only, no leverage
MAX_POSITION_PCT=0.10         # any single name
MAX_DAILY_LOSS_PCT=0.02       # one-day kill switch
MAX_DRAWDOWN_PCT=0.15         # full halt until a human resets
MAX_MARGIN_BORROWING_PCT=...
REQUIRE_CYCLE_APPROVAL=       # human confirms each rebalance
CYCLE_APPROVAL_TIMEOUT_S=
```

### What trades and where

```bash
UNIVERSE=sp500                # also drives the agents' candidate ladder
STRATEGY=top_k_momentum
CRON=                         # rebalance schedule
```

`UNIVERSE` and `STRATEGY` are shared between the trading loop and the
agent candidate ladder on purpose — the agents rank what the system
trades, so the ladder is the house view rather than a second opinion.

### Model routing

```bash
AGENTS_ENABLED=true
ANTHROPIC_API_KEY=...
AGENTS_MODEL=                 # specialists; default claude-sonnet-4-6
AGENTS_MODEL_FRONTIER=        # decision nodes; default claude-opus-4-8
```

Three nodes run on the frontier model: **challenger**, **manager**, and
the **agent PM**. Everything else is mid-tier. Opus is therefore in the
per-cycle path — watch credit burn. A June 2026 committee outage was
simply zero API credit (Anthropic returns HTTP 400).

## Hard caps that live in code, not config

These are in `agents/pm.py` and bind the simulated sleeve. They are
deliberately code rather than YAML: they are invariants, not tunables.

| Constant | Value | Why |
|---|---|---|
| `MAX_WEIGHT_PER_NAME` | 0.25 | ETFs are diversified |
| `MAX_WEIGHT_PER_STOCK` | 0.10 | single names carry idiosyncratic risk |
| `MAX_CLUSTER` | 0.50 | six 0.9-correlated names is one bet wearing six hats |
| `MAX_ETF_POSITIONS` | 3 | forces individual-name expression |
| `MAX_GROSS` | 1.0 | long-only |
| `STALE_BOOK_CYCLES` | 3 | cycles of an unchanged name set before the digest complains |
| `COST_BPS` | 10 | commission + slippage charged on sim turnover |

`CLUSTERS` in the same file maps ~60 tickers to five correlated groups
(tech_complex, energy, health, financials, defense). It is deliberately
beta-based rather than GICS-pure — AMZN and TSLA trade with tech beta, so
for *risk* purposes they live in the tech complex. **Symbols not in the
map are uncapped at cluster level**, which is a real gap as the universe
widens.

## Operator controls (Telegram)

| Command | Effect | Durability |
|---|---|---|
| `/hold SYM` | Freezes a position out of the rebalance; PM is hard-blocked from allocating to it | until `/unhold` |
| natural-language instruction | Captured as a **mandate**, graded strong/medium/soft by phrasing | expires |
| `/lesson <text>` | Written to permanent memory; firm tone → `established`, else `candidate` | permanent |
| `/lessons harden\|soften <id>` | Re-weight a lesson | permanent |
| `/halt`, `/resume` | Kill switch | until reset |
| `/approve`, `/reject`, `/pick` | Gate a pending cycle | per cycle |

Mandates and lessons are different things and the difference matters: a
mandate is an instruction for the *next run* that is then consumed; a
lesson is a durable belief that agents carry forward and that can be
graded. See `04_PRODUCTION_READINESS.md` for the current gap in how
mandates are detected.
