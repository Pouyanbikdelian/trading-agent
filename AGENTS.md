# AGENTS.md

This file orients Codex (or any AI coding assistant) on this project. Read it first.

## Project

A staged, defensive automated trading system for **Interactive Brokers (IBKR)**, covering US equities, FX, and crypto, with an advisory LLM committee layered on top of a mechanical strategy core.

**Current state: LIVE WITH REAL MONEY.** First live trade 2026-08-11 (a ~$9.7k PM basket). Phases 0–9 are built; Phase 10 (go-live) is in progress and Phase 13 (continuous learning) is partially built. `TODO.md` is the authoritative roadmap and is kept current — trust it over any summary here.

This is not a scaffold. Assume every change can reach a real broker account, and read `docs/incidents.md` before touching the order path, the risk manager, or the runner.

## Owner profile

- **Yan** (`podibiki@gmail.com`). Experienced developer. Skip hand-holding.
- IBKR account exists with **paper + live both enabled**. Use paper for everything until explicitly cleared for live.
- Wants both swing (days–weeks) and position (weeks–months) horizons. Not intraday HFT.
- Budget for paid data ≤ ~$20/mo. Free sources only for v1 (yfinance, ccxt, IBKR).
- Prefers direct, opinionated analysis with honest verdicts over hedged summaries.

## Hard rules (do not violate)

1. **Never auto-execute live trades.** Code may submit orders against the broker; *Codex* must not flip live-trading flags or run live execution without explicit user approval each time.
2. **Live trading requires BOTH `TRADING_ENV=live` AND `ALLOW_LIVE_TRADING=true`** in `.env`. The `Settings.is_live_armed()` gate enforces this. Do not weaken it.
3. **Paper-trade first, always.** New strategies go through backtest → walk-forward OOS → paper for ≥30 days → only then live, with sized-down position limits initially.
4. **Risk manager is the only path to orders.** Strategies emit `Signal` (target weights). The risk manager turns signals into `Order`s after applying limits. Strategies must not construct `Order`s directly.
5. **Timezone-aware datetimes only.** `Bar.ts` validates this. Use `trading.core.clock`, not `datetime.utcnow()`.
6. **Never commit `.env`, `data/`, `logs/`, `state/`.** All gitignored.
7. **A state directory belongs to one `TRADING_ENV`.** `core/state_env.py` stamps and verifies it. A paper baseline read by a live process once halted the desk — do not remove the stamp check.
8. **The agent layer is advisory.** Committee, PM, historian/curator and copilot write to memory and Telegram. Only the risk manager and the guards may move exposure. Do not give an LLM a direct order path. *Fact check (2026-09-23): with `AGENT_PM_SLEEVE_PCT > 0` the PM's target weights are executable targets — they reach orders only through `pm_signal` → risk manager → approval. The Sept 19 audit found live at PM sleeve 1.0 / strategy sleeve 0.0. "Advisory" means no direct order path, not no influence.*
9. **Operator blocklists bind in code, not in a prompt.** `/exclude` is a hard filter. A charter sentence asking an LLM nicely is not an enforcement mechanism.

## Architecture (settled)

```
src/trading/
  core/         types, settings, clock, logging, state_env stamps
  data/         DataSource Protocol + adapters (yfinance, ccxt, ibkr) + Parquet cache
  backtest/     vectorized engine, metrics, walk-forward harness, guards overlay
  strategies/   Strategy interface + library (trend, momentum, meanrev, pairs, RP)
  regime/       HMM + realized-vol regime classifiers
  selection/    OOS selection (PSR/DSR) + portfolio combination + overlays
  execution/    Broker Protocol + IBKR adapter + simulator
  risk/         pre-trade limits + kill switches + guards
  portfolio/    core/satellite sleeves and target construction
  runner/       APScheduler live loop, cycle, playbook, config
  runtime/      watchers: market, news, econ, ops, sentinel, portfolio stats
  agents/       committee (8 voices), simulated PM, historian/curator, candidate ladder, context
  memory/       journal, episodes, lessons, predictions, source trust, shadow book
  copilot/      approval-gated desk assistant over live + PM evidence
  bot/          Telegram command surface, desk, registry, keyboards
  dashboard/    read-only web view of live + PM state
  reporting/    digests and scorecards
  cli.py        single Typer CLI; subcommand groups
config/         universes.yaml, watchlist.yaml, playbook + portfolio examples (risk limits live in .env)
scripts/        backfills, analyses, one-off audits
tests/          pytest, fast smoke tests on every change
docs/           system_map, LEARNING_ARCHITECTURE, GO_LIVE, DRILLS, incidents, deploy
```

## The learning layer (Phase 13, advisory only)

Full argument in `docs/LEARNING_ARCHITECTURE.md`.

- **Memory** (`state/memory/memory.db`) holds the journal, graded episodes, lessons, predictions, the source-trust ledger and the shadow book.
- **Lessons** move `candidate → established → challenged → retired`, and nothing is ever deleted. Only `established` lessons reach agent context.
- **The Learning Curator** (`agents/historian.py`) runs a weekly (Friday 19:00 New York) evidence-gated pass: it reviews lessons against measured outcomes, proposes candidates, promotes and challenges, and *recommends* archiving challenged machine lessons. Each pass is persisted immutably to `curator_runs` / `curator_actions`.
- **Archiving a challenged lesson needs a human.** The curator only recommends. Restoration (`/lesson restore`) returns a lesson to `candidate`, never straight back to `established` — a prior belief must re-earn its place on fresh evidence.
- **Operator-stated lessons are protected** from machine archiving.

## Design decisions (don't relitigate)

- **Python 3.10–3.12, managed by `uv`.** Not poetry, not pip-tools.
- **Pandas + NumPy** for the data and backtester (numba is declared but currently unused). Polars considered, rejected for v1 (ecosystem fit).
- **pydantic v2 + pydantic-settings** for types and config. All domain models are `frozen=True`.
- **loguru** (not stdlib logging) for the logging sink.
- **typer** (not click directly, not argparse) for the CLI. Single `trading` entry point with subcommand groups.
- **ib-async** (the maintained successor to ib_insync) for IBKR.
- **Custom vectorized backtester** — not vectorbt, not backtrader. Keeps the engine ~200 LOC and fully ours.
- **Parquet local cache** under `data/parquet/{asset_class}/{symbol}/{freq}.parquet`. Partition layout fixed.
- **Strategy interface emits target weights** (not orders). Combiner aggregates; risk manager sizes.
- **Risk manager is hard-blocking**. Cannot be bypassed by a strategy. Returns `RiskDecision(action, reason, scale_factor)`.
- **Slow momentum config (126/21/63) stays.** Walk-forward says it wins; see `docs/winning_config.md`. Do not speed it up to make the agents look more active. *Caveat (2026-09-23): the evidence in the repo is full-sample on today's index members, with no in-fold parameter selection; the honest rebuild is wave 2. Note the compose command passes `-p rebalance=${REBALANCE:-5}`, i.e. weekly unless .env says otherwise.*

## How to work

```bash
# First-time setup
uv sync --all-extras
cp .env.example .env

# Everyday
uv run pytest -q                    # smoke tests
uv run pytest -m "not slow and not live"   # exclude network/broker tests
uv run trading status               # show env + risk config
uv run trading --help               # CLI help

# Lint/format
make fmt        # ruff format + autofix
make lint       # check, no fix
make typecheck  # mypy strict

# Operating
uv run trading data fetch <universe> --from 2018-01-01 --freq 1d
uv run trading backtest run <strategy> <universe> --from 2018-01-01
uv run trading paper run <universe>
uv run trading live run <universe>   # refuses unless ALLOW_LIVE_TRADING=true AND TRADING_ENV=live
python3 scripts/audit_committee_repetition.py   # is the committee saying anything new?
```

Deployment is the VPS at `/opt/trading-agent` via docker compose — see `docs/deploy.md`. Live drills are in `docs/DRILLS.md`; run them on the paper book.

## Test discipline

- `tests/test_smoke.py` is fast, hermetic, no network — runs every commit.
- Tests that hit the network or take >1s get `@pytest.mark.slow`.
- Tests that need a running IB Gateway get `@pytest.mark.live`.
- New code without a test does not get merged.

## Roadmap

See `TODO.md` — it is current. Phases 0–9 complete; **Phase 10 (go-live) in progress**, Phase 11 (Telegram bot v2), Phase 12 (HedgeAgents ideas) and Phase 13 (continuous learning) partially built.

## Conventions

- Public types in `core/types.py`, never `Dict[str, Any]` in signatures.
- Async only where the broker/network forces it. Backtester is synchronous.
- Logger lines: `logger.bind(strategy=...).info("...")` for attribution-friendly context.
- No `print()` outside `cli.py` and `scripts/`.
- Docstrings explain *why*, not *what* — this codebase uses them to record the incident that motivated the code. Keep that habit.
- A feature that "reports attempts, not achievements" is a bug. If a subsystem can silently never run, add the watchdog with it.

## What NOT to do without checking with Yan first

- Add a new paid data source.
- Change risk limit defaults in `.env.example` (the only source; the unread `config/risk.yaml` was removed 2026-09-23).
- Loosen the live-trading gates.
- Pick a different broker abstraction.
- Switch off the test markers.
- Give any agent or LLM a path to the order book.
- Retire or archive an established lesson by machine action alone.

## Imported Claude Cowork project instructions

Help me a develop a genious trading system that's automated.
