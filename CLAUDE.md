# CLAUDE.md

This file orients Claude Code (or any AI coding assistant) on this project. Read it first.

## Project

A staged, defensive automated trading system for **Interactive Brokers (IBKR)**, covering US equities, FX, and crypto. End state: a 24/7 live system with a built-in risk manager. Current state: Phase 0 complete (scaffolding); Phase 1 (data layer) is next.

## Owner profile

- **Yan** (`podibiki@gmail.com`). Experienced developer. Skip hand-holding.
- IBKR account exists with **paper + live both enabled**. Use paper for everything until explicitly cleared for live.
- Wants both swing (days–weeks) and position (weeks–months) horizons. Not intraday HFT.
- Budget for paid data ≤ ~$20/mo. Free sources only for v1 (yfinance, ccxt, IBKR).

## Hard rules (do not violate)

1. **Never auto-execute live trades.** Code may submit orders against the broker; *Claude* must not flip live-trading flags or run live execution without explicit user approval each time.
2. **Live trading requires BOTH `TRADING_ENV=live` AND `ALLOW_LIVE_TRADING=true`** in `.env`. The `Settings.is_live_armed()` gate enforces this. Do not weaken it.
3. **Paper-trade first, always.** New strategies go through backtest → walk-forward OOS → paper for ≥30 days → only then live, with sized-down position limits initially.
4. **Risk manager is the only path to orders.** Strategies emit `Signal` (target weights). The risk manager turns signals into `Order`s after applying limits. Strategies must not construct `Order`s directly.
5. **Timezone-aware datetimes only.** `Bar.ts` validates this. Use `trading.core.clock` not `datetime.utcnow()`.
6. **Never commit `.env`, `data/`, `logs/`, `state/`.** All gitignored.

## Architecture (settled)

```
src/trading/
  core/         types, settings, clock, logging
  data/         DataSource Protocol + adapters (yfinance, ccxt, ibkr) + Parquet cache  [Phase 1]
  backtest/     vectorized engine, metrics, walk-forward harness                       [Phase 2]
  strategies/   Strategy interface + library (trend, momentum, meanrev, pairs, RP)     [Phase 3]
  regime/       HMM + realized-vol regime classifiers                                  [Phase 4]
  selection/    OOS selection + portfolio combination                                  [Phase 5]
  execution/    Broker Protocol + IBKR adapter + simulator                             [Phase 6]
  risk/         pre-trade limits + kill switches                                       [Phase 7]
  runner/       APScheduler-based live loop                                            [Phase 8]
  cli.py        single Typer CLI; subcommands grow per phase
config/         universes.yaml, risk.yaml — YAML for things that don't belong in env
scripts/        one-off backfills and analyses
notebooks/      research notebooks (gitignored unless .template.ipynb)
tests/          pytest, fast smoke tests on every change
```

## Design decisions (don't relitigate)

- **Python 3.10–3.12, managed by `uv`.** Not poetry, not pip-tools.
- **Pandas + NumPy + numba** for the data and backtester. Polars considered, rejected for v1 (ecosystem fit).
- **pydantic v2 + pydantic-settings** for types and config. All domain models are `frozen=True`.
- **loguru** (not stdlib logging) for the logging sink.
- **typer** (not click directly, not argparse) for the CLI. Single `trading` entry point with subcommand groups.
- **ib-async** (the maintained successor to ib_insync) for IBKR.
- **Custom vectorized backtester** — not vectorbt, not backtrader. Keeps the engine ~200 LOC and fully ours.
- **Parquet local cache** under `data/parquet/{asset_class}/{symbol}/{freq}.parquet`. Partition layout fixed.
- **Strategy interface emits target weights** (not orders). Combiner aggregates; risk manager sizes.
- **Risk manager is hard-blocking**. Cannot be bypassed by a strategy. Returns `RiskDecision(action, reason, scale_factor)`.

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

# Phase-specific (as built)
uv run trading data fetch <universe> --from 2018-01-01 --freq 1d
uv run trading backtest <strategy> <universe>
uv run trading paper
uv run trading live   # refuses unless ALLOW_LIVE_TRADING=true AND TRADING_ENV=live
```

## Test discipline

- `tests/test_smoke.py` is fast, hermetic, no network — runs every commit.
- Tests that hit the network or take >1s get `@pytest.mark.slow`.
- Tests that need a running IB Gateway get `@pytest.mark.live`.
- New code without a test does not get merged.

## Roadmap

See `TODO.md`. Phase 0 done. Next up: **Phase 1 — data layer**.

Phase 1 deliverables:
- `data.base.DataSource` Protocol — `get_bars(instrument, start, end, freq) -> pd.DataFrame`.
- `data.cache.ParquetCache` — read-through cache with `{asset_class}/{symbol}/{freq}.parquet`.
- `data.yfinance_source` — daily US equities/ETFs.
- `data.ccxt_source` — daily and hourly crypto via Binance public.
- `data.ibkr_source` — FX (IDEALPRO) and IBKR-backed intraday equities via `ib-async`.
- CLI subcommand `trading data fetch <universe> --from --to --freq` driven by `config/universes.yaml`.
- Smoke tests using a tiny fixture parquet (no network in CI).

## Conventions

- Public types in `core/types.py`, never `Dict[str, Any]` in signatures.
- Async only where the broker/network forces it. Backtester is synchronous.
- Logger lines: `logger.bind(strategy=...).info("...")` for attribution-friendly context.
- No `print()` outside `cli.py` and `scripts/`.
- Docstrings explain *why*, not *what*.

## What NOT to do without checking with Yan first

- Add a new paid data source.
- Change risk limit defaults in `.env.example` or `config/risk.yaml`.
- Loosen the live-trading gates.
- Pick a different broker abstraction.
- Switch off the test markers.
