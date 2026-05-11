# Trading Agent

Autonomous research + execution platform for swing/position trading across US equities, FX, and crypto, executing through Interactive Brokers (IBKR).

## What this is

A staged, defensive trading stack you control end-to-end:

1. **Research** — multi-source historical data, vectorized backtester, walk-forward analysis, regime detection.
2. **Strategy portfolio** — trend-following, cross-sectional momentum, mean-reversion, pairs trading, risk-parity.
3. **Selection & combination** — pick winners out-of-sample, combine into an ensemble.
4. **Paper execution** — IBKR paper account via IB Gateway, identical code path to live.
5. **Risk manager** — hard pre-trade limits, daily-loss kill switch, drawdown halt.
6. **Live runner** — 24/7 scheduler with monitoring and alerts.

## What this is NOT

- It is **not** a get-rich-quick script. Edges are small and hard-won.
- It does **not** auto-execute live trades until you (a) set `ALLOW_LIVE_TRADING=true`, (b) set `TRADING_ENV=live`, and (c) flip the runtime kill-switch off. Defaults are paranoid.
- It is **not** an HFT or sub-second system. Decisions happen on bars (1-minute and up).

## Quick start

Prerequisites: Python 3.10–3.12, `uv` (https://docs.astral.sh/uv/), Interactive Brokers account.

```bash
# 1. Install dependencies
uv sync

# 2. Copy and edit env
cp .env.example .env

# 3. Verify install
uv run trading --help

# 4. Run the test suite
uv run pytest -q
```

Install `uv` on macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Repo layout

```
src/trading/
  core/         types, clock, logging, config
  data/         data source adapters + Parquet cache
  backtest/     vectorized engine, metrics, walk-forward
  strategies/   strategy interface + library
  regime/       HMM and vol-regime classifiers
  selection/    OOS strategy selection and combination
  execution/    broker adapters (IBKR, simulator)
  risk/         pre-trade risk manager
  runner/       live scheduler, state, monitoring
  cli.py        single CLI entry
config/         universes, risk limits, strategy params
scripts/        one-off scripts (data backfills, analysis)
notebooks/      research notebooks
tests/          unit + integration tests
```

## Operating modes

Controlled by `TRADING_ENV` in `.env`:

- `research` (default) — no broker connection. Backtesting, data fetches, analysis only.
- `paper` — connects to IBKR paper account on port 7497. Submits real paper orders.
- `live` — refuses to start unless `ALLOW_LIVE_TRADING=true`. Submits real money orders.

The risk manager applies the same hard limits in all modes.

## Roadmap

See `TODO.md`. Current phase tracked in the task list.
