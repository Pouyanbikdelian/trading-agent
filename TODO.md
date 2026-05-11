# Roadmap

A staged plan. Each phase is reviewable independently and unlocks the next.

## Phase 0 — Project scaffolding ✅ complete
- [x] pyproject.toml with uv, ruff, mypy, pytest
- [x] .env.example, .gitignore, Makefile
- [x] src/trading/{__init__,cli}.py
- [x] core/{types,config,logging,clock}.py
- [x] config/{universes,risk}.yaml
- [x] tests/test_smoke.py
- [x] `uv sync` succeeds end-to-end
- [x] `uv run pytest` green
- [x] `uv run trading status` prints config

## Phase 1 — Data layer
- [x] `data.base.DataSource` Protocol — `get_bars(instrument, start, end, freq) -> DataFrame`
- [x] `data.cache.ParquetCache` — partition by `{asset_class}/{symbol}/{freq}.parquet`
- [x] `data.yfinance_source` — daily US equities
- [x] `data.ccxt_source` — daily/hourly crypto via Binance public API
- [x] `data.ibkr_source` — FX and intraday equities via ib-async (unit-tested only; live IB Gateway smoke test deferred)
- [x] CLI: `trading data fetch <universe> --from --to --freq`

## Phase 2 — Backtester
- [x] `backtest.engine.run_vectorized(prices_df, weights_df) -> BacktestResult`
- [x] `backtest.metrics`: Sharpe, Sortino, Calmar, max DD, turnover, hit rate, exposure
- [x] `backtest.walkforward.expanding(...)` — rolling OOS evaluation
- [x] Slippage + commission models plugged in
- [x] CLI: `trading backtest <strategy> <universe>`

## Phase 3 — Strategy library
- [x] `strategies.base.Strategy` interface + registry
- [x] Trend: Donchian breakout + EMA crossover
- [x] Cross-sectional momentum (12-1 month)
- [x] Mean-reversion: RSI(2), z-score on residuals
- [x] Pairs / stat-arb: cointegration test + z-score entry
- [x] Risk-parity with inverse-vol weighting

## Phase 4 — Regime layer
- [x] HMM regime model (Gaussian, 2-3 states) over market returns
- [x] Realized-vol regime classifier (low/mid/high)
- [x] Strategy hook: `Strategy.modulate(weights, regime) -> scaled_weights`

## Phase 5 — Selection & combination
- [x] Walk-forward strategy ranking by OOS Sharpe with deflated-Sharpe penalty
- [x] Equal-weight, inverse-vol, and minimum-variance combiners
- [x] Portfolio vol-targeting overlay

## Phase 6 — IBKR execution
- [x] `execution.base.Broker` Protocol
- [x] `execution.simulator` — fills market orders against historical bars + slippage (limit/stop deferred)
- [x] `execution.ibkr` — ib-async wrapper, contract resolution, reconciliation
- [x] Order lifecycle persistence in SQLite

## Phase 7 — Risk manager
- [x] Pre-trade: per-position, gross/net exposure, sector caps
- [x] Intraday: daily-loss kill switch, drawdown halt
- [x] Force-flatten command + persisted halt state

## Phase 8 — Live runner
- [x] APScheduler-based daily cycle: fetch → signal → risk → execute → reconcile
- [x] State persistence (positions, P&L, halts) in SQLite
- [x] Telegram alerts on errors and significant events
- [x] Heartbeat file (HTTP health endpoint deferred — not needed v1)

## Phase 9 — Deploy
- [ ] Dockerfile + docker-compose with IB Gateway image
- [ ] Hetzner / DigitalOcean deploy runbook
- [ ] Log shipping (optional)
- [ ] Restore-from-state drill documented
