# Roadmap

A staged plan. Each phase is reviewable independently and unlocks the next.

## Phase 0 — Project scaffolding ✅ in progress
- [x] pyproject.toml with uv, ruff, mypy, pytest
- [x] .env.example, .gitignore, Makefile
- [x] src/trading/{__init__,cli}.py
- [x] core/{types,config,logging,clock}.py
- [x] config/{universes,risk}.yaml
- [x] tests/test_smoke.py
- [ ] `uv sync` succeeds end-to-end
- [ ] `uv run pytest` green
- [ ] `uv run trading status` prints config

## Phase 1 — Data layer
- [ ] `data.base.DataSource` Protocol — `get_bars(instrument, start, end, freq) -> DataFrame`
- [ ] `data.cache.ParquetCache` — partition by `{asset_class}/{symbol}/{freq}.parquet`
- [ ] `data.yfinance_source` — daily US equities
- [ ] `data.ccxt_source` — daily/hourly crypto via Binance public API
- [ ] `data.ibkr_source` — FX and intraday equities via ib-async
- [ ] CLI: `trading data fetch <universe> --from --to --freq`

## Phase 2 — Backtester
- [ ] `backtest.engine.run_vectorized(prices_df, weights_df) -> pnl, trades`
- [ ] `backtest.metrics`: Sharpe, Sortino, Calmar, max DD, turnover, hit rate, exposure
- [ ] `backtest.walkforward.expanding(...)` — rolling OOS evaluation
- [ ] Slippage + commission models plugged in
- [ ] CLI: `trading backtest <strategy> <universe>`

## Phase 3 — Strategy library
- [ ] `strategies.base.Strategy` interface (generates Signal from bars)
- [ ] Trend: Donchian breakout + EMA crossover
- [ ] Cross-sectional momentum (12-1 month, sector-neutral)
- [ ] Mean-reversion: RSI(2), z-score on residuals
- [ ] Pairs / stat-arb: cointegration test + z-score entry
- [ ] Risk-parity with realized-vol weighting

## Phase 4 — Regime layer
- [ ] HMM regime model (Gaussian, 2-3 states) over market returns
- [ ] Realized-vol regime classifier (low/mid/high)
- [ ] Strategy hook: `Strategy.modulate(regime) -> scaled_signal`

## Phase 5 — Selection & combination
- [ ] Walk-forward strategy ranking by OOS Sharpe with deflated-Sharpe penalty
- [ ] Equal-weight, inverse-vol, and correlation-aware combiners
- [ ] Portfolio vol-targeting overlay

## Phase 6 — IBKR execution
- [ ] `execution.base.Broker` Protocol
- [ ] `execution.simulator` — fills against historical bars + slippage
- [ ] `execution.ibkr` — ib-async wrapper, contract resolution, reconciliation
- [ ] Order lifecycle persistence in SQLite

## Phase 7 — Risk manager
- [ ] Pre-trade: per-position, gross/net exposure, sector caps
- [ ] Intraday: daily-loss kill switch, drawdown halt
- [ ] Force-flatten command + persisted halt state

## Phase 8 — Live runner
- [ ] APScheduler-based daily cycle: fetch → signal → risk → execute → reconcile
- [ ] State persistence (positions, P&L, halts) in SQLite
- [ ] Telegram alerts on errors and significant events
- [ ] Health endpoint + heartbeat file

## Phase 9 — Deploy
- [ ] Dockerfile + docker-compose with IB Gateway image
- [ ] Hetzner / DigitalOcean deploy runbook
- [ ] Log shipping (optional)
- [ ] Restore-from-state drill documented
