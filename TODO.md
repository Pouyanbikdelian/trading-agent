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
- [x] Dockerfile + docker-compose with IB Gateway image
- [x] Hetzner / DigitalOcean deploy runbook
- [ ] Log shipping (optional — deferred)
- [x] Restore-from-state drill documented

## Phase 10 — Go-live (real money) — IN PROGRESS 2026-07
The full working checklist lives in **docs/GO_LIVE.md** — read that first.
Order of operations:
- [x] Dashboard: per-sleeve real PnL + FX-correct curves (GO_LIVE.md §1)
      — "Live" tab shipped + deployed 2026-07-09; PM capped at $20K via
      PM_SLEEVE_CAPITAL_USD (bridge-time); PM blocked from /hold symbols
- [x] Live-tab bug fixes (2026-07-09): capital injections no longer read
      as PnL/returns; snapshots now carry real unrealized/realized PnL
      (ibkr get_positions uses portfolio(), was positions() with 0.0)
- [x] Read-only LIVE-account mirror built (compose profile `mirror`:
      second gateway with READ_ONLY_API=yes + `trading mirror run` →
      state_live/ → dashboard sleeve). NOT yet enabled — needs RAM check
      (`free -m`, want >700MB) + IBKR_LIVE_USERNAME in VPS .env + 2FA
      approval on first boot.
- [ ] Pre-live audit (GO_LIVE.md §2). Drill runbook ready: docs/DRILLS.md
      — run during US market hours. Still open: .env lint on VPS
      (partial: `trading status` values verified 2026-07-09), risk-limit
      review, the four drills, CHF sizing check, pins review, cron/TZ
      audit, tag live-candidate-1
- [ ] Live-day config: fresh state dir, sized-down limits
      (MAX_POSITION_PCT=0.05, MAX_GROSS_EXPOSURE=0.50), gates flipped by
      Yan only (GO_LIVE.md §3)
- [ ] Agent PM: sim observation → risk-manager bridge → 30d paper — NOT
      part of the first live wave (GO_LIVE.md §4)

## Phase 11 — Telegram bot v2 (backlog, added 2026-07-09 per Yan)

- [ ] **Robustness overhaul** — the bot should be "so much more robust":
      graceful reconnect/backoff on Telegram API flaps, command timeouts
      that never wedge the poll loop, per-command error isolation (one
      broken handler can't kill the bot), structured error replies
      instead of silence, health self-reporting (/health with uptime,
      last-poll age, handler failure counts), and tests for every
      command path.
- [ ] **Claude-powered agent assistant in the chat** — a conversational
      assistant (Anthropic API, reasoning-capable model) living in the
      Telegram chat: understands the system's state (positions, halts,
      committee journal, memory store), answers questions in natural
      language, and can ACT like an agent — proposing/queueing the
      existing safe commands (/cycle, /hold, /pm run...) rather than
      free-form execution. Design constraints to settle before building:
      tool whitelist (never the raw order path — rule #4), spend budget
      per day, and how it defers to the human on anything gated.

## Phase 12 — HedgeAgents-inspired upgrades (backlog, added 2026-07-11)

Full specs in **docs/HEDGEAGENTS_BACKLOG.md**. Advisory-layer only, never
the order path. Do after go-live wave 1.

- [ ] **Extreme Market Review** — extend sentinel with a 3-day ±10%
      cumulative wire per held symbol; on held-position alarms run a
      structured loss-review (owner presents thesis-broken?/plan, risk
      officer + challenger critique, quant adds numbers) → journaled
      verdict + Telegram; outcome tagged 5 trading days later. (~1 session)
- [ ] **Three-tier memory: reflection + retrieval + distillation** — add
      per-decision Reflection rows with 5d/21d outcomes; embed + retrieve
      top-5 closed cases/lessons into committee & PM prompts; historian
      distills from sleeve-nominated closed reflections with source_ids.
      (~2 sessions)
