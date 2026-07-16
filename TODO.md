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
- [x] Pre-live audit (GO_LIVE.md §2) — COMPLETE 2026-07-15 except the
      final tag. All 4 drills passed; SIX real defects found + fixed:
      dead daily-loss kill switch, month of silent no-trading
      (short-history symbol truncated the price matrix), CHF sizing off
      by the USDCHF factor, sector cap never bound (no fundamentals),
      ghost pins eating basket slots, and order-stacking → paper shorts
      (fixed via working-order netting + long-only invariant). See
      docs/incidents.md 2026-07-14/15.
- [ ] Tag `live-candidate-1` after one clean cycle on the deployed
      commit (book un-shorted, ~9 names, no clamp surprises)
- [ ] Live-day config: fresh state dir, sized-down limits
      (MAX_POSITION_PCT=0.05, MAX_GROSS_EXPOSURE=0.50), gates flipped by
      Yan only (GO_LIVE.md §3). NOTE stale `trader-live` compose service
      needs rework first (shares paper state dir + bridged networking).
- [ ] Agent PM: sim observation window ENDED ~2026-07-12 — review
      PM-vs-SPY record, then build the PM→Signal bridge (through the
      real risk manager; $20K cap via PM_SLEEVE_CAPITAL_USD), then 30d
      as ~20% paper sleeve — NOT part of the first live wave (§4)
- [ ] Before November: `CRON=5 22 * * FRI` in VPS .env (winter DST —
      21:05 UTC is only 5 min after the close in winter; see GO_LIVE §2)

## Phase 11 — Telegram bot v2 (backlog, added 2026-07-09 per Yan)

- [ ] **"No trades in N cycles" watchdog** — alert when the cycle
      produces no orders for N consecutive runs ("verify this is
      intentional"). Would have caught the June dead month in week one.
      Small; do first.
- [ ] **Execution upgrade** — IBKR Adaptive algo or marketable-limit
      orders instead of raw market orders (protects against bad opens;
      TWAP/VWAP overkill at current size).
- [x] **Telegram update-offset persistence** (external review 2026-07-15,
      fixed 2026-07-16) — offset persisted atomically to
      state/telegram_offset.json BEFORE dispatch; crash-restart can no
      longer replay an executed command. Corrupt/missing file degrades
      to the old behavior.
- [ ] **Order status never promoted for overnight fills** (found via
      copilot 2026-07-16) — the cycle marks FILLED only for fills that
      arrive within the submitting cycle; after-hours orders that fill
      at the next open stay 'submitted' in orders.db forever (all of
      June 10 + July 14 rows). Fix: reconcile fills since the LAST
      cycle, not since this cycle's start; backfill existing rows.
      Copilot flags these as stale meanwhile.
- [ ] **Single execution lock** (external review 2026-07-15) — cron
      cycle, trigger cycle, approval flow and manual commands lack one
      mutual-exclusion primitive. Mitigated by per-job locks, the 10s
      cooldown and the new submit-gate re-checks; a proper cross-path
      lock (file-lock around order submission) is the clean fix.
- [ ] **Robustness overhaul** — the bot should be "so much more robust":
      graceful reconnect/backoff on Telegram API flaps, command timeouts
      that never wedge the poll loop, per-command error isolation (one
      broken handler can't kill the bot), structured error replies
      instead of silence, health self-reporting (/health with uptime,
      last-poll age, handler failure counts), and tests for every
      command path.
- [x] **Copilot Phase 1 (read-only) — SHIPPED 2026-07-16.**
      `src/trading/copilot/` + `/ask` `/why SYM` `/thesis SYM`
      `/committee SYM` in the bot. Derives decisions+transcripts from
      the memory journal into state/copilot.db (FTS5), links orders/
      fills, answers THEN/NOW/CHANGED with mandatory citations, honest
      no-evidence path (no LLM call), Haiku default (Qwen/DeepSeek via
      env), rate-limited + audited, import-guard test bans any
      execution path. Docs: docs/COPILOT.md.
- [ ] **Copilot Phase 2 (acting assistant)** — propose/queue the
      existing safe commands (/cycle, /hold, /pm run...) with explicit
      confirmation; never the raw order path (rule #4). Design gates:
      tool whitelist, daily spend budget, human confirmation for
      anything gated. Build only after Phase 1 proves useful in daily
      use.

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
- [ ] **Anomaly watch MVP ("Mathematician", demoted — added 2026-07-15
      per Yan)** — deterministic scanners, NO new committee voice yet.
      Honest framing: a risk sensor, not an alpha source (stat anomalies
      in liquid names on free daily data are picked clean; own research
      precedent: rotation-cluster study — "real but not naively
      tradeable"). Scanners (pure code, after close, into
      state/anomalies.json): (1) correlation clustering on CURRENT
      holdings — alert when the book has become one trade (would have
      flagged the 90%-semis book); (2) long-standing pair/spread breaks;
      (3) realized-vs-implied vol dislocation on SPY + held names. Every
      finding reports effect size, sample size, AND search breadth
      (multiple-testing honesty — a scanner that hides how many things
      it tested is a p-hacking machine). Findings feed committee context
      as data; existing voices react. Promotion rule: if journaled
      findings prove useful for ~2 months (graded via the calibration
      store), THEN consider a full Mathematician voice with graded
      predictions; if noise, delete 3 functions. (~1-2 sessions; after
      go-live wave 1 + PM bridge)
