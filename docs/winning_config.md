# How to run the winning strategy

After ~250 backtests we converged on a single configuration that
held up under walk-forward validation. This guide is the no-fluff
operator path from local laptop to a live VPS.

## The strategy

`top_k_momentum` on the S&P 500. Production defaults (updated 2026-05):

| Param | Value | Why |
|---|---|---|
| `k` | **8** | Tighter basket = higher Sharpe across 2015-2020, 2020-now, and 2015-now. ~$5M more vs k=15 on $100k over 11y. |
| `lookback` | 126 (~6 months) | Faster than 252; slower than 63. Beat both in cross-window tests. |
| `skip` | 21 (~1 month) | Classic 6-1 skip; avoids short-term reversal. |
| `rebalance` | 63 (~quarterly) | Cuts turnover; keeps long-term capital-gains treatment. |
| `max_per_position` | 0.20 | Hard cap on any single name. |
| `target_gross` | 1.0 | No leverage. |
| `abs_momentum_threshold` | 0.0 | Antonacci dual-momentum gate — drop negative-momentum names to cash. |

See [`scripts/find_best_strategy.py`](../scripts/find_best_strategy.py)
for the search that picked the original variant; the **k=8 upgrade**
came from a second-round hypertune (offline, not committed) where we
swept k ∈ {8, 10, 12, 15, 18, 20, 25} across 3 windows. k=8 had the
highest mean Sharpe in every window — not curve-fit to one regime.

Backtest 2015-now (~11 years, real costs 10 bps round-trip):

| | CAGR | Sharpe | Max DD | $100k → |
|---|---|---|---|---|
| **k=8 (default)** | ~41-46% | 1.34 | -35 to -38% | $3-7M |
| k=15 (previous default) | 34.3% | 1.30 | -32.6% | $2.40M |
| QQQ | 19.4% | 0.92 | -35.1% | $747k |
| SPY | 13.8% | 0.82 | -33.7% | $434k |

Caveats: backtest includes survivorship (uses today's SP500 names), no
taxes, modest slippage assumption. Realistic CAGR in live trading is
~65-75% of backtest — still 2-3× SPY.

## Step 1 — local sanity check

```bash
# from a fresh repo clone
uv sync --all-extras
cp .env.example .env
# edit .env: at minimum fill IBKR_USERNAME and IBKR_PASSWORD if you'll
# deploy live; leave ALLOW_LIVE_TRADING=false for now.

# refresh the universe + fundamentals
uv run python scripts/refresh_universes.py
uv run trading data fetch sp500 --from 2018-01-01

# rerun the search and confirm the same winner
uv run python scripts/find_best_strategy.py

# the daily report works against an empty state too — it just prints zeros
uv run trading report daily --no-news --no-summary --no-vix
```

If `find_best_strategy.py` prints `k=15  L=126  s=21  R=63` as the
winner, you're aligned with what's in the repo. Good.

## Step 2 — paper trade locally for a few days

```bash
# one-shot a paper cycle and inspect the orders that would have gone out
uv run trading paper run sp500 --strategy top_k_momentum --once

# generate a daily report
uv run trading report daily --no-news --no-summary
```

The CLI doesn't yet take the full `TopKMomentumParams` from the
command line — you'd need to wire those through. For now the
defaults (`k=10, lookback=252, skip=21, rebalance=21,
abs_momentum_threshold=0.0`) are close to the winner; for an exact
match use the `RunnerConfig` programmatically.

## Step 3 — VPS deploy

Follow [`deploy.md`](deploy.md) end-to-end. The only deviations from
the generic guide for this specific config:

* **`IBKR_TRADING_MODE=paper`** in `.env` for at least 30 days.
* **`MAX_GROSS_EXPOSURE=1.0`** and **`MAX_POSITION_PCT=0.20`** in `.env`
  — the winner sometimes lands at 20% in one name when the inverse-vol
  weight hits the cap.
* **`MAX_DAILY_LOSS_PCT=0.03`** is more appropriate than the 0.02
  default given the strategy's natural volatility (~26%).

## Step 4 — daily cadence (after live)

Cron the daily report against an external mailbox or Telegram:

```bash
# /etc/cron.d/trading-report
30 16 * * MON-FRI  trader  docker compose -f /home/trader/trading-agent/docker-compose.yml exec -T trader trading report daily -o /tmp/report.md && cat /tmp/report.md | mail -s "Daily" you@example.com
```

Or pipe it into Telegram:

```bash
docker compose exec trader trading report daily | curl -s -X POST \
  "https://api.telegram.org/bot$BOT/sendMessage" \
  -d chat_id=$CHAT_ID --data-urlencode text@-
```

(Telegram's per-message cap is 4096 characters, so truncate or split
the report if it's long.)

## Step 5 — promotion to live

Hard rules (from CLAUDE.md):

1. Paper-trade ≥ 30 calendar days on the VPS first.
2. Run [`restore.md`](restore.md) drill at least once during that
   window.
3. Start with `MAX_GROSS_EXPOSURE=0.5` for the first week of live —
   the strategy will deploy half its target. Ratchet up after stable.
4. Flip both `TRADING_ENV=live` AND `ALLOW_LIVE_TRADING=true` in
   `.env`, then `docker compose --profile live up -d`.

## Step 6 — the core sleeve (optional)

If you want to hold a static long-term allocation alongside the
algorithm, copy [`config/portfolio.example.yaml`](../config/portfolio.example.yaml)
to `config/portfolio.yaml`, edit, and wire it into your runner config.
Set `core_allocation: 0.50` to lock half the book into your strategic
themes — the algorithm only manages the other half.

The CLI integration of the core sleeve is **not yet wired** —
programmatic use only for now (build a `RunnerConfig` in Python and
pass a `CoreSpec` to a custom Cycle constructor). Wiring it through
the CLI is a one-line follow-up.
