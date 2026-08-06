# First-month live checklist

Print this. Tick it. The point of the first month is not profit — it is
**finding out what is broken while the amount at risk is small enough not
to matter.**

Everything on this list exists because something in this system once ran
without erroring, without alerting, and without working. On 2026-08-06 a
single sweep found the agent scorecard had never graded a prediction, the
counterfactual ledger had never graded a leg, and the historian had never
been scheduled on a box without an unrelated feature flag set. All three
were green on every check that existed at the time.

**So the governing rule of this checklist: never accept "it ran" as
evidence. Ask what it produced.**

---

## Before you switch it on — day 0 gate

Do not proceed past a single unticked box.

| ☐ | Check | How | Pass |
|---|---|---|---|
| ☐ | Account holds only the money you can lose | IBKR portal | 8–10k, everything else moved out |
| ☐ | Both live gates deliberately set | `grep -E "TRADING_ENV\|ALLOW_LIVE" .env` | `live` **and** `true`, typed by you today |
| ☐ | Risk limits reviewed line by line | `uv run trading status` | position/gross/daily-loss/drawdown are what you intend **for a 10k account** |
| ☐ | Margin borrowing off | `.env` | `MAX_MARGIN_BORROWING_PCT=0.0` |
| ☐ | Cycle approval on for month one | `.env` | `REQUIRE_CYCLE_APPROVAL=true` — you approve every rebalance by hand |
| ☐ | IBKR 2FA resolved for unattended login | test a gateway restart | comes back **without** you tapping a phone |
| ☐ | Full suite green on the Mac | `uv run pytest -q -m "not slow and not live"` | 0 failures |
| ☐ | VPS has headroom | `free -m` | ≥800 MB available, swap < 20% |
| ☐ | Deployed image is the commit you think | `git log --oneline -1` on VPS + `docker compose ps` | container ages in minutes, not hours |
| ☐ | Kill switch rehearsed | `/halt` then `/resume` in Telegram | halt takes effect and is visible in `/status` |
| ☐ | You know how to flatten manually | IBKR portal, not the bot | you have done it once |
| ☐ | Baseline snapshot recorded | see "Week 0 baseline" below | written down on paper |

### Open items already logged in `TODO.md` / `GO_LIVE.md`

Found 2026-08-06 by re-reading the backlog. These predate this document
and several are live-blocking. Do not treat the day-0 gate as complete
until each has been resolved or consciously accepted.

| ☐ | Item | Why it matters live | Source |
|---|---|---|---|
| ☐ | **Order status never promoted for overnight fills** | An after-hours order that fills at the next open stays `submitted` in `orders.db` **forever**. The state machine loses track of a real position. On live that risks re-submitting or mis-sizing. | TODO Phase 11 |
| ☐ | **No single execution lock** | Cron cycle, trigger cycle, approval flow and manual commands have no shared mutual-exclusion. Per-job locks + a 10s cooldown mitigate; they do not prevent. Order stacking already shorted two names on paper once. | TODO Phase 11, external review 2026-07-15 |
| ☐ | **`trader-live` compose service is stale** | Shares the paper **state dir** and uses bridged networking. Live must have a *fresh* state dir — sharing halt state, order ledger and snapshots between paper and live is a data-corruption path. | TODO Phase 10 |
| ☐ | **Sized-down live limits not yet set** | Plan on record: `MAX_POSITION_PCT=0.05`, `MAX_GROSS_EXPOSURE=0.50` for month one, kills unchanged. | GO_LIVE §3 |
| ☐ | **`live-candidate-1` tag** after one clean cycle on the deployed commit | You cannot roll back to "the version that worked" without one. | GO_LIVE §2 |
| ☐ | **IB Gateway live login** — live username is the paper username *without* the trailing "2"; port 4001 not 4002; needs `up -d --force-recreate`, a plain restart does not re-read `.env` | Getting this wrong on live day means no connection at the open. | GO_LIVE §3 |
| ☐ | **Abort procedure written on paper** | How to halt, how to flatten manually in TWS, what happens if the VPS dies mid-session. Literally on paper. | GO_LIVE §3 |
| ☐ | **Telegram alerts on a channel you watch with sound on** | First live week. | GO_LIVE §3 |
| ☐ | **"No trades in N cycles" watchdog** — still open | The backlog says it "would have caught the June dead month in week one. Small; do first." The new ops liveness checks cover the committee/PM/historian, **not** "the cycle produced no orders". | TODO Phase 11 |
| ☐ | **Winter DST** — before November set `CRON=5 22 * * FRI` | 21:05 UTC is only 5 minutes after the close in winter. | TODO Phase 10 |
| ☐ | **Execution quality** — raw market orders today | Adaptive algo or marketable-limit protects against bad opens. Not blocking at this size, but it is real money now. | TODO Phase 11 |

### Week 0 baseline — write these numbers down

You cannot detect drift without a starting point.

```bash
cd /opt/trading-agent && docker compose exec -T trader python - <<'EOF'
from trading.memory.store import default_store
m = default_store(); s = m.stats()
print("journal", s["journal"], "| episodes", s["episodes"],
      "| lessons", s["lessons"], "| predictions", s["predictions"])
print("calibration:", m.calibration())
EOF
```

| Metric | Day 0 value |
|---|---|
| account equity | |
| open positions | |
| predictions recorded / graded | |
| episodes | |
| lessons established / candidate | |
| agents with a calibration row | |

---

## Every trading day — 5 minutes

### 1. Features working

| ☐ | Check | Pass |
|---|---|---|
| ☐ | Ops channel silent, or every alert understood | silence = healthy; an alert you cannot explain is an incident |
| ☐ | Daily summary arrived (20:10 UTC) | present, and the P&L matches the dashboard |
| ☐ | Cycle ran and you approved or rejected it deliberately | no cycle auto-expired unattended |
| ☐ | `/health` and `/positions` respond | bot alive |

### 2. Nothing stale

| ☐ | Check | Pass |
|---|---|---|
| ☐ | Dashboard "SYSTEM" panel ages | broker snapshot < 1h, news < 24h, macro < 24h |
| ☐ | Price cache reached yesterday's close | ladder `as_of` is the last trading day |
| ☐ | No "⚠️ price cache still stale after refresh" alert | absent |

### 3. No abnormal behaviour — read every fill

This is the one that catches the thing nobody predicted. **Look at each
trade and say out loud why it happened.**

| ☐ | Check | Red flag |
|---|---|---|
| ☐ | Every fill maps to a strategy signal you can name | a name you do not recognise |
| ☐ | No position exceeds `MAX_POSITION_PCT` of equity | any breach = stop and investigate |
| ☐ | No short positions | system is long-only; a negative quantity is a bug |
| ☐ | No duplicate/stacked orders in the same name | order stacking shorted two names on paper once before |
| ☐ | Fill prices near the close you expected | > 1% slippage on a liquid name needs an explanation |
| ☐ | Currency: no unexpected USD cash negative | margin auto-loan starting |

---

## Every week — 30 minutes, Saturday

### 4. Is it learning?

The whole reason for the first month. Run this and compare to last week:

```bash
cd /opt/trading-agent && docker compose exec -T trader python - <<'EOF'
from trading.memory.store import default_store
m = default_store(); s = m.stats()
print("predictions:", s["predictions"], "| episodes:", s["episodes"], "| lessons:", s["lessons"])
print("\ncalibration (agent, n, hit rate, brier):")
for c in m.calibration(): print(" ", c)
print("\nestablished lessons:")
for r in m.lessons(status="established")[:10]: print(" -", r["statement"][:160].replace("\n"," | "))
EOF
```

| ☐ | Check | Pass | Fail means |
|---|---|---|---|
| ☐ | `predictions graded` **increased** | strictly higher than last week | the grader is dead again |
| ☐ | `calibration` has a row per active agent | 8 rows eventually | predictions recorded but not scored |
| ☐ | Brier scores are moving | not frozen | grading stuck |
| ☐ | `episodes` increased if anything closed | matches your round-trips | `add_episode` not firing |
| ☐ | Historian ran Friday | a `historian` journal row dated this week | never-scheduled bug returned |
| ☐ | New lessons carry **applies_when / fails_when / retire if / sample** | all four present | historian is proposing superstitions |
| ☐ | `/edge` returns numbers | picks vs passes populated | shadow ledger not grading |
| ☐ | No lesson established on < 20 distinct names | check the `sample:` line | over-fitting |

**The trap:** a lesson that reads as obviously true. Your own
buy-quality-after-drawdowns idea tested at +5.8% pooled over eight years
and *below 50% hit rate every year since 2023*. If a new lesson has no
stated failure mode, it has not been thought through.

### 5. Configs respected

| ☐ | Check | How |
|---|---|---|
| ☐ | Effective limits match `.env` | `uv run trading status` — remember **`risk.yaml` portfolio numbers are NOT read by the risk manager** |
| ☐ | No `/hold` pin silently dropped | `/holds` |
| ☐ | Mandates expired as expected | `/mandates` |
| ☐ | Universe refreshed Sunday | membership-delta alert arrived, or "no changes" in logs |
| ☐ | Model routing unchanged | `AGENTS_MODEL*` unchanged unless you changed it |
| ☐ | API credit remaining | a zero balance returns HTTP 400 and silently kills the committee |

### 6. Pipelines end to end

Trace **one** real decision all the way through. Pick a fill from this week:

| ☐ | Stage | Where to look |
|---|---|---|
| ☐ | signal | `/signal` — was the name on the ladder? |
| ☐ | selection | cycle report — did it make the top-K? |
| ☐ | risk decision | order store — accepted, scaled, or blocked, and why |
| ☐ | order | `/orders` — one order, correct side and size |
| ☐ | fill | `/positions` — quantity matches |
| ☐ | snapshot | dashboard equity moved by roughly the right amount |
| ☐ | memory | an `episode` when it closes |

If any stage cannot be traced, **that is the finding.** Write it down.

---

## Every month

| ☐ | Check | Pass |
|---|---|---|
| ☐ | Realised vs expected slippage | within the 5bps model, or update the model |
| ☐ | Live vs paper divergence | same strategy, same universe — large gaps mean an execution problem, not an alpha problem |
| ☐ | Drawdown vs `MAX_DRAWDOWN_PCT` | never approached without the halt firing |
| ☐ | Restore drill | `docs/restore.md` — restore state to a scratch dir and confirm it loads |
| ☐ | Disk trend | `df -h` — Parquet and logs grow forever |
| ☐ | Read `docs/incidents.md` | every incident this month written down |

---

## Incident rules — agree these with yourself now, not at the time

| Trigger | Action |
|---|---|
| A trade you cannot explain | `/halt` immediately, investigate before resuming |
| A position over its cap | `/halt`, flatten manually via IBKR |
| Any short position | `/halt` — long-only invariant broken |
| Gateway down > 1 hour in market hours | flatten manually if holding, do not wait |
| Two consecutive days with `graded_today: 0` | learning loop dead — not urgent for money, urgent for the experiment |
| Ops alert you do not understand | treat as an incident until proven otherwise |

**Halting is free. Being wrong about why a trade happened is not.**

---

## What "success" looks like after one month

Not profit. Profit over one month on 10k is noise, and treating it as a
verdict is the single most likely way to fool yourself.

Success is:

- every trade explainable,
- no cap breached, no invariant violated,
- prediction count graded rising every week,
- at least one lesson established *with conditions*, or an honest none,
- at least one bug found — **if you found nothing, you were not looking**,
- and the incidents log has entries.

A clean month with an empty incidents log usually means the monitoring is
broken, not that the system is perfect. That is the lesson of 2026-08-06.
