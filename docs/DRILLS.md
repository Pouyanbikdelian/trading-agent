# Pre-live drills — GO_LIVE.md §2, ready to fire

Run these on the VPS against the PAPER book, during US market hours
(15:30–22:00 CET) so the gateway has a live session and fills are real.
Each drill: setup → action → PASS criteria → restore. Run them in order;
stop and investigate on any FAIL. Nothing here touches live — the gates
stay `paper`/`false` throughout.

Every drill starts from a healthy baseline:

```bash
ssh trading-vps
cd /opt/trading-agent
docker compose ps                        # all Up/healthy
docker compose exec trader trading status # env=paper, live armed? no
```

Keep Telegram open — half of what we're testing is that alerts arrive.

---

## Drill 1 — Kill-switch (daily-loss and drawdown halts)

The halts compare equity against the day's open (daily loss) and the
all-time high (drawdown). We can't order the market to drop 2%, so we
temporarily shrink the threshold so normal intraday noise trips it.

**Setup** (nano `.env`):

```
MAX_DAILY_LOSS_PCT=0.0001        # was 0.02 — any tick down trips it
```

```bash
docker compose up -d --force-recreate trader
```

**Action**: force a cycle so the risk manager evaluates now:
Telegram → `/cycle` (or wait ≤1 cycle).

**PASS when ALL of:**
- `state/halt.json` exists with `halted: true` and a daily-loss reason:
  `docker compose exec trader cat /app/state/halt.json`
- Telegram alert announcing the halt arrived.
- A follow-up `/cycle` submits NOTHING (log line shows halted refusal):
  `docker compose logs trader --tail 50 | grep -i halt`
- `/status` shows halted.

**Restore**:

```
MAX_DAILY_LOSS_PCT=0.02          # back in .env
```

```bash
docker compose up -d --force-recreate trader
```

Telegram → `/resume`, then `/status` shows active.

**Repeat once** with `MAX_DRAWDOWN_PCT=0.0001` instead (restore to 0.15
after) — same PASS criteria, drawdown reason in halt.json.

---

## Drill 2 — Startup reconciliation (broker-vs-state drift)

**Setup**: none — needs open positions (we hold MU, SNDK).

**Action 1 (clean restart)**:

```bash
docker compose restart trader
docker compose logs trader --tail 100 | grep -i -E "reconcil|drift"
```

**PASS**: reconciliation runs at startup, reports zero drift, no alert.

**Action 2 (forced drift)**: while trader is STOPPED, change the broker
book behind its back:

```bash
docker compose stop trader
```

Telegram is down with it — so place the order in the IBKR web portal or
TWS on the PAPER account: buy 1 share of anything liquid (e.g. 1 SPY).

```bash
docker compose start trader
docker compose logs trader --tail 100 | grep -i -E "reconcil|drift"
```

**PASS**: startup reconciliation detects the extra position, logs the
drift explicitly, and a Telegram alert arrives. The runner must NOT
silently adopt or silently ignore it.

**Restore**: sell the 1 share in the portal/TWS, restart trader once
more, confirm zero drift.

---

## Drill 3 — Order idempotency (crash mid-cycle)

**Setup**: pick a symbol the strategy will rebalance, or just rely on
`/cycle` producing at least one order (if the basket is unchanged, first
`/hold`-release or a tiny `/k` change can provoke turnover — ask Claude
before improvising here).

**Action**:

```bash
# fire a cycle and kill the trader ~2s in, mid-submission window
docker compose exec -d trader sh -c 'sleep 0'   # warm-up no-op
```

Telegram → `/cycle`, then immediately:

```bash
sleep 2 && docker kill trader
docker compose up -d trader
```

**PASS when ALL of:**
- After restart + reconciliation, each intended order exists ONCE:

```bash
docker compose exec trader python -c "
import sqlite3;c=sqlite3.connect('/app/state/orders.db')
rows=c.execute('SELECT client_order_id,status,created_at FROM orders ORDER BY created_at DESC LIMIT 12').fetchall()
print(*rows,sep=chr(10))
dupes=c.execute('SELECT client_order_id,COUNT(*) FROM orders GROUP BY client_order_id HAVING COUNT(*)>1').fetchall()
print('DUPES:',dupes)"
```

  `DUPES: []` is the pass.
- IBKR portal/TWS order history shows no doubled fills for the cycle.
- Positions after restart match what `/status` believes.

**Restore**: nothing — paper book continues.

---

## Drill 4 — IB Gateway death (self-heal + reconnect)

**Action** (during market hours):

```bash
docker stop ibkr-gateway
docker compose logs trader --tail 20        # should start failing/reconnecting
```

Wait ~2–4 min.

```bash
docker compose ps                            # gateway back (autoheal/restart)?
docker compose logs trader --tail 50 | grep -i -E "reconnect|connected"
```

**PASS when ALL of:**
- Gateway container comes back WITHOUT manual help (autoheal or
  restart policy) and reaches healthy.
- Trader reconnects by itself and the next `/cycle` works end-to-end.
- A Telegram alert fired about the disconnection (silence = FAIL: a
  dead gateway at 3am must never be discovered at 9am).

**Restore**: none if self-heal worked. If it didn't:
`docker compose up -d ib-gateway` — and file the failure; that's a
go-live blocker.

---

## After all four pass

```bash
cd /opt/trading-agent && git tag live-candidate-1 && git push --tags
```

Then tick the four drill boxes in GO_LIVE.md §2 with the date. The
remaining §2 items (CHF sizing check, pins review, cron/timezone audit)
are desk work — do them with Claude in a session, no market hours needed.
