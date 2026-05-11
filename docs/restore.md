# Restore-from-state drill

You should run this **before** going live — and again every 60–90 days as
a fire drill. The goal: take a fresh server and a backup archive, and end
up with a runner that resumes trading without re-submitting yesterday's
orders or forgetting yesterday's halt state.

> **The whole point of persistence is to survive a crash.** This drill
> proves that the survival actually works. If you don't run it, you don't
> know what breaks until it matters.

## What persists, and where

Three SQLite databases and one JSON file. All live under `/app/state/`
inside the container, which is the `state` docker volume:

| File                       | Owner module             | What it carries                                |
|----------------------------|--------------------------|------------------------------------------------|
| `state/orders.db`          | `execution.store`        | Every order + every fill, keyed by `client_order_id` (PK enforces idempotency on re-submit). |
| `state/runner.db`          | `runner.state`           | One `account_snapshots` row per cycle (cash, equity, positions); one `cycles` row per cycle attempt with risk decisions and outcome. |
| `state/halt.json`          | `risk.manager`           | `halted` flag, reason, equity high-water mark, daily opening equity — the kill-switch's persistent memory. |
| `state/heartbeat.json`     | `runner.heartbeat`       | Liveness ping — *not* state, just monitoring. Safe to delete. |

Notably **NOT** persisted:

- The Parquet cache (`data/parquet/...`). Rebuildable in one
  `trading data fetch` call; backing it up is bandwidth-saving, not
  correctness-critical.
- In-memory `Simulator` state. Paper-trading restart loses position
  history; live (`IbkrBroker`) reconciles from the broker.

## The drill

### 0. Set up a *separate* host

Use a second VPS / VM / your laptop. Do **not** restore onto the live
host you're trying to verify.

### 1. Pick a backup

```bash
ls -lh /var/backups/trader/
# state-2026-05-10.tar.gz   12K
# state-2026-05-11.tar.gz   13K
# ...
```

Choose one. Daily granularity is fine because the runner reconciles
against the broker at each cycle anyway — even a 24-hour-old state will
"catch up" by reading IBKR's fills.

### 2. Prepare the new host

Follow steps 1–4 of [deploy.md](./deploy.md) (provision, harden, install
Docker, clone repo). Stop before `docker compose up`.

### 3. Drop in the state

```bash
docker compose up --no-start          # creates the volumes
docker run --rm \
    -v trading-agent_state:/state \
    -v /path/to/state-2026-05-11.tar.gz:/backup.tgz:ro \
    alpine:3 \
    sh -c 'cd / && tar xzf /backup.tgz --strip-components=1'
```

> **Strip-components matters.** The backup tar in `deploy.md` archives
> `state/` as the top directory; without `--strip-components=1` you get
> `/state/state/orders.db` and the runner can't find it.

Sanity check:

```bash
docker run --rm -v trading-agent_state:/s alpine:3 ls -la /s
# orders.db    runner.db    halt.json
```

### 4. Verify the halt state

A restore that comes up *unhalted* when it should have been halted is
dangerous. Confirm explicitly:

```bash
docker run --rm -v trading-agent_state:/s alpine:3 cat /s/halt.json
```

If the source server was halted, the new server should be halted too.
The runner picks `halted=true` up on first cycle and emits no orders.

### 5. Verify the order ledger

```bash
docker run --rm -v trading-agent_state:/s alpine:3 \
    sqlite3 /s/orders.db \
    'SELECT status, COUNT(*) FROM orders GROUP BY status;'
```

You should see roughly the same counts as the source server reported.
Any `submitted` orders are the ones the original server *thought* it sent
but never saw filled — the runner will re-check these against IBKR on
the next cycle (and IBKR's `client_order_id` index will reject duplicate
submissions).

### 6. Verify the equity curve

```bash
docker run --rm -v trading-agent_state:/s alpine:3 \
    sqlite3 /s/runner.db \
    'SELECT ts, equity FROM account_snapshots ORDER BY ts DESC LIMIT 5;'
```

The most-recent equity should match what you saw on the source server
right before the backup was taken. If they differ by more than a
rounding error, the backup is from a different machine and you've got a
bookkeeping issue — don't proceed.

### 7. Boot in `paper` first, even if the source was `live`

```bash
# .env on the new host
TRADING_ENV=paper
ALLOW_LIVE_TRADING=false
```

```bash
docker compose up -d
docker compose exec trader trading paper run us_large_cap --once
```

Confirm the cycle prints `halted` if it was halted, `ok` / `no_orders`
otherwise. The simulator broker will start with a fresh
`initial_cash`, but the **risk state** (HWM, daily open, halt flag) came
across — that's the part we wanted to prove.

### 8. Cut over

If everything looked sane:

1. On the **original** host: `docker compose down`. This stops the
   runner cleanly.
2. On the **new** host: update `.env` to match (TRADING_ENV, ports,
   ALLOW_LIVE_TRADING).
3. Point DNS / your operator inventory at the new host.
4. `docker compose --profile live up -d` if going straight back to live.
5. Watch the first cycle on the new host. Confirm orders submitted ==
   orders expected; confirm Telegram alerts arrive.

### 9. Document what you learned

Drift between docs and reality is how restores fail. Edit this file with
anything you tripped over. The next person doing the drill (which might
be you in six months) will thank you.

## Failure modes worth rehearsing

- **Halt file present but DBs missing.** Runner boots, sees a halt flag,
  refuses to trade — correct. Confirm you can `unhalt()` only after
  manually verifying broker positions.
- **DBs present but halt file missing.** Runner boots fresh-halt-state
  (`halted=false`, HWM=0). The first cycle will set HWM = current equity
  and *not* fire the drawdown halt for a recent drawdown. This is the
  worst restore-state combo — back up `halt.json` *with* the DBs, and
  alert on its absence.
- **DBs present but `client_order_id` collisions** (same id, different
  payload). Should not happen — `client_order_id` is generated with
  `uuid.uuid4()` — but if it does, the SQLite PK on `orders` will reject
  the duplicate. Confirm via the count check in step 5.
- **Disk full mid-write.** Atomic-write of `heartbeat.json` is via
  `tmp + rename`, which is safe; SQLite is also atomic per transaction.
  But a `.db-wal` file can linger; the next open replays it.
