# Go-live runbook

Seven gates. Each has a pass criterion. **Do not pass a gate on "it didn't
error" — pass it on the stated evidence.** If a gate fails, stop there;
the later gates assume the earlier ones held.

Everything runs on the VPS unless marked LOCAL.

```bash
ssh trading-vps
cd /opt/trading-agent
```

---

## Gate 1 — Deploy the new code, still on paper

Nothing since `4a4bb49` is on the VPS. Fixes A–D exist only on GitHub.

```bash
cd /opt/trading-agent
git pull
docker compose build trader          # ~5 min
docker compose up -d --force-recreate
```

Then:

```bash
git log --oneline -1                 # expect 95a8193
docker compose ps
docker compose logs --tail 60 trader
```

**Pass when:** commit is `95a8193`, every container is `Up`/`healthy`, and
the trader log ends on a scheduler line rather than a traceback.

Why paper first: A–D have never touched a real broker. If the new code is
broken, it must break against play money.

---

## Gate 2 — Prove A–D actually run

Deployment is not verification. Four checks, one per fix.

**A — overnight fills reconcile.** The bug left orders stuck at
`submitted` forever (all of Jun 10 and Jul 14). First run should back-fill
them.

```bash
docker compose exec trader python -c "
from trading.execution.store import OrderStore
import os
s = OrderStore(os.environ.get('STATE_DIR','/app/state') + '/orders.db')
rows = s.all_orders()
from collections import Counter
print(Counter(r['status'] for r in rows))
"
```

**Pass when:** the `submitted` count is 0, or every remaining one was
placed today.

**B — the execution lock exists and is unheld.**

```bash
docker compose exec trader ls -la /app/state/execution.lock 2>/dev/null \
  && docker compose exec trader cat /app/state/execution.lock; echo
```

**Pass when:** the file exists after the first cycle (it is created on
first acquisition). Contents naming a long-dead pid are fine — flock is
released by the kernel, the JSON is only a label.

**C/D — the pre-cycle readiness job is scheduled.**

```bash
docker compose logs trader | grep -iE "precycle|broker.?ready|scheduled job" | tail -20
```

**Pass when:** a job is registered roughly one hour before your `CRON`.
With `CRON=5 21 * * FRI` that is **20:05 UTC Friday**.

**Universe refresh is importable** (the bug that broke twice):

```bash
docker compose exec trader python -c "
from trading.data.universe_refresh import refresh, out_path
print('importable:', callable(refresh)); print('writes to:', out_path())
"
```

**Pass when:** it prints a path under `state/`, not `config/`.

---

## Gate 3 — Seed the live state directory

Live gets its own `STATE_DIR`. That is deliberate: `runner.db` holds the
daily-equity baseline the kill switch computes from, and mixing ~1.07M CHF
paper snapshots with ~88k live ones makes the daily-loss stop either fire
instantly or never.

But `state/memory/` also lives under `STATE_DIR`. Moving wholesale would
restart your **lessons, predictions, source trust and shadow ledger from
zero** — months of accumulated learning, gone silently.

- Must be **fresh**: `runner.db`, `orders.db`, `halt.json`
- Must **carry over**: `memory/`

```bash
docker compose exec trader sh -c '
  mkdir -p /app/state/live &&
  cp -r /app/state/memory /app/state/live/memory &&
  ls -la /app/state/live && echo "---" &&
  ls /app/state/live/memory'
```

**Pass when:** `state/live/memory/` contains the same files as
`state/memory/`, and `state/live/` contains **no** `runner.db`.

---

## Gate 4 — Dry run: the live code path, paper money

This is the gate that matters. `trader-live` has **never been started**.
It uses host networking, a different client id, a different state dir and
a different command line from the paper trader. Every one of those is
untested.

The safety here comes from the **gateway being in paper mode**, not from a
flag — `trader-live` hardcodes `ALLOW_LIVE_TRADING=true` in compose. So we
point the live service at the paper gateway.

In `.env`:

```bash
IBKR_TRADING_MODE=paper
IBKR_PORT=4002                 # paper port
STATE_DIR=/app/state/live
REQUIRE_CYCLE_APPROVAL=true
IBKR_LIVE_CLIENT_ID=37
```

Then swap the runners — never both at once, they would submit into the
same account:

```bash
docker compose stop trader
docker compose --profile live up -d --force-recreate trader-live
docker compose logs -f trader-live
```

Trigger a cycle by hand rather than waiting for the cron:

```bash
docker compose exec trader-live touch /app/state/run_cycle_now
docker compose logs -f trader-live
```

**Pass when:** it connects to the gateway, reads the account, produces a
proposed batch, and **asks for approval instead of submitting**. Approve
it from Telegram and confirm the fills land in the paper account.

If it cannot connect, the fault is host networking or the client id — fix
it here, where it costs nothing.

---

## Gate 5 — Clear the decks

Two decisions before arming.

**Open paper positions (WMT, VST).** The live account starts flat. Choose:

- `/hold` both and let the live system ignore them, or
- let the next paper cycle flatten them, or
- close them manually now.

Cleanest is to flatten paper before arming live, so the two books do not
look like one.

**The cron.** Currently `5 21 * * FRI` = 21:05 UTC — that is **after the US
close** (20:00 UTC in summer). Post-hours execution on IBKR works but fills
are thinner and spreads wider. For the first live month, consider
`45 19 * * FRI` (19:45 UTC, 15 min before the close) so the first real
orders execute in full liquidity.

**Pass when:** the paper book is in a state you understand, and you have
consciously chosen the cycle time.

---

## Gate 6 — Arm live

Set the full LIVE DAY block in `.env`:

```bash
TRADING_ENV=live
ALLOW_LIVE_TRADING=true
IBKR_TRADING_MODE=live
IBKR_PORT=4001                     # live port; healthcheck follows this
IBKR_USERNAME=<paper username WITHOUT the trailing "2">
STATE_DIR=/app/state/live

# ~10k sleeve on an ~88k account. These are FRACTIONS OF EQUITY.
MAX_GROSS_EXPOSURE=0.11            # ≈ 9.7k deployed
MAX_POSITION_PCT=0.015             # ≈ 1.3k per name
MAX_DAILY_LOSS_PCT=0.006           # halts ≈ -530  (≈5% of the sleeve)
MAX_DRAWDOWN_PCT=0.02              # halts ≈ -1.8k (≈18% of the sleeve)
MAX_MARGIN_BORROWING_PCT=0.0       # cash-account behaviour
REQUIRE_CYCLE_APPROVAL=true        # you approve every batch by hand
```

The kill-switch percentages matter as much as the exposure ones. At the
`0.02` default, the daily stop would not fire until **-1,765** — 18% of a
10k sleeve, which is effectively never.

```bash
docker compose stop trader                       # paper runner OFF
docker compose up -d --force-recreate ib-gateway # re-auth in LIVE mode
# → approve the IBKR Mobile 2FA prompt now
docker compose --profile live up -d --force-recreate trader-live
```

Verify the limits that are **actually loaded**, not the ones you typed:

```bash
docker compose exec trader-live trading status
```

**Pass when:** it reports `env=live`, `live armed=True`, and the
sized-down limits above. A plain `docker compose restart` does not re-read
`.env` — always `up -d --force-recreate`.

---

## Gate 7 — Supervise the first live cycle

- **T-60 min:** the broker-readiness check runs. Silence means healthy; a
  🔴 Telegram alert means tap IBKR Mobile now.
- **T-0:** the cycle proposes a batch and waits for your approval.
- **Read the batch before approving.** Total notional should land near the
  sleeve, not near the account. If you see ~88k of orders, the limits did
  not load — **do not approve**, go back to Gate 6.
- **After fills:** confirm positions on the live account match what you
  approved.

`/halt` is your stop at any moment. It writes `halt.json` under
`STATE_DIR`, which is why Gate 6 sets `STATE_DIR` in `.env` and not in the
compose service — so the bot writes where the live runner reads.

---

## First month

Approval stays on. Watch for:

- a cycle that submits nothing without saying why
- positions that never change for 3+ cycles (the staleness telemetry)
- the daily equity curve diverging from what the dashboard claims
- any 🔴 ops alert

Only after a clean month: consider turning off `REQUIRE_CYCLE_APPROVAL`
and stepping the sleeve up.

---

## Known, deferred

- `test_exec_lock.py` forks a multi-threaded process — a `DeprecationWarning`
  on macOS/Python 3.12, an error in 3.14. Switch to `spawn`.
- Commit `d30087d`'s message lost the words "live run" to a zsh backtick
  substitution. Cosmetic; not worth a force-push.
- Winter DST will move the US close an hour; the cron does not follow it.
