# Incidents

Chronological log of production incidents, what we thought, and what was
actually broken. Newest at the top. Useful for:

- Operators triaging similar symptoms later.
- AI assistants and new contributors: when an issue *looks* identical to
  a past one but the root cause is different, this file is the
  tie-breaker.

The pattern in every entry: **Symptoms → Wrong theories → Actual root
cause → Fix → Lessons.** The wrong theories are the most valuable part —
they're what you'll think next time too.

---

## 2026-08-18 — A loss halt became an information blackout

**TL;DR:** The account dropped through the −0.60% daily-loss limit at
14:54 UTC. Nothing halted until the operator manually ran `/cycle` at
19:28 UTC — four and a half hours later — because **risk was only ever
evaluated inside a cycle**. Worse, once halted, the command gate refused
two automatic trailing-stop exits (`GEV`, `OUST`) along with `/close` and
`/flatten`: the kill switch that fired to protect the book had also
removed every way of reducing it. `/resume` → `/cycle` correctly
re-halted, which looked to the operator like a broken bot.

### Symptoms

- Live equity CHF 86,087.26 against a stored baseline of CHF 86,989.82
  (−1.04%); the halt when it finally landed read −1.09%.
- `/resume` reported success, then the next `/cycle` halted again after a
  two-minute price refresh — twice, with no explanation of why.
- Two guard-triggered closes rejected with "risk manager halted".
- `/pm run` returned a `$1.09M` book in USD next to a CHF account with
  ~CHF 86k in it, and its prose claimed a 70% cap above a 77% target.

### Wrong theories

1. **"The order queue is stuck."** It was not: `/cycle` ran, evaluated
   risk, and correctly declined. Zero orders is not the same as zero work.
2. **"The gateway outage blocked it."** There was a brief outage, but it
   resolved before either cycle.
3. **"`/resume` didn't clear the halt."** It did. `/resume` deliberately
   preserves the daily-loss baseline, so the next cycle re-measured the
   same breach and re-halted. Working as designed, explained nowhere.

### Actual root causes

1. **Risk was cycle-scoped.** The 60-second snapshot refresh persisted
   equity but never asked the risk manager anything. With a weekly cron
   the blind spot is up to a week wide. The 2% drawdown threshold was
   crossed around 15:00 UTC with the same non-result.
2. **The halt gate was all-or-nothing.** `_ORDER_SUBMITTING_COMMANDS` was
   refused wholesale, so exposure-reducing commands died with the
   exposure-increasing ones.
3. **The daily baseline had no provenance.** It was stamped by whichever
   process observed the account first — the same class of bug as
   2026-08-07, one layer up.
4. **The PM's prose was treated as its decision.** Hard caps were applied
   after the model spoke, and nothing reconciled the two.

### Fix

- `Runner._monitor_live_account_risk` evaluates every 60-second snapshot
  and halts atomically the moment a limit breaks.
- `HaltState` carries `daily_baseline_session/captured_at/source/currency`.
  `RiskManager.capture_session_open` will only stamp a baseline inside a
  five-minute window at the **real NYSE open** (`runtime/nyse_session.py`),
  and `evaluate_session_risk` refuses to execute against any baseline it
  cannot prove came from there.
- `trading/risk/reduce_only.py` is the single proof that an order lowers
  exposure. `/close` and `/flatten` use it to stay available during a
  halt; the cycle uses it for the defensive path below.
- **Defensive cycles.** A `/cycle` run while halted builds the real
  CHF basket, strips it to the reduce-only subset, and asks for approval.
  Buys can never survive the filter, approval is mandatory regardless of
  `REQUIRE_CYCLE_APPROVAL`, and the proof is repeated against a fresh
  broker snapshot after the approval wait. Off switch:
  `ALLOW_DEFENSIVE_CYCLE_WHEN_HALTED=false`.
- `/review` renders a strictly non-executable card from a separate state
  file, so an approval can never be built from it.
- The PM's hard constraints now bind before its rationale is published;
  when they change the target, the operator-facing "why" is generated
  from the executed facts and the model's original text is retained as
  `model_rationale` for audit.

### Lessons

1. **A check that runs only on the execution path is not monitoring.**
   It answers "may I trade right now", which is a different question from
   "is this account in trouble", and only the second one has a deadline.
2. **A kill switch that blocks exits is not a kill switch.** Halting must
   be asymmetric: refuse anything that adds risk, permit anything that
   provably removes it. "Block everything" felt conservative and left the
   book fully exposed for four hours.
3. **Refusing is only half the job; the other half is saying what would
   change the answer.** `/resume` reporting success and `/cycle` promising
   "orders in ~4 minutes" while neither could trade is what turned a
   correct risk decision into a lost evening.
4. **Never let a model's prose be the record of what was executed.** Cap
   the numbers first, then describe what actually happened.

**Operator note.** After this deploy the runner must be up during the
NYSE open (13:30 UTC) to capture a trusted daily baseline. Restart it
mid-session and cycles stay review/reduce-only until the next open —
by design, and announced in Telegram.

---

## 2026-08-07 — First live session lasted seven minutes

**TL;DR:** Went live on IBKR account U15995232 at ~14:44 UTC, PM-only
(`AGENT_PM_SLEEVE_PCT=0.11`, `STRATEGY_SLEEVE_PCT=0.0`), ~10k cap. Two
incidents inside seven minutes. The position guards sold a personal VST
position, and the kill switch halted at "daily loss -91.82%" against a
baseline written by a paper cycle an hour earlier. Neither was a code
defect. Both were operator sequencing, and both are now enforced by
code because a runbook step is only as good as the day nobody skips it.

### Symptoms

- 63 shares of VST sold at ~139.12 (~8,765 USD) minutes after arming.
  Nothing in the PM's decision mentioned VST.
- Telegram: `HALT: daily loss -91.82% breaches limit -2.00%` on an
  account that had made one trade.

### Wrong theories

- *"The PM hallucinated a ticker and sold VST."* The bridge does run the
  PM's picks through the same risk manager as everything else and the PM
  can name a symbol it does not hold — but the PM never emitted VST. The
  seller was `runtime/guards.py`, a subsystem neither the PM nor the
  strategy knows about.
- *"Turning the guards off would have prevented it."* Only half. The
  rebalance sells anything whose target weight is zero, and every symbol
  the strategy did not pick has a target weight of zero. `/hold` is the
  only thing that blocks both paths.
- *"The account really did drop 91%."* It did not move. The comparison
  was against another account's equity.
- *"Clearing the daily figure fixes the halt."* It re-halted
  immediately. `equity_high_watermark` was poisoned by the same write,
  and at `MAX_DRAWDOWN_PCT=0.02` an 87k account against a 1.07M peak
  breaches on the next bar. The second half of a poisoned baseline is
  the half you find out about later.

### Actual root causes

**1. A paper cycle wrote into the live state directory.** `STATE_DIR`
was switched to `/app/state/live` an hour before arming, to dry-run the
live code path — while the gateway was still in PAPER mode. That cycle
stamped CHF 1,068,862 of paper equity into `daily_equity_open` and
`equity_high_watermark`. `start_of_day` is idempotent per date, so when
the live session opened an hour later `last_day` already equalled today
and the baseline was never re-taken. A state directory is just files: it
carried no record of which account wrote it, and nothing could tell that
the two figures came from different places.

**2. Pre-existing personal positions were never `/hold`-ed.** The
position guards (ATR trailing stop + portfolio ratchet, `GUARDS_ENABLED`)
apply to every position in the account regardless of who opened it. On
paper they had only ever seen system positions, so nothing had made them
visible — Yan did not know they existed. `guards.py:190` exempts symbols
in `holds.json`, so `/hold VST` would have prevented the sale. The
runbook had "resolve WMT/VST with /hold" as a gate; it was dropped when
the scope narrowed to PM-only.

### Fix

- `core/state_env.py` — state directories are stamped with the
  `TRADING_ENV` that wrote them. A mismatched runner raises
  `StateEnvMismatchError` at `Runner.from_config`, before anything reads a
  baseline. Unstamped directories are adopted (deployments have
  several); a mismatch is fatal and never auto-resolved, because every
  way of self-healing amounts to guessing which account the numbers on
  disk belong to. `trading state check` / `trading state stamp`.
- `risk/manager.py::start_of_day` — the date is no longer the only thing
  that decides. A stored baseline more than
  `BASELINE_SANITY_DIVERGENCE_PCT` (50%) from actual equity is re-stamped
  whatever the date says, and BOTH figures go. Returns a note; the cycle
  raises it as a critical alert, because a baseline silently repaired is
  a baseline nobody investigates.
- `runtime/broker_ready.check_unheld_positions` + `Runner.preflight_unheld`
  — the live runner refuses to start while any account position is
  neither in `holds.json` nor explicitly acknowledged. `trading preflight
  check` lists them; `trading preflight ack` records the exact symbol set,
  so a position opened later raises the question again.
- GO_LIVE.md §3.0 — both rules written as hard gates at the top of the
  live-day checklist.

**Recovery on the day:** backed up `state/live/` to a tarball, deleted
`halt.json`, `runner.db*`, `orders.db*`, `consecutive_errors.json`; kept
`memory/` and `agent_pm/`.

### Lessons

1. **A number that cannot be produced by the market is a measurement
   bug.** -91.82% on an account that had made one trade is not a loss.
   The kill switches were working perfectly; what they measured against
   was from somewhere else. Any limit worth halting on deserves a sanity
   check on its own baseline.

2. **Deleting `halt.json` clears the halt.** The system resumes silently
   — no announcement, no re-confirmation. That is its own trap and it
   fired during the recovery.

3. **A subsystem that has only ever seen its own positions has not been
   tested.** The guards ran for months on paper and looked well-behaved
   because every position they saw belonged to the system. Their real
   scope — the whole account — only became visible the first time the
   account contained something else.

4. **A gate that lives only in a checklist is not a gate.** The `/hold`
   step existed, in writing, and was dropped in good faith when the scope
   narrowed. The version that survives a change of scope is the one the
   process refuses to start without.

---

## 2026-08-07 (evening) — Nine defects found by auditing the live order path

**TL;DR:** No incident. After standing down from the morning's live
session, an audit of the code between a decision and an order found
nine defects, all on paths that execute. Two were visible in the
`halt.json` the account was sitting on at the time.

### 1. `/halt` promised a liquidation it never performed

`/halt` replied *"Next cycle will force-flatten positions"* and wrote
`flatten_on_next_cycle: true`. **Nothing read that key.** The only
force-flatten in the system comes from a playbook regime rule. The one
test that mentioned the flag asserted the bot had *written* it — which
is how a flag can be covered by a test and still do nothing.

Harmless the day it was found, because a flatten was not wanted. The
cost is the day someone halts in an emergency, reads the confirmation,
and walks away believing the book is closing.

Removed rather than implemented (Yan's call): halting is reversible,
liquidating is not, and one word should not market-sell a book. The
reply now states what is true, and names whether the guards are on —
"halted" reads like "safe", and a halted book with `GUARDS_ENABLED=false`
has nothing watching it at all.

### 2. Every halt and every resume wiped both kill-switch baselines

Four writers overwrote `halt.json` with a four-key dict. `HaltState`
carries three more — `equity_high_watermark`, `daily_equity_open`,
`last_day` — and neither forbids extra keys nor requires the missing
ones, so all three silently reloaded as `0.0 / 0.0 / None`.

The daily-loss switch skips a zero baseline outright
(`if daily_equity_open > 0`); the drawdown switch restarts its peak from
wherever equity happens to be. So **neither kill switch could fire on a
decline spanning a halt** — exactly the situation that someone halted
for. It is also, by accident, what cleared the poisoned CHF 1,068,862
watermark that morning.

All four writers now go through `risk/halt_file.set_halted`, which
read-modify-writes under the existing lock. The runner's auto-halt also
stopped stamping a naive `datetime.now()`.

### 3. `/close` and `/flatten` never got the 2026-07-15 fix

Netting working orders before sizing, and clamping sells so they cannot
cross zero, was applied to the **cycle only**. `/sell all`, `/close` and
`/flatten` size off settled shares and call `submit_order` directly, so
an order already working at the broker is invisible to them.

The guards make it reachable rather than theoretical: they run every 15
minutes through 20:59 UTC — past the US close — so their exits queue to
the next open. An operator who sees "🛑 Trailing stop VST — closing" and
reaches for `/close VST` sells the position twice, and a long-only
system ends up short. That is the morning's sequence plus one keystroke.

The execution lock does not help. It serialises *submissions*; it says
nothing about an order submitted an hour ago that is still working.

Fixed in `risk/presubmit.py`, enforced at the last point before the
broker against IBKR's own `openTrades`.

### 4. The PM could buy but never sell

A target weight of zero is how every book here says "get out". A name the
PM has **dropped** is absent from its weights, not zero — and an absent
key gets no delta in `signal_to_orders`, so no order, so the position
stays.

The mechanical strategy masks this for S&P names: its weight vector
spans the whole universe, so anything unpicked carries an explicit 0.0.
The PM's ETF shelf is outside that universe, and those are exactly what
it uses to express a hedge. Its first live decision was GLD 0.15 + XLV
0.12. `_add_pm_targets` compounded it by pricing only the PM's *current*
picks, so a dropped name was not even in `instruments_by_key` and a zero
target would have been rejected as unpriceable.

Net effect: the PM's book ratchets — adds, never removes — while the
stranded position keeps counting toward equity and gross and quietly
eats the sleeve. It would not have surfaced as an error anywhere; a
position no code path reaches is not a failure, it is an absence.

The PM now records the keys it directed on each tradeable cycle, prices
that set alongside the current one, and zeroes what it has dropped.

### 5. `/cancel` reported success for orders it had not cancelled

`cancel_order` matched on `orderRef` alone. `get_open_orders` — the
source for `/orders`, which is what the operator reads — falls back to
`broker-{orderId}` for anything without an orderRef, i.e. every order
placed in TWS by hand or by an older build. So the id on screen was one
`cancel` could not match. The miss was a silent no-op, while
`_h_cancel_order` returned `{"cancelled": True}` regardless.

Closes a loop with defect 3's fix, which tells the operator to "cancel
the working order first" — advice that would have failed silently. The
`Simulator` had raised on an unknown id all along; the two broker
implementations had quietly disagreed about the contract.

### 6. Force-flatten was the *third* path missing the netting fix

`force_flatten_orders` — reached by the playbook's regime rule and by
`/approve flat` — built a full-size close for every position with no
regard for orders already working. A close on top of a queued close
crosses zero: the panic button leaving you short the names you were
exiting.

Same bug as §3 and as 2026-07-15. Fixed in the cycle then, in the
command handlers this morning, here an hour later. The lesson written up
two entries above says to ask which other paths reach the invariant —
and it was written before this one was found.

### 7. Approvals never checked which cycle they belonged to

The bot stamps the pending cycle's id into every decision file. Nothing
compared it — written and discarded, exactly like `flatten_on_next_cycle`.
Inline keyboards already refuse a tap minted for another basket, but a
**typed** `/approve` had no equivalent, so scrolling back to an older
prompt could approve today's book. `_request_cycle_approval` also had no
tests at all, despite being the control the operator leans on for live.

### 8. The fill ledger stored the same execution once per reconciliation pass

`save_fill` was a plain INSERT with no uniqueness, and `_reconcile_from`
deliberately widens the window back to the oldest still-open order so an
overnight fill is not missed. Those two multiply: while any order is
open, every cycle re-reads and re-stores every fill in the window.

Not an edge case — `CRON=5 21 * * FRI` against a 20:00 UTC close means
orders queue to the next open most weeks.

The duplicates were silent and not inert. `fills` feeds
`fills_with_symbols` → `realized_by_symbol` (the Live tab's realized PnL
and fees) and `memory/episodes.py` (the historian's trade
reconstruction). Both counted the same execution repeatedly, so realized
PnL was wrong in an unpredictable direction — it depends on whether the
duplicated leg was a buy or a sell.

`Fill` now carries IBKR's `execId`; the ledger has a UNIQUE `dedupe_key`
and `INSERT OR IGNORE`. **The migration deletes existing duplicate rows**
— the realized-PnL figure on the dashboard will move when this deploys,
and that movement is the correction.

### 9. An order went terminal on its FIRST execution

`OrderStatus.PARTIAL` was defined, listed in `OPEN_STATUSES`, and set by
nothing. The cycle marked an order FILLED on any fill event; IBKR reports
each execution separately, so a 63-share order filling 20 then 43 went
terminal after 20.

`oldest_open_created_at` drives the reconciliation window, so an order
going terminal early stops holding it open — and the remaining 43 could
arrive after the window closed and never be recorded. Same divergence
`_reconcile_from` was built to prevent, reached by another route.

### Lessons

1. **A test that asserts a flag was written proves nothing about whether
   it is read.** `flatten_on_next_cycle` had coverage and no consumer.
   For anything safety-critical, the test that matters names the
   behaviour — "the book is flat afterwards" — not the byte on disk.

1c. **An enum member nothing ever assigns is a design that was planned
   and not finished.** `OrderStatus.PARTIAL` was defined AND wired into
   `OPEN_STATUSES` — the machinery to keep the window open on a partial
   fill existed and was simply unreachable. Grepping for writes of each
   state is a cheap audit that would have found it years earlier.

1b. **Absent is not zero.** Twice in one audit — the PM's dropped names,
   and the halt file's missing baselines. Any map keyed by "what we want"
   needs an explicit answer for the things we *stopped* wanting, or the
   omission reads as "no opinion" and nothing happens.

2. **A partial write to a shared state file is a silent reset of every
   field it omits.** Pydantic defaults turned three missing keys into
   three plausible zeros. `extra="forbid"` would not have caught this;
   only round-tripping the *whole* record does.

3. **A fix applied to one path is not applied to the invariant.** The
   long-only rule was enforced in the risk manager and, separately, not
   enforced in three command handlers that reach the same broker. Worth
   asking of every past fix: which other paths reach this?

---

## 2026-07-15 — Paper book goes SHORT two names at the open

**TL;DR:** Three independent order batches (a guard full-close and two
after-hours `/cycle` runs) sat unfilled at IBKR overnight, none aware of
the others, and all filled on top of each other at the open. MU ended
−63, SNDK −51. The cycle sized deltas against *positions* only — never
against *working orders* — and nothing enforced "long-only" as an
invariant.

### Symptoms

- Morning after two evening `/cycle` runs: MU −63, SNDK −51 in the
  paper account; AMD/ON/CIEN roughly doubled vs their target weights.

### Wrong theories

- "The second cycle will see the first's fills" — true intraday (fills
  land in the same session), FALSE after hours (everything queues).
- "The no-margin check would block it" — it guards cash going
  *negative*; short sells ADD cash, so they sail through.
- "Guard closes are in orders.db, the cycle can read them" — guard
  exits go through the command pipeline straight to the broker and are
  never persisted to orders.db.

### Actual root cause

Delta sizing used `account.positions` as "current," ignoring orders
already working at the broker. Compounding design gap: no long-only
invariant anywhere in the order path.

### Fix (2026-07-15)

1. `Broker.get_open_orders()` — cycle nets ALL working orders (any
   source) into effective position before sizing.
2. `RiskLimits.allow_short=False` (default): negative targets clamp to
   0, sells clamp to the effective position. Shorting is opt-in.

### Lessons

- After-hours automation must reason about *pending* state, not just
  settled state.
- Every "this strategy never does X" belief needs an invariant in the
  risk manager, not an assumption in the strategy.
- Cash-flow-based guards don't catch shorts.

---

## 2026-07-14 — Strategy silently traded NOTHING for a month

**TL;DR:** Two freshly listed symbols (FDXF, Q) with ~26 bars of history
joined the sp500 universe in June. The cycle's price matrix used an
inner join across all 503 symbols, so the whole matrix truncated to 26
rows — starving the 126-day momentum lookback. The strategy fell back to
"hold" and reported "no orders — portfolio already on target" for ~5
weeks while sitting 84% in cash.

### Symptoms

- Every cycle since 2026-06-10: "no orders — portfolio already on
  target". Last real order 2026-06-10.
- Log showed `prices_shape=(26, 503)` — visible for weeks, alarming to
  nobody, because nothing about it *looked* like an error.

### Wrong theories

- "Portfolio is genuinely on target" (the bot's own message).
- "The FX/no-margin rejection from June is back" — decisions said
  `allow`, so no.

### Actual root cause

`pd.DataFrame(series).dropna(how="any")` — one short-history column
truncates every row. A data-shape bug masquerading as a calm strategy.

### Fix

Short-history symbols are dropped from the matrix (loud
`dropping N short-history symbol(s)` warning) instead of truncating it;
regression test pins it. Backlog: a "no orders in N consecutive cycles"
watchdog — silence should never look healthy.

### Lessons

- Inner joins across a universe are a time bomb; every universe refresh
  can add a young symbol.
- "No orders" is a state needing positive confirmation, not a default
  message.
- The same week's audit found the daily-loss kill switch had never
  armed (stale baseline) and the sector cap had never bound (missing
  fundamentals file): **a safety that has never fired is untested, not
  working.** Drills exist to fire them on purpose.

---

## 2026-05-22 — IBKR `accountSummary` hangs at 30s for every cycle

**TL;DR:** Trader called `ib_async`'s sync API from a worker thread that
wasn't the thread running ib-async's event loop. Requests went into a
detached per-thread loop, responses landed on the main loop, the worker
waited forever. Every "broker timeout" all afternoon was a Python
asyncio mismatch, not a sick gateway.

### Symptoms

- Every cycle's `_fetch_account` step hung exactly 30 seconds, then:
  ```
  BrokerTimeoutError: IBKR accountSummary timed out after 30s —
  gateway likely has a dead broker session (try restarting
  ib-gateway container)
  ```
- After auto-halt at the 3-failure threshold, `/resume` cleared the halt
  but the very next `/cycle` failed identically and re-halted the trader.
- Reproducible across two paper sessions (`DUQ053932`), three full
  `docker compose down/up` cycles, and a freshly logged-in IB Gateway.
- The trader's `_reconnect_session` self-heal would re-handshake the
  socket cleanly (`reconnected ibkr@127.0.0.1:4002 client_id=17`) — then
  the retry of `accountSummary` *also* hung 30 seconds and triggered a
  gateway restart. The gateway restart succeeded. The next cycle, against
  the freshly-restarted gateway, still hung 30s. Infinite loop of
  symptom → mitigation → symptom.

### Wrong theories we burned hours on

1. **Docker socket permission (real, but a separate bug).** The trader
   container couldn't write to `/var/run/docker.sock` because the
   `group_add` GID default in `docker-compose.yml` was 999 and the host
   was 987. That blocked the gateway-restart self-heal from firing.
   Fixing it let the self-heal trigger, which surfaced the next symptom
   below — but didn't fix the underlying hang. *Commit `4059291`,
   `0290959`.*

2. **Stale `isConnected()` after gateway bounce.** `_reconnect_session`
   was calling `connect()`, which short-circuited on `self._ib.isConnected()`
   that ib-async briefly reported `True` even after the TCP teardown.
   Real bug, also fixed (we forced `connectAsync` unconditionally in
   the reconnect path), but irrelevant to the hang — the hang persisted
   even on cleanly re-handshaked connections. *Commit `7f89161`.*

3. **Gateway session is dead / Error 1100.** The error message text
   said exactly this ("gateway likely has a dead broker session"). It
   was our own message, written into `BrokerTimeoutError` as a guess.
   Wrong. The gateway was healthy.

4. **`clientId=17` poisoned on IBKR's side.** Plausible: IBKR sometimes
   refuses subscription requests from a clientId that has a phantom
   active session. Disproved by the diagnostic — using clientId=99
   showed the same hang under the same conditions.

5. **Paper account needs portal login / 2FA expiry / data permission
   revoked.** Plausible enough that we did a full `compose down` + portal
   login, which appeared to fix it (one diagnostic call returned in
   0.3s). That was a coincidence — the diagnostic was running directly
   on the main thread, so it never tripped the actual bug. Restarting
   the trader and running a real cycle hung again immediately.

### Actual root cause

`ib-async`'s `IB.connectAsync(host, port, ...)` binds the underlying
TCP transport to **whichever asyncio event loop awaited it**. Every
subsequent read/write on that socket must be scheduled on that same
loop. The transport doesn't keep a reference to the loop it can be
woken up on; the loop owns the transport's selector.

Our broker dispatched calls through this chain:

```
Cycle thread (APScheduler worker)
  └─ broker.get_account()
      └─ _bounded("accountSummary", ib.accountSummary, timeout=30)
          └─ _call_with_timeout — uses ThreadPoolExecutor for hard timeout
              └─ executor.submit(ib.accountSummary)
                  └─ runs on Worker Thread #2
                      └─ ib_async sync wrapper calls util.run(coro)
                          └─ util.run does asyncio.get_event_loop()
                              └─ no loop on this thread →
                                 creates a FRESH per-thread loop
                                 └─ reqAccountSummary scheduled on fresh loop
                                    Socket isn't on this loop.
                                    Bytes get queued but never sent.
                                    Response arrives on the original loop,
                                    no one listening on Worker Thread #2.
                                    Worker waits exactly until executor
                                    timeout (30s), raises TimeoutError.
```

The comment in our old `connect()` *thought* `util.startLoop()` set up
a persistent background loop in a daemon thread that all threads could
share. It doesn't. `util.startLoop()` only patches `nest_asyncio` (loop
reentrance) on the current thread and sets the loop as the policy's
default — but `asyncio.get_event_loop()` returns a per-thread instance
regardless.

The cached calls (`positions`, `fills`, `openTrades`, `accountValues`)
read from `ib.wrapper.<dict>` without awaiting — they survived the
thread boundary fine, which is why the trader could log "8 positions"
during startup reconciliation but then hang on `accountSummary` minutes
later. The fact that *some* calls worked was the biggest red herring.

### How we finally caught it

User's intuition: "*when I ran [the diagnostic] separately it didn't
work but when you gave full code... it worked.*"

Crucial observation. The script that worked (one fresh `IB()` on the
main thread, direct sync call) didn't reproduce the bug. Adding a
`ThreadPoolExecutor` around the sync call inside the same process
reproduced it deterministically in 20 seconds.

Same Python interpreter, same gateway, same clientId, same socket.
Only difference: which thread the sync call lived on.

### Fix

Dedicate one daemon thread (`ibkr-loop`) to own ib-async's event loop,
running `loop.run_forever()` for the broker's lifetime. All async work
dispatches to that loop via `asyncio.run_coroutine_threadsafe`. The
transport stays where it expects to be; every thread in the process
gets the same hard-timeout semantics through `future.result(timeout=…)`
on the returned `concurrent.futures.Future`.

Implementation: `IbkrBroker._ensure_ib_loop_thread`,
`IbkrBroker._await_async`, and `IbkrBroker._async_variant`. The last
one is the trick that keeps callers untouched: when a caller passes
`self._ib.accountSummary` (sync), we transparently substitute
`self._ib.accountSummaryAsync` (coroutine) and dispatch it. Lambdas
and methods without an `Async` sibling (cached reads, fire-and-forget
sends like `placeOrder`) fall back to the existing
`ThreadPoolExecutor` path because they don't await internally — no
cross-thread hang to fix.

Regression test:
`tests/execution/test_ibkr_timeouts.py::test_async_variant_dispatches_to_loop_thread`
asserts that when a stub has both `accountSummary` and
`accountSummaryAsync`, `_bounded` calls the async sibling and never the
sync one.

*Commit `6c0a90d`.*

### Lessons

1. **Trust your own error messages less.** The
   `"gateway likely has a dead broker session"` text inside
   `BrokerTimeoutError` was *our guess at causation*, written into a
   generic timeout class. Every theory we chased downstream took that
   guess as established fact. If you write a fallback error message,
   make it describe *the observed behavior*, not your best guess at the
   underlying cause: `"IBKR accountSummary timed out after 30s"` is a
   fact; `"gateway likely has a dead broker session"` is a hypothesis.

2. **ib-async is single-loop, single-thread.** Anywhere it's used in
   the codebase from a worker thread is suspect. The library's
   `util.startLoop()` name oversells what it does — it doesn't spin up
   a background loop, it just patches `nest_asyncio` on the calling
   thread. If you want true cross-thread usage, you have to build the
   dedicated-loop-thread + `run_coroutine_threadsafe` plumbing
   yourself.

3. **The user's "running it separately worked but in the trader it
   didn't" was the key insight.** Two days of theorizing collapsed
   the moment we accepted that the *trader's call path* (not the
   gateway, not the network, not the IBKR backend) was the variable.

### Other bugs found in the same investigation

- **`docker.sock` GID mismatch** — `group_add: ${DOCKER_GID:-999}` got a
  documented default in `.env.example`. Commits `4059291`, `0290959`.
- **`_reconnect_session` short-circuiting on stale `isConnected()`** —
  fixed independently. Commit `7f89161`.
- **`/resume` from Telegram didn't notify the running trader's
  RiskManager.** `evaluate_intraday` now reloads `halt.json` on every
  call. Commit `a914db2`.
- **Hardcoded `$` in bot/cycle status messages on a CHF-base account**
  — `AccountSnapshot` now carries `base_currency`, populated from
  IBKR's `NetLiquidation` row. Commit `b6f2b45`.

---

## 2026-06-10 — Three weeks of refused cycles: no-margin check was blind to per-currency cash

**Symptom.** Every scheduled and manual `/cycle` since late May ended with
`Cycle plan: refused by risk manager — no-margin breach — would overdraw:
USD -816k (limit -0)`. The watchdog escalated to "no successful cycle in
165h". Operator `/fx` conversions (600k CHF → USD) changed the reported
deficit by *zero dollars* — the tell that the check wasn't reading cash
at all.

**Root cause (three stacked bugs).**

1. `IbkrBroker.get_account()` never populated
   `AccountSnapshot.cash_by_currency` (only the separate `get_balances()`
   read CashBalance rows). The risk manager's `_check_no_margin` falls
   back to `{base_currency: cash}` when the dict is empty. On this
   CHF-base account that meant *USD cash = 0 forever*, so any USD basket
   breached at full notional. FX conversions could never unblock it —
   the check structurally couldn't see the USD side. Fix: `get_account`
   now folds `get_balances()` into the snapshot. Commit `a8e458d`.

2. `RunnerStore.save_snapshot()` dropped `base_currency` and
   `cash_by_currency` (columns didn't exist), so `/balances` never showed
   the per-currency split — the operator had no way to see the standoff.
   Fix: idempotent `ALTER TABLE` migration + round-trip. Same commit.

3. `runner.py` used `datetime.now(tz=timezone.utc)` in the snapshot-
   refresh heartbeat touch **without importing `timezone`** — a silent
   (debug-logged) NameError every 60s. heartbeat.json stayed stale, so
   `/health` and the watchdog reported the system dead even while the
   broker link was fine, burying the real signal in alert noise.
   Commit `c820dd2`.

**Lessons.**

1. **A deficit that doesn't move when you fund it is not a funding
   problem.** The June 6 and June 10 rejections differed only by market
   drift (-812,534 → -816,421) across a 605k CHF conversion. Constancy
   under intervention localizes the bug to the *measurement*, not the
   state.

2. **Fallbacks that silently change semantics are landmines.** The
   `{base_currency: cash}` fallback was reasonable for a USD account and
   catastrophic for a CHF one. If a fallback can invert a safety
   decision, log it loudly at decision time ("margin check using
   base-ccy fallback — per-currency cash unavailable").

3. **Every advisory channel that lies costs trust in the ones that
   don't.** The watchdog crying wolf (bug 3) made it easy to ignore the
   one alert that mattered (the risk-manager rejection).

**Operator runbook after redeploy.** `git pull && docker compose build
trader && docker compose up -d trader bot`, then `/balances` (now shows
the CHF/USD split), `/fx <amount> CHF to USD` to fund the USD leg fully,
`/cycle`, and confirm the basket submits.
