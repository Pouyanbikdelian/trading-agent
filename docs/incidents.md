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
