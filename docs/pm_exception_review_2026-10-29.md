# PM exception workflow review — 2026-10-29

## Result

The retained-simulation exception dead end is fixed. Review also reproduced
and fixed an approval-identity defect: approval for one thesis could authorize
a newer, unapproved thesis for the same ticker. The bridge now requires the
exact exception IDs selected by the PM decision, with active approval and
the reviewed size limit. Missing, rejected and expired reviews remain blocked.

No exception was approved and no live trading cycle was requested during this
work. Simulated ownership never supplies live execution authorization.

## Validation

- Full offline suite: **2,341 passed, 1 deselected**, with two existing Python
  multiprocessing/fork deprecation warnings. Network/broker tests were excluded.
- Repository lint and format checks: passed across 375 files.
- Strict type check: **310 errors in 65 files**. Running the same check against
  the pre-change commit `4d11406` produced the same diagnostics; this work added
  none. These are unresolved type-check failures, not proof of 310 runtime bugs.
- Regression coverage includes retained targets falling off the ladder, missing
  paperwork, retry deduplication, rejection, expiry, exact proposal approval,
  budget limits, malformed model numbers and actionable Telegram messages.

## Remaining findings and limits

**P2 — Shared rotating log file can lose file-log entries.**
`src/trading/core/logging.py` gives every service the same daily log filename,
while Compose mounts a shared logs volume. Each process configures its own
rotation/compression handler. The October 27 bot log already records a
`FileNotFoundError` while compressing `trading.2026-10-23.log`, consistent with
another process rotating that shared file. `enqueue=True` does not coordinate
independently configured handlers across containers. Use service/process-specific
file sinks or one central logging writer. This pre-existing logging defect was
not changed by the PM workflow repair; stderr logs remain separately available.

**Static typing is not clean.** The 310 existing diagnostics include incompatible
return types, possibly-null values, missing generic parameters and stale ignores.
They need a separate, incremental cleanup with tests rather than suppression.

This was a focused code review of exception persistence, PM selection, signal
eligibility and Telegram/cycle messaging, plus repository-wide offline checks.
It does not establish that the whole trading system is bug-free, nor does it
verify a broker fill or authorize any trade. IBKR login remediation was excluded
at the operator's request.
