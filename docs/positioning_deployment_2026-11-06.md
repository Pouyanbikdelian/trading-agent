# Positioning deployment — November 6, 2026

Runtime code `464b997` is pushed on `codex/pm-exception-review` and deployed to
`trader-live`, `trader-bot` and `trader-dashboard` on DigitalOcean.

Validation: 2,404 offline tests passed (one excluded network/broker-marked test),
Ruff passed, and focused strict mypy passed for all three new modules. An isolated
image import/status check used no network or broker. Runtime verification passed
for all 170 source/config/docker/metadata files in all three application containers.
The final image reuses the newly built `308f21a` dependency layer, after checking
that Dockerfile, uv.lock and pyproject.toml are unchanged; all runtime source and
configuration are copied from the final commit and independently hash-verified.

The first read-only collection on the VPS at 14:07 UTC returned 12 COT category
observations across seven futures markets, report date October 27. That was the
normally expected latest report before Friday's 15:30 New York release. SPY and
QQQ gamma were unavailable because their underlying bars were stale before the
US open. Goldman coverage is zero until a checked public report is imported.
No missing observation was fabricated or converted into a bearish signal.

Application health checks were healthy following restart, but **broker readiness
was false**. IB Gateway had already exited at 23:31 UTC November 5, before this
work's rollout. It was not restarted, and no successful trading cycle is claimed.
Broker-dependent operation remains blocked until that service reconnects.
Follow-up inspection found the saved halt reason **"gateway stopped for manual
trading"**. This is evidence of an operator session choice, not permission to
restart Gateway automatically. Gateway can take over the operator's manual IBKR
session, so its stopped state was preserved. Reconnection is an operator decision.

The existing halt remained active with identical file contents, `.env` remained
byte-for-byte unchanged, basket approval remains required, and the selected
Friday 15:35/16:15/17:00 New York sequence is preserved. No live trade, manual
cycle, order approval, subscription purchase or executable positioning overlay
was initiated.

Private rollback artifacts on the host:
`/opt/trading-agent-releases/20261106-464b997/` contains the prior image/commit,
environment copy, consistent state archive/checksum, build logs, runtime
verification and `rollback.sh`. Rollback restores application code/images only;
it deliberately does not overwrite broker ledgers or approval history.

Behavior and remaining source/validation requirements:
[Positioning pipeline](positioning_pipeline.md).
