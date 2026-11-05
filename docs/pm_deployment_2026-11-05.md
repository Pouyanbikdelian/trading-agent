# PM improvements deployment — November 5, 2026

The owner authorized activation after choosing personal review of every
proposed basket as the deployment safeguard. The forward paper evaluation
previously recommended has not been completed; passing tests does not establish
better investment performance.

## Release and verification

- Deployed code: `644ad3f7c3c89eade3d7689fb859e7036b7bbfe5`.
- Previous code: `13c18bf`.
- New image: `trading-agent:644ad3f`, also tagged `trading-agent:latest`.
- Image ID: `sha256:2cef6914050c55938ca0b3b8935919d1886c1edf5f49b7e8e4b419196d228fe8`.
- Recreated only `trader-live`, `bot` and `dashboard`, with `--no-deps`.
  The gateway and its settings were not changed.
- `scripts/verify_deployment.py` passed for all three containers: runtime
  source/configuration hashes match the checkout and all share the new image.
- Built from a Git archive containing tracked files only. Isolated imports
  passed with networking disabled and no live credentials or mounted state.
- Local validation before deployment: 2,384 offline tests passed, lint passed
  and focused type checks passed.
- After startup, all three app containers reported healthy with zero restarts.
  The dashboard answered HTTP 401 to an unauthenticated local request.

`REQUIRE_CYCLE_APPROVAL=true` was verified both before and after recreation.
The `.env` file is unchanged, the live state path remains `/app/state/live`,
and the existing `halt.json` was preserved byte-for-byte. There was no pending
basket approval or active execution lock holder when stopping the services.
No manual cycle, order approval, flag arming or broker order was performed.
This statement concerns basket approval; existing position-guard behavior was
not redesigned by this release.

## Rollback

The server release directory is:
`/opt/trading-agent-releases/20261104-644ad3f/` (preparation began November 4).
It contains the previous commit/image identifiers, a private configuration
backup, a consistent state archive taken while app services were stopped,
its checksum, build log, before/after runtime verification and `rollback.sh`.

The previous image is retained as `trading-agent:rollback-13c18bf-20261104`.
The rollback script stops the three app services, checks out the previous
commit, restores the image tag, recreates the services and verifies runtime
files. Review current activity before invoking it.

Rollback deliberately keeps the current state/order ledger. Restoring an old
ledger can duplicate broker actions; the state archive is for deliberate
manual recovery only. A code rollback cannot undo trades or losses.

## Outstanding operational limitation

The runner and bot were already unhealthy before deployment, with a stale
heartbeat and `broker_ready_failed.flag`. The gateway was repeatedly restarting;
its post-deployment log showed second-factor authentication awaiting completion.
Consequently, runtime code identity is verified, but broker readiness and a
successful live cycle are not certified. App health checks recovered after
restart; the runner still reported broker bootstrap waiting on the API port.
The existing halt remains. No new
login approval was requested from the owner during this deployment.

New reasoning/audit records will be created by future PM runs. Matched shadow
outcomes need forward prices and complete metadata/FX coverage; they are not
available merely because the code has been deployed.
