---
name: deploy-vps
description: Deploy trading-agent code changes to the live VPS — test on the Mac, commit/push, check out on the detached VPS, rebuild, recreate trader-live, and verify gateway 2FA, halt state and the new feature. Use for any "deploy", "ship it", "push to the VPS" request.
---

# Deploying to the VPS

The desk trades REAL MONEY. Every step below exists because skipping it once
cost something. Hand Yan paste-ready blocks; the sandbox cannot reach the VPS.

## 1. Mac — test, commit, push

Run from the repo, NOT from `~`, and NOT inside conda `(base)`: from `~`, `uv`
finds no project and falls back to Anaconda's Python 3.9 pytest (NumPy 1.x/2.x
crash + `No module named 'trading'`). That failure says nothing about the code.

```bash
conda deactivate
cd "$HOME/Downloads/Trading project/Trading agent"
uv run pytest -q -m "not slow and not live" && make lint
git add <files> && git commit -m "..." && git push
```

`make typecheck` is expected to fail (pre-existing mypy backlog; CI gates on
ruff + pytest only) — never chain it with `&&` before a commit.

## 2. VPS — check out the commit (NOT `git pull`)

The VPS repo at `/opt/trading-agent` is on a **detached HEAD**. `git pull`
fetches, then fails with "You are not currently on a branch" — and the build
that follows is 100% CACHED, silently redeploying the OLD code.

```bash
ssh trading-vps
cd /opt/trading-agent
git status --short            # only state-backup-*.tgz expected; anything else -> stop and look
git fetch && git checkout --detach origin/<branch>
git log --oneline -1          # must show the commit you just pushed
```

## 3. VPS — build and recreate the RIGHT runner

The live runner is `trader-live` (profile `live`). `trader` is the PAPER runner
and must never run on this box (it crash-loops on the live port 4001; remove
with `docker compose rm -sf trader`).

```bash
docker compose build trader bot dashboard        # all share trading-agent:latest
docker compose up -d --force-recreate trader-live bot dashboard
docker compose ps
```

Check the build output: `COPY src ./src` must actually run. If it says
CACHED, the checkout did not happen — go back to step 2.
Check `docker compose ps`: trader-live IMAGE must read `trading-agent:latest`,
not a bare sha. `docker compose restart` does NOT re-read `.env`.

## 4. Verify — gateway, broker, halt

Recreating trader-live can bounce `ib-gateway`, which then waits for an **IBKR
Mobile 2FA tap**. Nothing alerts on this; the desk once sat disconnected two days.

```bash
docker compose logs --tail 30 ib-gateway | grep -i "second factor\|login\|Closed"
docker compose logs --tail 20 trader-live | grep -i "bootstrap\|managed view"
docker compose exec -T trader-live cat /app/state/live/halt.json
```

- "Second Factor Authentication initiated" -> Yan approves on the phone now
  (prompt expires ~3 min; `docker compose restart ib-gateway` for a fresh one).
- Healthy: `managed view: [...] excluded, equity X -> Y`. `waiting (connect)`
  = gateway not logged in.
- Never infer halt state — read `halt.json`.
- Service is `ib-gateway`, container is `ibkr-gateway` (`docker compose logs
  ibkr-gateway` fails).

## 5. Timing and the feature itself

- Cycle cron is `CRON=0 19 * * FRI` UTC (two-week cadence); the PM decides
  ~45 min before. A deploy around 18:00–19:00 UTC on a Friday can miss the PM
  run — check logs or `/proposal`; `/cycle` reruns it (approval still required).
- Then exercise the change from Telegram (e.g. `/events now` for event_watch).
  A feature is deployed when it has been seen working, not when the build passed.

## Housekeeping

- Periodically: `docker builder prune -af --filter until=168h` (build cache
  grew to 13.8 GB once). Measure memory with `vmstat` si/so, not `free -m`.
- Dockerfile copies `src config docker pyproject.toml` only — never `scripts/`.
  Anything the runner needs at runtime must live under `src/`.
