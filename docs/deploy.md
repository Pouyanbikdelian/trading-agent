# Deploy runbook — Hetzner / DigitalOcean

End-to-end deployment of the trading agent + IB Gateway on a fresh Linux
VPS. Default target: **Hetzner CX22** (2 vCPU, 4 GB, EU) or **DigitalOcean
basic 2 GB**. Either works; pick whatever you already have an account on.

> **One-time, do this first.** Skim [restore.md](./restore.md) too — when
> the server dies (and it will, eventually), that drill is the one you'll
> follow. This document is the *first* deploy; [restore.md](./restore.md)
> is every deploy after.

## Pre-flight checklist

- [ ] `.env` populated locally (see `.env.example`); know your IBKR
      username/password and that 2FA is configured for unattended login.
- [ ] Telegram bot token + chat ID, if you want alerts.
- [ ] A domain or static IP you can SSH into (avoid changing IPs).
- [ ] You've already paper-traded locally for ≥ 30 days. **Do not deploy a
      strategy that hasn't gone through that loop** — the hard rules in
      `CLAUDE.md` apply on the VPS too.

## 1. Provision the VPS

```bash
# Hetzner example
hcloud server create \
    --name trader \
    --type cx22 \
    --image debian-12 \
    --ssh-key your-key \
    --location nbg1
```

DigitalOcean equivalent:

```bash
doctl compute droplet create trader \
    --region fra1 \
    --size s-2vcpu-4gb \
    --image debian-12-x64 \
    --ssh-keys your-key-id
```

> **Pick a region close to your IBKR account's home server.** A round-trip
> across the Atlantic costs 100–150 ms per API call, which adds up for
> reconciliation. Frankfurt or Amsterdam if your account is on
> `cdc1.ibllc.com`; New York/Ashburn for US accounts.

## 2. Harden the host

SSH in as root, then:

```bash
adduser --disabled-password --gecos "" trader
usermod -aG sudo trader
mkdir -p /home/trader/.ssh
cp ~/.ssh/authorized_keys /home/trader/.ssh/
chown -R trader:trader /home/trader/.ssh
chmod 700 /home/trader/.ssh
chmod 600 /home/trader/.ssh/authorized_keys
```

Disable root SSH and password auth — `/etc/ssh/sshd_config`:

```
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

```bash
systemctl reload ssh
ufw allow OpenSSH
ufw --force enable
```

From now on, SSH as `trader`.

## 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker trader
newgrp docker
```

Confirm: `docker run --rm hello-world`.

## 4. Pull the repo

```bash
cd ~
git clone <your-git-url> trading-agent
cd trading-agent
```

## 5. Configure `.env`

```bash
cp .env.example .env
nano .env
```

Set at minimum:

| Variable                | Notes                                              |
|------------------------|----------------------------------------------------|
| `IBKR_USERNAME`        | Your IBKR account login.                           |
| `IBKR_PASSWORD`        | Bot-only password if your account supports it.     |
| `IBKR_TRADING_MODE`    | `paper` (default) or `live`.                       |
| `IBKR_HOST`            | `127.0.0.1` for the host-networked runner/gateway. |
| `IBKR_PORT`            | `4002` paper, `4001` live.                         |
| `TRADING_ENV`          | `research` / `paper` / `live`. Start at `paper`.   |
| `ALLOW_LIVE_TRADING`   | `false` until you've explicitly decided otherwise. |
| `TELEGRAM_BOT_TOKEN`   | Optional.                                          |
| `TELEGRAM_CHAT_ID`     | Optional.                                          |
| `TZ`                   | `UTC` is safest.                                   |

Confirm permissions:

```bash
chmod 600 .env
```

## 6. First boot — paper mode

```bash
docker compose --profile paper up -d --build
docker compose --profile paper logs -f trader
```

The first cycle won't fire until the next crontab tick — usually 16:00
weekdays. To validate the wiring right now without waiting:

```bash
docker compose --profile paper exec trader trading paper run us_large_cap --once
```

You should see a `Cycle @ ...` table with `status=ok` or `no_orders`
(neither is a failure — `no_orders` just means the strategy didn't want to
trade today). `status=error` is a real problem; check the logs.

## 7. Smoke-check the gateway

```bash
docker compose logs ib-gateway | tail -50
```

Look for `IBC: detected "Login failed"` or `…re-login required` — those
mean your credentials or 2FA didn't pass. The gateway image documents the
2FA setup; IBKR's mobile push is the path of least resistance.

## 8. Backfill data

```bash
docker compose --profile paper exec trader trading data fetch us_large_cap --from 2018-01-01
```

The Parquet cache lives in the `data` named volume. Backfill once;
subsequent cycles only fetch the missing tail.

## 9. Schedule, but in paper

Edit `docker-compose.yml` if you need a different cron (default: weekdays
16:00). Restart:

```bash
docker compose --profile paper restart trader
```

The healthcheck reads the heartbeat file every 60 s. A stale heartbeat
marks the runner unhealthy; `restart: unless-stopped` restarts an exited
container, **not** a container that merely becomes unhealthy. The autoheal
sidecar currently watches only the gateway. The bot still shares the
runner heartbeat, so its health status does not independently measure
Telegram responsiveness.

## 10. Telegram alerts

Confirm the runner sent its "started" message. If not:

```bash
docker compose --profile paper exec trader python -c \
  "from trading.runner.alerts import TelegramAlerts; \
   import os; \
   a = TelegramAlerts(token=os.environ['TELEGRAM_BOT_TOKEN'], \
                      chat_id=os.environ['TELEGRAM_CHAT_ID']); \
   a.info('hello from the trader VM')"
```

## 11. Promotion to live — **don't**, yet

Do not flip `ALLOW_LIVE_TRADING=true` until **all** of these are true:

- [ ] The strategy has paper-traded ≥ 30 calendar days on this VPS.
- [ ] You've reviewed the equity curve from
      `docker compose exec trader sqlite3 /app/state/runner.db
      'SELECT ts, equity FROM account_snapshots ORDER BY ts;'`.
- [ ] You've manually run [restore.md](./restore.md) at least once and
      know it works.
- [ ] You've sized `MAX_POSITION_PCT` and `MAX_GROSS_EXPOSURE` in
      `.env` to start small. Doubling later is easy; recovering
      from a sized-too-big day is not.

When that day comes:

```bash
# In .env:
TRADING_ENV=live
ALLOW_LIVE_TRADING=true
IBKR_TRADING_MODE=live
IBKR_PORT=4001
```

```bash
# Stop the paper profile before changing `.env` to live values.
docker compose --profile paper stop trader
docker compose --profile live up -d
```

The runner profiles are deliberately separate: normal paper startup uses
`--profile paper`; live startup uses `--profile live`. The `live` profile
uses `trading live run`, which double-checks the gate and refuses without
both flags.

## 12. Backups

Daily snapshot of the state volumes — they're tiny:

```bash
# /etc/cron.daily/trader-backup, run as root
docker run --rm \
    -v $(docker volume inspect trading-agent_state -f '{{.Mountpoint}}'):/state:ro \
    -v $(docker volume inspect trading-agent_logs  -f '{{.Mountpoint}}'):/logs:ro \
    -v /var/backups/trader:/backup \
    alpine:3 \
    sh -c 'tar czf /backup/state-$(date +\%F).tar.gz /state /logs'
```

Rotate to 14 days, sync to a different host (`rsync`, S3, B2). When the
VPS dies you want this archive somewhere else.

## 13. Day-2 operations

- `docker compose --profile paper logs -f trader` — follow the paper runner.
- `docker compose --profile paper exec trader trading status` — paper config sanity check.
- `docker compose --profile paper exec trader cat /app/state/heartbeat.json` — last cycle.
- `docker compose --profile paper exec trader sqlite3 /app/state/runner.db
   'SELECT * FROM cycles ORDER BY ts DESC LIMIT 10;'` — recent cycles.
- `docker compose --profile paper exec trader cat /app/state/halt.json` — current halt state.
- Halt manually: `docker compose exec bot trading halt --reason manual`.
  This uses the configured `STATE_DIR` and the locked, atomic halt writer.
  Do not edit `halt.json` directly while processes can write it.
- Resume is an explicit operator action through `/resume`; it preserves
  the loss references. A halt blocks added exposure but still permits
  verified risk-reducing paths. It is not a no-orders maintenance barrier.

## 14. Verify the code that is actually running

The September 2026 baseline incident persisted after the source checkout
had been fixed: the bot container still contained the old reset handler.
Neither `git rev-parse HEAD` nor seeing `trading-agent:latest` in `docker ps`
proves the service loaded the intended image. A Docker restart also keeps
the old image; services must be recreated to use a rebuilt image.

Run the stdlib-only checker on the Docker host, against the reviewed
checkout. It needs Python 3.10+ and read access to Docker; it does not
load `.env`, invoke Compose, import `trading`, or connect to the broker:

```bash
cd /opt/trading-agent
python3 scripts/verify_deployment.py
```

Defaults cover `trader-live`, `trader-bot`, and `trader-dashboard`. To
verify paper or a differently named deployment, provide the checkout
directory and the complete container list explicitly:

```bash
python3 scripts/verify_deployment.py /opt/trading-agent trader trader-bot trader-dashboard
```

The JSON report includes immutable container/image IDs, the current tag's
image ID, a checkout manifest fingerprint, and changed/missing/unexpected
paths. It verifies all files under `src/`, `config/`, and `docker/`, plus
`pyproject.toml`, excluding Python bytecode caches and `.DS_Store`. It
also checks that Python resolves the package to `/app/src/trading`, that
all selected services share an image, and that the container identity and
checkout remain stable during each check. Exit status is **0** for a
match, **1** for drift, **2** when verification is incomplete. Missing or
stopped services cannot pass. Save the JSON alongside the reviewed commit
ID in the deployment record.

Scope matters: `config/` is mounted from the host in production, so the
check proves the files currently visible to the container. It cannot prove
that a long-lived process reloaded an edited file. It does not verify
installed dependency versions, `uv.lock` (absent from the runtime image),
broker readiness, baseline correctness, or account reconciliation. Container
health is reported separately and does not make source drift acceptable.

## 15. Maintenance release with the live runner stopped

Prepare and test the reviewed change locally first. Do not change risk
limits, arming flags, baseline state, or the live state-environment stamp
as a side effect of deployment. Take a consistent backup of the state
databases and halt file before any subsequent state migration; retain
the prior image ID and deployment manifest for rollback.

For an operator-authorized maintenance window, record a halt, then stop
the runner and the command/scheduling processes. A halt alone leaves
guard exits and other verified reductions available. Stopping the runner
does not cancel orders already resting at the broker.

```bash
docker compose exec bot trading halt --reason 'maintenance: reviewed accounting repair'
docker compose --profile live stop trader-live bot scheduler
```

Build the shared image without starting any service. The build definition
lives on the paper `trader` service; selecting it here only builds code:

```bash
docker compose --profile paper build trader
```

Verify the candidate image in a temporary container with no network,
broker credentials, state mounts, or trading entrypoint. The checker can
read this isolated process while the live runner remains stopped:

```bash
deploy_probe=$(docker run -d --network none --no-healthcheck --entrypoint python trading-agent:latest -B -c 'import time; time.sleep(600)')
python3 scripts/verify_deployment.py /opt/trading-agent "$deploy_probe"
docker rm -f "$deploy_probe"
```

Continue only if verification returned 0. Prepare all consumers of the
shared image with **no start**, so the bot cannot be left on stale code:

```bash
docker compose --profile live up --no-start --no-deps --force-recreate trader-live bot dashboard
```

Leave the live runner stopped until the owner explicitly authorizes live
execution. [AGENTS.md](../AGENTS.md) requires approval each time Codex
runs live execution; starting an armed live runner can submit orders even
if a halt exists. After the approved start, immediately rerun the checker
against all three services and inspect independent broker/snapshot
freshness and reconciliation evidence. Re-enable the scheduler only as
part of that approved restart. Do not clear the persisted halt as part of
an image release or treat a source match as authorization to resume.

Repairing an already contaminated baseline is a separate, reviewed state
operation: use a fresh reconciled account snapshot, prove the managed
book and currency, and retain the previous reference and reset provenance.
A code deployment cannot reconstruct true historical drawdown from the
corrupted high-water mark.

## 16. Wave-1 release notes (2026-09-23)

`.env` changes on the VPS (the compose defaults changed too, but `.env`
sets these explicitly and wins):

```bash
CRON=5 17 * * FRI                 # Friday 17:05 New York, both DST seasons
SCHEDULE_TZ=America/New_York
HEALTHCHECK_PING_URL=https://hc-ping.com/<your-check-uuid>   # optional, recommended
MANUAL_ORDER_CONFIRM_PCT=0.05     # optional; defaults shown
MANUAL_ORDER_MAX_PCT=0.50
```

New behaviour an operator will notice:

- `/flatten`, `/resume` (while halted) and manual orders of 5%+ of equity
  reply with a Confirm button / `/confirm TOKEN` (5-minute expiry).
- A basket approved after the book changed is refused with a critical
  alert; `/cycle` re-plans.
- Weekday 16:30 New York: broker reconciliation message when the ledger
  changed or the set of unresolved rows changed.
- Hourly watchdog: missed cycle, desk-cannot-trade streak, no-trade streak.
- A submission on an account whose kind contradicts TRADING_ENV (paper
  ids start with `D`) is refused by the broker adapter.

Deploy with the maintenance sequence in §15; this release changes no risk
limit, arming flag or baseline.

## Troubleshooting

| Symptom                              | Likely cause                              | Action                                                          |
|--------------------------------------|-------------------------------------------|-----------------------------------------------------------------|
| `trader` healthcheck failing         | Heartbeat stale or `status=error`         | `docker compose logs trader` for the traceback                  |
| `ib-gateway` restarting every ~30s   | Login / 2FA failure                       | Check `ib-gateway` logs; confirm 2FA, sometimes restart from UI |
| `BrokerError: not connected`         | Gateway booted slower than the trader     | Add `depends_on.condition: service_healthy` (compose v3.9+)     |
| Orders rejected with `position` reason | Requested position breaches the cap    | Inspect sizing and approved limits; do not loosen limits as a deployment fix |
| `Cycle … status=error` after upgrade | Strategy params drift after a refactor    | Pin the strategy params dict in your runner config              |
