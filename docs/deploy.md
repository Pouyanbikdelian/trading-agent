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
| `IBKR_HOST`            | `ib-gateway` inside compose (docker DNS).          |
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
docker compose up -d --build
docker compose logs -f trader
```

The first cycle won't fire until the next crontab tick — usually 16:00
weekdays. To validate the wiring right now without waiting:

```bash
docker compose exec trader trading paper run us_large_cap --once
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
docker compose exec trader trading data fetch us_large_cap --from 2018-01-01
```

The Parquet cache lives in the `data` named volume. Backfill once;
subsequent cycles only fetch the missing tail.

## 9. Schedule, but in paper

Edit `docker-compose.yml` if you need a different cron (default: weekdays
16:00). Restart:

```bash
docker compose restart trader
```

The healthcheck reads the heartbeat file every 60 s; if no cycle runs for
5 min and the container reports unhealthy, Docker will restart it (with
`restart: unless-stopped`).

## 10. Telegram alerts

Confirm the runner sent its "started" message. If not:

```bash
docker compose exec trader python -c \
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
- [ ] You've sized `max_position_pct` and `max_gross_exposure` in
      `config/risk.yaml` to start small. Doubling later is easy; recovering
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
docker compose --profile live up -d
docker compose stop trader     # the paper service
```

The `live` profile uses `trading live run`, which double-checks the gate
and refuses without both flags.

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

- `docker compose logs -f trader` — follow the runner.
- `docker compose exec trader trading status` — config sanity check.
- `docker compose exec trader cat /app/state/heartbeat.json` — last cycle.
- `docker compose exec trader sqlite3 /app/state/runner.db
   'SELECT * FROM cycles ORDER BY ts DESC LIMIT 10;'` — recent cycles.
- `docker compose exec trader cat /app/state/halt.json` — current halt state.
- Halt manually: `docker compose exec trader python -c "from pathlib import
   Path; import json; p = Path('/app/state/halt.json'); s = json.loads(p.read_text());
   s['halted']=True; s['reason']='manual'; p.write_text(json.dumps(s))"`.
   The runner picks this up at the next cycle.
- Unhalt: same edit, set `halted=false`.

## Troubleshooting

| Symptom                              | Likely cause                              | Action                                                          |
|--------------------------------------|-------------------------------------------|-----------------------------------------------------------------|
| `trader` healthcheck failing         | Heartbeat stale or `status=error`         | `docker compose logs trader` for the traceback                  |
| `ib-gateway` restarting every ~30s   | Login / 2FA failure                       | Check `ib-gateway` logs; confirm 2FA, sometimes restart from UI |
| `BrokerError: not connected`         | Gateway booted slower than the trader     | Add `depends_on.condition: service_healthy` (compose v3.9+)     |
| Orders rejected with `position` reason | `max_position_pct` too tight            | Adjust `config/risk.yaml`, restart trader                       |
| `Cycle … status=error` after upgrade | Strategy params drift after a refactor    | Pin the strategy params dict in your runner config              |
