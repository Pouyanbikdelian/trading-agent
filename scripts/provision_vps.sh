#!/usr/bin/env bash
#
# One-shot VPS provisioning for the trading agent on Ubuntu 22.04 / 24.04.
#
# What it does (idempotent — safe to re-run):
#   1. apt update + install Docker, git, ufw, unattended-upgrades
#   2. Create a non-root 'trading' user with docker access
#   3. Clone the repo into /opt/trading-agent
#   4. Lock down SSH (key-only, no password, no root via ssh after we verify)
#   5. UFW firewall — SSH in, everything else out
#   6. Set up unattended security updates
#   7. Print the next-steps checklist
#
# This script DOES NOT:
#   - Put any secrets on disk (you fill in .env interactively after)
#   - Start the trading runner (you flip TRADING_ENV after paper testing)
#   - Touch your local Mac
#
# Run as root on the freshly-provisioned droplet:
#   curl -fsSL https://raw.githubusercontent.com/Pouyanbikdelian/trading-agent/main/scripts/provision_vps.sh | bash
# or, after cloning:
#   bash scripts/provision_vps.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours + helpers
# ---------------------------------------------------------------------------
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
RESET=$'\033[0m'

log()    { printf "%s[provision]%s %s\n" "${BLUE}" "${RESET}" "$*"; }
ok()     { printf "%s[ok]%s %s\n"        "${GREEN}" "${RESET}" "$*"; }
warn()   { printf "%s[warn]%s %s\n"      "${YELLOW}" "${RESET}" "$*"; }
fatal()  { printf "%s[fatal]%s %s\n"     "${RED}" "${RESET}" "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || fatal "must be run as root (try: sudo bash $0)"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "apt update + base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y -o Dpkg::Options::="--force-confold"
apt-get install -y \
    ca-certificates curl gnupg lsb-release \
    git ufw fail2ban \
    unattended-upgrades \
    htop tmux jq

# ---------------------------------------------------------------------------
# 2. Docker (official repo, not the Ubuntu-bundled docker.io which lags)
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    log "installing Docker"
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
         https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
         > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    ok "docker installed: $(docker --version)"
else
    ok "docker already installed: $(docker --version)"
fi

# ---------------------------------------------------------------------------
# 3. Trading user — runs the containers, never root
# ---------------------------------------------------------------------------
if ! id -u trading >/dev/null 2>&1; then
    log "creating user 'trading'"
    useradd -m -s /bin/bash trading
    usermod -aG docker trading
    # Copy authorized_keys from root so the same SSH key works
    mkdir -p /home/trading/.ssh
    if [[ -f /root/.ssh/authorized_keys ]]; then
        cp /root/.ssh/authorized_keys /home/trading/.ssh/authorized_keys
    fi
    chown -R trading:trading /home/trading/.ssh
    chmod 700 /home/trading/.ssh
    chmod 600 /home/trading/.ssh/authorized_keys
    ok "user 'trading' created"
else
    ok "user 'trading' already exists"
fi

# ---------------------------------------------------------------------------
# 4. Clone the repo
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/Pouyanbikdelian/trading-agent.git}"
REPO_DIR="/opt/trading-agent"

if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "cloning $REPO_URL → $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "repo already present; pulling latest"
    git -C "$REPO_DIR" pull --ff-only || warn "git pull failed (uncommitted local changes?)"
fi
chown -R trading:trading "$REPO_DIR"
ok "repo at $REPO_DIR"

# ---------------------------------------------------------------------------
# 5. .env scaffolding (NOT filled in — operator does that next)
# ---------------------------------------------------------------------------
if [[ ! -f "$REPO_DIR/.env" ]]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    chown trading:trading "$REPO_DIR/.env"
    chmod 600 "$REPO_DIR/.env"
    ok "wrote $REPO_DIR/.env from .env.example (fill in secrets next)"
else
    ok ".env already exists — leaving it alone"
fi

# ---------------------------------------------------------------------------
# 6. UFW firewall — SSH in, everything else blocked inbound
# ---------------------------------------------------------------------------
log "configuring ufw firewall"
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
yes | ufw enable
ok "ufw active — only SSH is open inbound"

# ---------------------------------------------------------------------------
# 7. fail2ban — SSH brute-force protection
# ---------------------------------------------------------------------------
systemctl enable --now fail2ban
ok "fail2ban running"

# ---------------------------------------------------------------------------
# 8. Unattended security upgrades
# ---------------------------------------------------------------------------
log "enabling unattended security upgrades"
dpkg-reconfigure -f noninteractive unattended-upgrades || true
# Make sure the security pattern is selected
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
ok "unattended-upgrades configured"

# ---------------------------------------------------------------------------
# 9. SSH hardening — disable password auth (keys only), keep root login for now
# ---------------------------------------------------------------------------
log "hardening sshd_config"
SSHCFG=/etc/ssh/sshd_config
sed -i.bak \
    -e 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' \
    -e 's/^#\?ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' \
    -e 's/^#\?KbdInteractiveAuthentication.*/KbdInteractiveAuthentication no/' \
    "$SSHCFG"
systemctl restart ssh || systemctl restart sshd || warn "sshd restart failed; check logs"
ok "sshd: password auth disabled (keys only)"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
cat <<'EOF'

================================================================
 Provisioning complete.

 Next steps (run as the 'trading' user — never root for app work):

   # switch user
   su - trading
   cd /opt/trading-agent

   # 1) edit .env. Required:
   #      IBKR_USERNAME=<your IBKR login>
   #      IBKR_PASSWORD=<your IBKR password>
   #      IBKR_TRADING_MODE=paper   (NEVER live until paper-tested for 30 days)
   #      TRADING_ENV=paper
   #      TELEGRAM_BOT_TOKEN=<from BotFather>
   #      TELEGRAM_CHAT_ID=<from @getmyid_bot>
   nano .env

   # 2) build images + start everything
   docker compose build
   docker compose up -d

   # 3) confirm everything is running
   docker compose ps
   docker compose logs -f trader     # follow runner output

   # 4) sanity-check from Telegram: send /status to your bot
   #    or run a smoke test:
   docker compose exec trader trading status

   # 5) hard kill from anywhere:
   #    - Telegram:   /halt
   #    - Console:    docker compose exec trader trading halt --reason "manual"
   #    - Manual:     echo '{"halted":true,"reason":"x"}' > state/halt.json

 Reminders:
   - 'trading live' is gated by .env. Don't flip until paper has run for ≥30 days.
   - IB Gateway needs 2FA the first time it connects — approve the push on your phone.
   - To go live later: edit .env (TRADING_ENV=live, ALLOW_LIVE_TRADING=true,
     IBKR_TRADING_MODE=live, IBKR_PORT=4001), then:
       docker compose down
       docker compose --profile live up -d
================================================================
EOF
