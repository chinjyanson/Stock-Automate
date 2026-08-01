#!/usr/bin/env bash
#
# Prepare a GCP e2-micro Always Free VM to run the backend 24/7.
#
# Run once, on the VM, as a sudo-capable user:
#   curl -fsSL <raw-url>/bootstrap.sh | bash
# or: scp it over and `bash bootstrap.sh`.
#
# Sizing note, because "will it fit in 1 GB?" is the whole question here and it
# was answered by measurement, not estimate:
#
#   celery worker, 600-instrument refresh   peak RSS  160 MB
#   scanner, 800-instrument scan            peak RSS  148 MB
#   celery beat (idle scheduler)                     ~80 MB
#   postgres, tuned as below                        ~250 MB
#   redis (broker + budget counters)                 ~30 MB
#   Debian 12 minimal                               ~120 MB
#                                                   -------
#                                                   ~640 MB
#
# That leaves roughly a third of the box free, which is why the swap file below
# is insurance rather than load-bearing. Both heavy jobs chunk their work (50
# symbols per fetch, one instrument at a time when scoring), so peak memory does
# not grow with catalogue size — a 12,800-instrument sweep costs the same RSS as
# a 600-instrument one, just more wall time.

set -Eeuo pipefail

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL:\033[0m %s\n\n' "$*" >&2; exit 1; }

log "Checking this is the machine you think it is"

# Cloud Shell is the easy mistake: you open it to run `gcloud`, forget to SSH
# onward, and run this here instead. It looks like it is working — Debian,
# apt, docker — right up until nothing you installed is on the VM. Refuse
# rather than warn, because the failure is otherwise invisible for several
# steps.
if [[ -n "${DEVSHELL_PROJECT_ID:-}" ]] || [[ "$(hostname)" == cs-* ]]; then
  die "This is Cloud Shell, not the VM.

  Cloud Shell is a container: it has no swap, its /proc/meminfo reports the
  host's memory, and anything installed here vanishes. SSH to the instance
  first, then run this again:

      gcloud compute ssh stock-automate --zone=us-central1-a

  or use the SSH button beside the instance in the Console."
fi

if ! grep -qi debian /etc/os-release 2>/dev/null; then
  echo "Expected Debian (the GCP e2-micro default image). Continuing anyway." >&2
fi

INSTANCE=$(curl -s -m 2 -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || true)
[[ -n "$INSTANCE" ]] && echo "Instance: ${INSTANCE}"

TOTAL_MB=$(awk '/MemTotal/ {printf "%.0f", $2/1024}' /proc/meminfo)
echo "RAM: ${TOTAL_MB} MB"
# An e2-micro reports ~970 MB. Substantially more means a machine type that is
# not in the free tier — worth stopping over, because the difference between
# e2-micro and e2-standard-4 is roughly £0 and £75 a month.
if [[ "$TOTAL_MB" -gt 4000 ]]; then
  echo
  echo "  WARNING: ${TOTAL_MB} MB is far more than an e2-micro's ~970 MB." >&2
  echo "  Only e2-micro is Always Free. Check with:" >&2
  echo "    gcloud compute instances list --format='table(name,machineType.basename())'" >&2
  echo
  read -r -p "  Continue anyway? [y/N] " reply
  [[ "$reply" == [yY]* ]] || die "Stopped. Recreate the instance as e2-micro."
fi

log "Swap"
# On a 1 GB box the stack fits with headroom, but a swap file turns a
# transient spike from an OOM kill into a slow minute. An OOM kill here would
# most likely take Postgres, which is the one process whose death costs data.
#
# Not fatal if it fails: swap is insurance, and the measured footprint (~640 MB
# of 1 GB) does not depend on it. Some filesystems and virtualised environments
# refuse swap files outright.
if ! swapon --show 2>/dev/null | grep -q '/swapfile'; then
  if sudo fallocate -l 2G /swapfile 2>/dev/null \
    && sudo chmod 600 /swapfile \
    && sudo mkswap /swapfile >/dev/null \
    && sudo swapon /swapfile 2>/dev/null; then
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    # Prefer reclaiming cache over swapping a running process: this box is
    # latency-insensitive but should not thrash during a nightly sweep.
    echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swap.conf >/dev/null
    sudo sysctl -p /etc/sysctl.d/99-swap.conf >/dev/null
    echo "2 GB swap enabled"
  else
    sudo rm -f /swapfile
    echo "WARNING: could not enable swap; continuing without it." >&2
    echo "  The stack fits in 1 GB without swap, but has less margin." >&2
  fi
else
  echo "swap already configured"
fi

log "Docker"
if ! command -v docker >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "Docker installed. Log out and back in for group membership to apply."
else
  echo "Docker already present"
fi

log "Unattended security updates"
# This box holds broker credentials and runs unattended for months. Automatic
# security patching is the minimum; it is not a substitute for looking at it.
sudo apt-get install -y -qq unattended-upgrades
sudo dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null 2>&1 || true

log "Directories"
sudo mkdir -p /opt/stock-automate/infrastructure/backups
sudo chown -R "$USER":"$USER" /opt/stock-automate

log "Firewall"
# Nothing listens publicly. Every service binds to loopback and is reached over
# an SSH tunnel — see docs/deployment.md. That removes the need for TLS, a
# domain, and a public API surface, which is three fewer things to get wrong on
# a box that can place trades.
if command -v ufw >/dev/null; then
  sudo ufw --force reset >/dev/null
  sudo ufw default deny incoming >/dev/null
  sudo ufw default allow outgoing >/dev/null
  sudo ufw allow 22/tcp >/dev/null
  sudo ufw --force enable >/dev/null
  sudo ufw status verbose
else
  echo "ufw not installed; relying on the GCP firewall (SSH only)"
fi

log "Done"
cat <<'NEXT'
Next:
  1. Log out and back in   (so `docker` works without sudo)
  2. Copy .env.prod to     /opt/stock-automate/.env.prod   (chmod 600)
  3. Copy the compose files and restore a database dump
  4. docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml up -d

Reach the API from your laptop without exposing anything:
  ssh -L 8000:localhost:8000 <vm>
NEXT
