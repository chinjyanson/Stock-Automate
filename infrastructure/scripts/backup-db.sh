#!/usr/bin/env bash
#
# Encrypted Postgres backup to S3, with local retention.
#
# What is actually irreplaceable in this database:
#
#   * `audit_events` — append-only by database trigger, and the record of what
#     the system did with money. It cannot be reconstructed from anywhere.
#   * `broker_credentials` — encrypted at rest under SECRETS_ENCRYPTION_KEY.
#   * positions, orders, trade intents — reconcilable against the broker, but
#     only laboriously.
#
# Candles and scanner results are re-derivable from providers, so a restore that
# loses a day of them costs a backfill, not an accounting problem. The dump
# takes everything anyway; at ~1.3 GB compressed to a few hundred MB, being
# selective would buy nothing and add a way to be wrong.
#
# Two things this deliberately does NOT do:
#
#   1. It does not back up SECRETS_ENCRYPTION_KEY. A dump encrypted under a key
#      stored beside it is not encrypted. That key belongs in a password manager
#      or AWS Secrets Manager, and nowhere near these files.
#   2. It does not claim success on a partial dump — `set -o pipefail` plus an
#      explicit size check, because a truncated backup that reports success is
#      worse than no backup at all. You would find out during a restore.
#
# Usage:  backup-db.sh [--verify]
#   --verify  additionally restore the dump into a scratch database and count
#             its tables. Slower; run it at least weekly.
#
# Cron (daily 03:15 UTC, after the nightly scan has finished):
#   15 3 * * * /opt/stock-automate/infrastructure/scripts/backup-db.sh >> /var/log/db-backup.log 2>&1

set -Eeuo pipefail

BACKUP_DIR="${BACKUP_DIR:-/opt/stock-automate/infrastructure/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-14}"
S3_BUCKET="${S3_BUCKET:-}"
COMPOSE_FILE="${COMPOSE_FILE:-/opt/stock-automate/docker-compose.prod.yml}"
ENV_FILE="${ENV_FILE:-/opt/stock-automate/.env.prod}"
PG_SERVICE="${PG_SERVICE:-postgres}"

VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

log() { printf '%s  %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# Any failure past this point is loud. A backup script that fails quietly is
# indistinguishable from one that works, for exactly as long as it takes to need
# the backup.
trap 'die "backup failed at line $LINENO"' ERR

[[ -f "$ENV_FILE" ]] || die "env file not found: $ENV_FILE"

# Read one key out of the env file *without executing it*.
#
# Deliberately not `source`. A .env is a Docker Compose file, not a shell
# script, and Compose's parser is the more permissive of the two: it accepts
# `NAME=Stock Automate` (unquoted, with a space), which bash reads as an
# assignment followed by a command. This repo's own .env contains exactly that.
# Worse, sourcing would execute any `$(...)` in any value — and this file holds
# broker credentials, so it is the last thing that should be evaluated.
#
# Last assignment wins, which is how Compose resolves duplicates too.
read_env() {
  local key="$1" value
  value=$(sed -n "s/^[[:space:]]*${key}=//p" "$ENV_FILE" | tail -n 1)
  # Strip one layer of surrounding quotes, if present.
  value="${value%$'\r'}"
  [[ "$value" == \"*\" ]] && value="${value:1:-1}"
  [[ "$value" == \'*\' ]] && value="${value:1:-1}"
  printf '%s' "$value"
}

POSTGRES_USER="$(read_env POSTGRES_USER)"
POSTGRES_DB="$(read_env POSTGRES_DB)"
[[ -n "$POSTGRES_USER" ]] || die "POSTGRES_USER not found in $ENV_FILE"
[[ -n "$POSTGRES_DB" ]] || die "POSTGRES_DB not found in $ENV_FILE"

mkdir -p "$BACKUP_DIR"
STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
DUMP="$BACKUP_DIR/${POSTGRES_DB}-${STAMP}.dump"

log "dumping ${POSTGRES_DB}"
# Custom format (-Fc): compressed, and restorable selectively with pg_restore.
# A plain SQL dump would need decompressing in full to inspect one table.
docker compose -f "$COMPOSE_FILE" exec -T "$PG_SERVICE" \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc --no-owner --no-acl \
  > "$DUMP"

# A dump smaller than this is not a small database, it is a failed dump — an
# empty file is a perfectly valid zero-byte write.
SIZE=$(wc -c < "$DUMP")
[[ "$SIZE" -gt 1000000 ]] || die "dump is only ${SIZE} bytes — refusing to treat this as a backup"
log "wrote $(numfmt --to=iec "$SIZE" 2>/dev/null || echo "${SIZE}B") to $DUMP"

if [[ "$VERIFY" -eq 1 ]]; then
  # The only test of a backup is a restore. Everything else is a test of the
  # backup script.
  SCRATCH="verify_${STAMP}"
  log "verifying by restoring into ${SCRATCH}"
  docker compose -f "$COMPOSE_FILE" exec -T "$PG_SERVICE" \
    createdb -U "$POSTGRES_USER" "$SCRATCH"
  # shellcheck disable=SC2002
  cat "$DUMP" | docker compose -f "$COMPOSE_FILE" exec -T "$PG_SERVICE" \
    pg_restore -U "$POSTGRES_USER" -d "$SCRATCH" --no-owner --no-acl >/dev/null
  TABLES=$(docker compose -f "$COMPOSE_FILE" exec -T "$PG_SERVICE" \
    psql -U "$POSTGRES_USER" -d "$SCRATCH" -tAc \
    "select count(*) from information_schema.tables where table_schema='public'")
  docker compose -f "$COMPOSE_FILE" exec -T "$PG_SERVICE" \
    dropdb -U "$POSTGRES_USER" "$SCRATCH"
  [[ "$TABLES" -gt 20 ]] || die "restored database has only ${TABLES} tables"
  log "verified: ${TABLES} tables restored cleanly"
fi

if [[ -n "$S3_BUCKET" ]]; then
  # SSE-KMS, and a bucket that should have Object Lock or at minimum versioning
  # plus a deny-delete policy. A backup an attacker can delete is a backup that
  # protects against disk failure only.
  log "uploading to s3://${S3_BUCKET}/"
  aws s3 cp "$DUMP" "s3://${S3_BUCKET}/$(basename "$DUMP")" \
    --sse aws:kms --storage-class STANDARD_IA --only-show-errors
  log "uploaded"
else
  log "S3_BUCKET unset — local copy only. This survives a bad deploy, not a lost instance."
fi

DELETED=$(find "$BACKUP_DIR" -name "${POSTGRES_DB}-*.dump" -mtime "+${RETENTION_DAYS}" -print -delete | wc -l)
log "pruned ${DELETED// /} local dump(s) older than ${RETENTION_DAYS}d"
log "done"
