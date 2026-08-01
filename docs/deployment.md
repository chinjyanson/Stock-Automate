# Deployment

How this system gets onto the internet, and the decisions behind the shape it
takes. Written to be followed from nothing: no domain, no AWS account.

---

## 1. What is being deployed

Six processes, and they do not scale alike.

| service | scaling | notes |
|---|---|---|
| `api` | N replicas | stateless; behind the proxy |
| `worker` | N replicas | Celery worker; idle except during the nightly sweep |
| `beat` | **exactly 1, always** | the scheduler |
| `web` | N replicas | Next.js; stateless |
| `postgres` | 1 | the system of record |
| `redis` | 1 | Celery broker + provider budget counters |

### The beat constraint

Everything else here is ordinary web deployment. This is the part that is
specific to this system, and it is the one that costs money if you get it wrong.

Celery beat is the scheduler: it decides that the market-regime measurement runs
at 21:50, that the universe syncs at 22:10, that strategies evaluate on a
schedule. **Two beat processes means every scheduled task fires twice.** Some of
those tasks place orders.

This is why `docker-compose.prod.yml` splits `beat` out of `worker` — the local
development image runs `celery worker --beat`, which is convenient and would be
a duplicate-order bug the first time you scaled the worker.

Anywhere you deploy, the rule is the same: **one beat, and deployments must not
briefly run two.** On ECS that means `desiredCount: 1` with
`maximumPercent: 100, minimumHealthyPercent: 0` — stop the old task before
starting the new one. On a single host, Compose does this naturally.

---

## 2. Environments

Two, on separate machines:

* **dev** — `LIVE_TRADING_ENABLED=false`, Trading 212 **demo** credentials only.
  Break it freely.
* **prod** — live-capable, but see §7: it is deployed with live trading *off*.

Separate machines rather than two stacks on one box. The point of dev is that
you can destroy it, and that is not true of anything sharing a kernel, a disk,
or a Docker daemon with prod.

### What is promoted, and what is rebuilt

`api` and `worker` images are **built once and promoted** dev → prod. These hold
the trading logic; what you validated is byte-for-byte what runs.

`web` is **built per environment**, because `NEXT_PUBLIC_API_URL` is compiled
into the browser bundle — a Next.js constraint, not a choice. That asymmetry is
tolerable only because the web container holds no trading logic.

---

## 3. A domain (whatever you host on)

You need one before TLS, and TLS before production — `SESSION_COOKIE_SECURE`
must be `true` in production, which this app's own config validator enforces at
startup. A cookie marked `Secure` is never sent over plain HTTP, so without a
certificate you cannot log in at all.

Buy from anywhere (Cloudflare and Porkbun sell at cost; Route 53 is fine and
one less account). Then plan records:

```
app.example.com      → prod web
api.example.com      → prod API
app.dev.example.com  → dev web
api.dev.example.com  → dev API
```

**Keep both halves on one registrable domain.** The session cookie is set
`SameSite=Lax` with no `domain` attribute. Browsers treat `app.example.com` and
`api.example.com` as *same-site*, so the cookie is sent. Split them across
registrable domains — frontend on `*.vercel.app`, API on `example.com` — and the
browser silently drops the session cookie on every request. Login appears to
succeed and every subsequent call is a 401.

Set `CORS_ALLOW_ORIGINS` to the exact web origin. It is not a wildcard, and it
cannot be: the API is called with `credentials: "include"`.

---

## 4. AWS setup

### 4.1 Account and identity

1. Create the account. Enable MFA on the root user, then never use it again.
2. Create an admin IAM user for yourself, with MFA.
3. Create an **OIDC identity provider** for GitHub Actions:
   `token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
4. Create a deploy role trusting that provider, scoped to this repository:

   ```json
   {
     "Effect": "Allow",
     "Principal": { "Federated": "arn:aws:iam::<acct>:oidc-provider/token.actions.githubusercontent.com" },
     "Action": "sts:AssumeRoleWithWebIdentity",
     "Condition": {
       "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
       "StringLike": { "token.actions.githubusercontent.com:sub": "repo:<you>/Stock-Automate:*" }
     }
   }
   ```

   Grant it ECR push only. Nothing about a deployment needs to read a database.

   **No long-lived AWS access keys, anywhere.** This repository can reach a
   broker account; a leaked static key is the worst available outcome, and OIDC
   means there is no key to leak.

### 4.2 A container registry

Three repositories: `stock-automate-api`, `stock-automate-worker`,
`stock-automate-web`. On ECR, enable scan-on-push and a lifecycle policy that
expires untagged images after ~14 days.

> **Not hosting on AWS?** Use GitHub Container Registry instead — it is free for
> this, needs no separate account, and authenticates with the `GITHUB_TOKEN` the
> workflow already has. Point `release.yml` at `ghcr.io/<you>` and drop the two
> AWS steps. An Oracle or home host can pull from either; ECR only earns its
> place when the thing pulling is also AWS and can use an IAM role instead of a
> stored credential.

### 4.3 GitHub repository configuration

Variables (`Settings → Secrets and variables → Actions`):

| kind | name | example |
|---|---|---|
| variable | `AWS_REGION` | `eu-west-2` |
| variable | `BUILD_PLATFORM` | `linux/amd64` |
| variable | `DEV_API_URL` | `https://api.dev.example.com` |
| variable | `PROD_API_URL` | `https://api.example.com` |
| secret | `AWS_DEPLOY_ROLE_ARN` | `arn:aws:iam::<acct>:role/gh-deploy` |

> **`BUILD_PLATFORM` is not cosmetic.** An Apple Silicon machine builds
> `linux/arm64` by default; ECS Fargate defaults to `X86_64`. The mismatch fails
> at container start with `exec format error`, which reads like a corrupt image
> rather than a platform problem. Use `linux/amd64` for x86 Fargate or x86 EC2,
> `linux/arm64` for Graviton (`t4g`, `c7g`) or Fargate ARM64 — and never build
> deployable images from your laptop.

---

## 5. Secrets

Generate **per environment**. Never reuse a dev value in prod.

```bash
openssl rand -base64 32   # SECRETS_ENCRYPTION_KEY
openssl rand -base64 32   # POSTGRES_PASSWORD
```

### `SECRETS_ENCRYPTION_KEY` deserves its own paragraph

It encrypts broker credentials at rest.

* **Lose it** and every stored credential becomes undecryptable. You re-enter
  them; recoverable, but only because you can.
* **Leak it** *together with a database dump* and the credentials are readable.

So: store it in AWS Secrets Manager or a password manager, and **never in the
same place as a database backup**. A backup encrypted under a key sitting beside
it is not encrypted.

The app refuses to start in production if this is left at its placeholder — see
`_validate_production_secrets` in `apps/api/app/config.py`. That is deliberate:
booting with a key published in this repository would be worse than not booting.

### Getting secrets onto the host

Store the file in AWS Secrets Manager and fetch at deploy time:

```bash
aws secretsmanager get-secret-value --secret-id stock-automate/prod/env \
  --query SecretString --output text > /opt/stock-automate/.env.prod
chmod 600 /opt/stock-automate/.env.prod
```

Never `scp` a `.env` from your laptop, and never commit one. CI already fails
the build if `.env` becomes tracked.

> **Do not `source` a `.env` in a shell script.** Compose's parser accepts
> `NAME=Stock Automate` (unquoted, spaces) — this repo's own `.env` contains
> exactly that — and bash reads it as an assignment followed by a command. Any
> `$(...)` in a value would execute. `infrastructure/scripts/backup-db.sh`
> extracts the keys it needs textually for this reason.

---

## 6. Topology

All options below run the same images, the same `docker-compose.prod.yml`, and
the same backup script. The choice is where the container runs, not what runs.

### The filter

Celery beat must run continuously — the scheduled jobs at 21:50 and 22:10 are
the entire reason this is deployed rather than run from a laptop. **Any host
that sleeps on inactivity is disqualified**, and disqualified quietly: scanning
stops, exits stop being evaluated, and nothing reports an error.

That rules out Render's free tier (services sleep, no background workers, free
Postgres expires at 90 days), Neon and Supabase free Postgres (0.5 GB against a
1.3 GB database), and GCP's always-free `e2-micro` (1 GB RAM will not hold
Postgres, Redis and four containers).

### Free options

| | notes |
|---|---|
| **Oracle Cloud Always Free** | The only free tier that runs the whole stack. **Halved on 15 June 2026** from 4 OCPU/24 GB to 2 OCPU/12 GB, with no announcement. 12 GB is still ample. ARM capacity is often exhausted in popular regions. Set `BUILD_PLATFORM=linux/arm64`. |
| **A machine at home + Cloudflare Tunnel** | Free indefinitely. A spare laptop or Raspberry Pi 5 is sufficient. The tunnel supplies HTTPS and a hostname with no port forwarding and no public IP. Bounded by your home power and internet. |
| **AWS** | *Not* free in the old sense. Accounts created after 15 July 2025 receive **$200 of credits over 6 months**, and EC2/RDS usage draws down that balance rather than sitting beside it. Option A below consumes it in ~7 months; Option B in ~2.5. |

**Free is the right choice while `LIVE_TRADING_ENABLED=false`.** You are proving
the deployment is boring (§7), and that does not need a contract.

**Move to a paid host before enabling live trading.** Not for capability — free
tiers are technically adequate — but because an unannounced terms change or an
out-of-capacity reclaim stops beat, and positions then sit unmanaged with stops
that never fire. Oracle demonstrated exactly that in June 2026. Roughly €4/month
at any boring provider buys a contractual relationship, which is the thing
actually being purchased.

### Recommended free setup: GCP e2-micro Always Free

Runs the backend 24/7 at no cost. Measured, not assumed — both heavy jobs were
profiled before this was chosen:

| | peak RSS |
|---|---|
| celery worker, 600-instrument refresh | **160 MB** |
| scanner, 800-instrument scan | **148 MB** |
| beat (idle scheduler) | ~80 MB |
| postgres (tuned, see the overlay) | ~250 MB |
| redis | ~30 MB |
| Debian 12 | ~120 MB |
| **total** | **~640 MB of 1 GB** |

Peak memory does not grow with catalogue size: the refresh chunks at 50 symbols
per fetch and the scanner scores one instrument at a time, so a 12,800-name
sweep costs the same RSS as a 600-name one — only more wall time.

Three ways to be silently billed on a "free" VM:

* **Region must be `us-west1`, `us-central1` or `us-east1`.** Anywhere else,
  including all of Europe, is charged at the standard rate.
* **Disk must be `pd-standard`.** Recent `gcloud` defaults to `pd-balanced`,
  which is not free. This is the easiest one to get wrong.
* **1 GB/month egress from North America.** Market data is *ingress* (free);
  only your own browsing counts against it.

**→ Step-by-step walkthrough: [`gcp-setup.md`](gcp-setup.md)** — from creating
the account to confirming the schedule fires, including the budget alert to set
*before* creating anything.

The short version:

```bash
gcloud compute instances create stock-automate \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-type=pd-standard \
  --boot-disk-size=30GB \
  --metadata=enable-oslogin=TRUE

# then, on the VM
bash infrastructure/gcp/bootstrap.sh          # docker, 2 GB swap, firewall
# copy .env.prod (chmod 600) and a database dump across, then:
docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml up -d
```

The overlay tunes Postgres for 1 GB, drops the worker to one process (these
jobs wait on the network, not the CPU), leaves the frontend off the box, and
sets per-container memory limits so a leak degrades one service rather than
letting the kernel pick a victim — and its usual choice is Postgres, the one
process here whose death costs data.

**Nothing is exposed publicly.** Every service binds to loopback; the firewall
allows SSH only. Reach the API from your laptop with a tunnel:

```bash
ssh -L 8000:localhost:8000 stock-automate
pnpm --filter web dev        # locally; it already talks to localhost:8000
```

That is why this setup needs no domain and no TLS. It also means
`SESSION_COOKIE_SECURE` can stay false — the session cookie never crosses a
network. Expose the API publicly and that stops being true, at which point you
need §3 and a certificate.

**Shared core:** an e2-micro bursts but is not fast. A refresh measured at 2.6
minutes on a laptop should be expected to take 10–20 there. Irrelevant for a job
that runs at 21:30 unattended.

### Paid options

Two, both AWS, both using the same images and the same Compose file.

### Option A — one EC2 instance per environment (~$27/mo each)

`t4g.medium` (4 GB, Graviton) running `docker-compose.prod.yml`, with Caddy in
front for automatic TLS. Postgres and Redis run as containers on the instance.

* Cheapest, fewest moving parts, and the beat guarantee is trivially true.
* You own Postgres backups (`infrastructure/scripts/backup-db.sh` + cron).
* Set `BUILD_PLATFORM=linux/arm64`.

Sizing is comfortable: the database is ~1.3 GB today and grows ~1.2 GB/year,
dominated by candles.

### Option B — ECS Fargate + RDS + ElastiCache (~$80/mo each)

| | monthly |
|---|---|
| ALB | ~$17 |
| RDS `db.t4g.micro`, 20 GB gp3 | ~$13 |
| ElastiCache `cache.t4g.micro` | ~$11 |
| Fargate (~1.25 vCPU / 2.5 GB, always on) | ~$35 |
| ECR, Secrets Manager, transfer | ~$5 |

* Managed backups and patching; no host to maintain.
* `beat` **must** be its own service with `desiredCount: 1` and
  `minimumHealthyPercent: 0`.
* Migrations run as a one-off ECS task, not in the API's start command — two API
  tasks both running `alembic upgrade head` race each other.

**Recommendation: Option A for both environments.** You have one user and a
nightly batch job. Fargate's value is elastic scaling and no host maintenance,
and you need neither; it costs ~3× for capacity that stays idle. Option B is
the upgrade path if this ever serves more than you.

---

## 7. First production deploy

Deliberately staged. The system can move real money, so the deployment is proven
boring before it is allowed to.

1. **Deploy dev.** Demo credentials, `LIVE_TRADING_ENABLED=false`.
2. **Watch a full nightly cycle.** The scan, the universe sync, the regime
   measurement, the digest email. Confirm beat fired each job **once** —
   `celery -A worker.app inspect scheduled`, and the audit trail.
3. **Deploy prod with `LIVE_TRADING_ENABLED=false`.** Paper only.
4. **Run it for two weeks.** You are testing the deployment, not the strategy.
5. **Take and restore a backup** (§8). Before real money, not after.
6. **Only then** set `LIVE_TRADING_ENABLED=true` — and note that this alone does
   not trade. Live mode still requires the in-app arming step. Two switches, on
   purpose.

---

## 8. Backups

```bash
sudo crontab -e
15 3 * * * /opt/stock-automate/infrastructure/scripts/backup-db.sh >> /var/log/db-backup.log 2>&1
15 4 * * 0 /opt/stock-automate/infrastructure/scripts/backup-db.sh --verify >> /var/log/db-backup.log 2>&1
```

Daily at 03:15 UTC, after the nightly scan. Weekly with `--verify`, which
restores the dump into a scratch database and counts tables — **the only test of
a backup is a restore**; everything else tests the backup script.

Set `S3_BUCKET`. Enable versioning and Object Lock: a backup that a compromised
host can delete protects you against disk failure and nothing else.

What is irreplaceable: `audit_events` (append-only by database trigger, the
record of what the system did with money) and `broker_credentials`. Candles and
scanner results are re-derivable from providers.

**Restore drill** — do this once before trusting any of it:

```bash
docker compose -f docker-compose.prod.yml exec -T postgres \
  createdb -U trading restore_test
cat backup.dump | docker compose -f docker-compose.prod.yml exec -T postgres \
  pg_restore -U trading -d restore_test --no-owner --no-acl
```

---

## 9. Operational notes

**Migrations** run as the `migrate` service, to completion, before anything
else starts. Keep them backward-compatible for one release: during a rolling
deploy, old and new code briefly share a schema.

**Redis persistence** is on (`--appendonly yes`) and must stay on. It holds the
per-minute provider budget counters; losing them silently restores spend already
made, and the free-tier budgets are what keep the data pipeline inside its
limits.

**Beat's schedule file** lives on a named volume. In the container filesystem a
restart resets it and every periodic job re-fires as though it had never run.

**Log rotation** is configured in the Compose file (10 MB × 5 per service). The
default json-file driver has no limit, and a chatty worker fills the disk out
from under Postgres — which presents as data loss, not as a logging problem.

**Postgres and Redis publish no host ports** in `docker-compose.prod.yml`, and
`api`/`web` bind to `127.0.0.1` so only the reverse proxy reaches them. The
development Compose file publishes 5433/6380 for convenience; that is exactly
why production has its own file rather than an override.
