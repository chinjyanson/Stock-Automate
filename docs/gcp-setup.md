# GCP setup, step by step

Getting the backend onto a Google Cloud `e2-micro` Always Free VM, from nothing.

Why this machine at all: the scheduled jobs fire at 21:30 and 22:00 UTC. A
laptop that sleeps misses them, and misses them *silently* — there is no error,
the scan simply never happened. See [`deployment.md`](deployment.md) for the
sizing measurements behind the choice.

Roughly 40 minutes end to end, most of it waiting.

---

## 1. Project, billing and a budget — all in the browser

**Do this part in the Console, not the CLI.** Not a style preference: on a fresh
account the CLI cannot bootstrap itself. Enabling any API requires billing to be
linked, and every CLI route to inspect or link billing goes through an API that
is itself disabled until billing is linked. You get `PERMISSION_DENIED ...
PreconditionFailure subject: '110002'` on everything, in a loop with no exit.

The Console links billing through Google's own endpoints and does not need any
API enabled. It also *shows* you the state, where the CLI only lets you infer it
from errors that deliberately do not distinguish "does not exist" from "you
cannot see it".

1. [Sign in](https://console.cloud.google.com)
2. **[Create a project](https://console.cloud.google.com/projectcreate)** — name
   it `stock-automate`. Note the **project ID** shown beneath the name: it is
   *not* the name. IDs are globally unique, so it may be `stock-automate-483712`.
   Every command below wants the ID.
3. **[Create a billing account](https://console.cloud.google.com/billing)** if
   you do not have one. A card is required even for free-tier usage; the free
   quotas do not draw on it.
4. **[Link billing to the project](https://console.cloud.google.com/billing/linkedaccount)**
   — the step that unblocks everything else. Having a billing account is not the
   same as attaching it to this project.
5. **Billing → Budgets & alerts → Create budget**
   * Amount: **£1**
   * Alert thresholds: 50%, 90%, 100%
   * Tick *Email alerts to billing admins*

Before going further, confirm on
[the project's billing page](https://console.cloud.google.com/billing/linkedaccount)
that it names an account rather than offering to link one. Everything after this
fails obscurely if it does not.

£1 rather than £0 for the budget, because a budget of zero alerts on the first
penny of rounding. At £1 the alert means something has genuinely gone wrong.

> **The most likely thing to go wrong** is a disk or region setting, covered in
> step 3. The second is an external IP left reserved but unattached — reserved
> static IPs bill when idle. The instructions below use an *ephemeral* IP, which
> avoids it.

---

## 2. Open Cloud Shell

The terminal icon `>_` in the top-right of the Console.

This is a free browser VM with `gcloud` already installed and already
authenticated. Nothing to install locally, and no credentials on your laptop.

Point it at your project. Look the ID up rather than typing the name — they are
different, and `gcloud` fails with *"The required property [project] is not
currently set"* rather than telling you that:

```bash
gcloud projects list          # PROJECT_ID is the first column, not NAME
```

```bash
export PROJECT_ID=stock-automate-483712      # yours, from the list above
gcloud config set project "$PROJECT_ID"
gcloud config get-value project              # confirm it stuck
```

No project listed? You have not created one yet — the Console step above is
skippable from here:

```bash
gcloud projects create "stock-automate-$RANDOM" --name=stock-automate
```

Now enable Compute. This is the first command that requires step 1 to have been
completed properly — if billing is not linked it fails with `PERMISSION_DENIED`
and a `PreconditionFailure subject: '110002'`, which says nothing about billing.
Go back and finish step 1 rather than debugging permissions.

```bash
gcloud services enable compute.googleapis.com
```

Enabling the Compute API takes a minute or two on a new project.

> A new account also comes with **$300 of trial credit over 90 days**, which
> runs *alongside* Always Free rather than instead of it. During the trial some
> usage may show as drawing on credit; the `e2-micro` still costs nothing once
> the credit expires, provided the region and disk type in step 3 are right.

---

## 3. Create the VM

Use the CLI, not the Console's *Create Instance* form. Three settings decide
whether this is free, and two of them are easy to miss in the UI.

```bash
gcloud compute instances create stock-automate \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-type=pd-standard \
  --boot-disk-size=30GB \
  --metadata=enable-oslogin=TRUE
```

| flag | why it matters |
|---|---|
| `--zone=us-central1-a` | Only `us-west1`, `us-central1`, `us-east1` are free. Any European zone bills at the standard rate. |
| `--boot-disk-type=pd-standard` | Recent `gcloud` defaults to `pd-balanced`, **which is not free**. The single easiest way to pay for a "free" VM. |
| `--machine-type=e2-micro` | One per month, across all regions combined. A second one bills. |

> **The Console will quote you ~$6/month. Ignore it.**
>
> That is the list price. The instance-creation page has no knowledge of the
> Always Free tier, which is not a discounted SKU — it is a credit applied at
> billing time. You are charged the ~$6 and an offsetting *"Free tier discount"*
> lands on the same invoice line, netting to zero.
>
> The `2 vCPU / 1 GB` spec shown is simply what an `e2-micro` is: two shared
> vCPUs with a 0.25 baseline. That is the free machine, not an upsell.
>
> Verify from the bill rather than the estimate — Billing → Reports, grouped by
> SKU, after two or three days. Expect an `E2 instance core/ram` charge with a
> matching free-tier discount against it.
>
> On the $300 / 90-day trial, usage draws on the credit first, so the discount
> may not appear until the trial ends. That is not a sign you are outside the
> free tier.

Verify before moving on:

```bash
gcloud compute instances describe stock-automate --zone=us-central1-a \
  --format="value(machineType.basename(), disks[0].type.basename())"
# expect:  e2-micro   pd-standard
```

---

## 4. Bootstrap it

Easiest is the **SSH** button beside the instance in the Console — a browser
terminal, already authenticated, no local `gcloud` involved. Or:

```bash
gcloud compute ssh stock-automate --zone=us-central1-a
```

Then, on the VM:

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq git
# /opt is root-owned, so git cannot create a directory there as you.
sudo mkdir -p /opt/stock-automate
sudo chown -R "$USER":"$USER" /opt/stock-automate
git clone https://github.com/<you>/Stock-Automate.git /opt/stock-automate
cd /opt/stock-automate
bash infrastructure/gcp/bootstrap.sh
exit          # log out and back in so docker works without sudo
```

**Private repo?** Create a [fine-grained personal access
token](https://github.com/settings/tokens) with *Contents: read-only* scoped to
this repository, then:

```bash
git clone https://<token>@github.com/<you>/Stock-Automate.git /opt/stock-automate
```

A read-only token scoped to one repository, rather than an SSH key with wider
reach — this machine holds broker credentials, so what it can reach should be
the minimum that works.

The script installs Docker, adds 2 GB of swap, enables unattended security
updates, and closes the firewall to everything but SSH.

---

## 5. Secrets

Generate fresh values — **do not reuse your development ones**:

```bash
openssl rand -base64 32     # SECRETS_ENCRYPTION_KEY
openssl rand -base64 32     # POSTGRES_PASSWORD
```

Build `.env.prod` from `.env.example`, then copy it up from your laptop (not
Cloud Shell — the file should not pass through a machine you do not control):

```bash
gcloud compute scp .env.prod stock-automate:/opt/stock-automate/.env.prod \
  --zone=us-central1-a
gcloud compute ssh stock-automate --zone=us-central1-a \
  --command='chmod 600 /opt/stock-automate/.env.prod'
```

Set in that file:

```
ENVIRONMENT=production
LIVE_TRADING_ENABLED=false      # leave false until this has run quietly for weeks
SESSION_COOKIE_SECURE=false     # see the note in step 8
```

> `SECRETS_ENCRYPTION_KEY` encrypts your broker credentials at rest. Store it in
> a password manager as well. Lose it and every stored credential becomes
> unreadable; keep it *only* beside your database backups and they are not
> meaningfully encrypted.

---

## 6. Move the database

Do not re-backfill from yfinance — it would take days against rate limits, and
you already have the data.

```bash
# On your laptop
BACKUP_DIR=./dump COMPOSE_FILE=docker-compose.yml ENV_FILE=.env \
  ./infrastructure/scripts/backup-db.sh
gcloud compute scp ./dump/trading_platform-*.dump \
  stock-automate:/opt/stock-automate/infrastructure/backups/ --zone=us-central1-a
```

~1.3 GB compresses to ~160 MB. Uploading is *ingress* to GCP, which is free.

On the VM, start the datastores and restore into them:

```bash
cd /opt/stock-automate
docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod up -d postgres redis
sleep 20

docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod exec -T postgres \
  psql -U trading -d trading_platform -c 'select 1' >/dev/null && echo "postgres up"

cat infrastructure/backups/trading_platform-*.dump | \
  docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod exec -T postgres \
  pg_restore -U trading -d trading_platform --no-owner --no-acl
```

Restoring on a shared core takes several minutes. `pg_restore` warnings about
existing roles are expected and harmless.

---

## 7. Start everything

Build the two Python images first. They are built here rather than pulled — a
single box does not need a registry, and building on the VM produces x86
binaries natively (built on an Apple Silicon laptop they would be arm64, and
every container would die with `exec format error`).

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod build api worker
```

Expect **10–20 minutes** on a shared core, and near-silence for long stretches
while `uv` resolves dependencies. It is not stuck. This is a one-off; later
builds reuse the layer cache.

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod up -d

docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml ps
```

Expect `postgres`, `redis`, `api`, `worker`, `beat` running and `migrate`
exited 0. **`web` should not be there** — the overlay leaves it off this box.

Confirm there is exactly one scheduler, which is the property that matters most:

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml ps beat
# exactly one container. Two schedulers means every job fires twice,
# and some of those jobs place orders.
```

Watch memory settle — expect roughly 640 MB of the 1 GB:

```bash
docker stats --no-stream
free -h
```

---

## 8. Reach it from your laptop

Nothing listens publicly; the firewall allows only SSH. Tunnel instead:

```bash
gcloud compute ssh stock-automate --zone=us-central1-a -- -L 8000:localhost:8000
```

Leave that open, and in another terminal:

```bash
pnpm --filter web dev     # http://localhost:3000, already pointed at :8000
```

Your existing frontend works unchanged.

> This is why `SESSION_COOKIE_SECURE=false` is acceptable here: the session
> cookie never crosses a network. If you ever expose the API publicly that stops
> being true, and you need a domain and a certificate — §3 of
> [`deployment.md`](deployment.md).

---

## 9. Confirm the schedule actually fires

The point of the exercise. After the next 21:30 UTC:

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  logs --since 24h beat | head -20

docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod exec -T postgres psql -U trading -d trading_platform -c "
    select started_at::timestamp(0), instruments_considered
    from scanner_runs where is_ad_hoc = false
    order by started_at desc limit 3;"
```

One row per night, not two. Two means a second scheduler somewhere — most
likely the worker still running on your laptop. **Stop that one.**

---

## 10. Ongoing

```bash
# Nightly backups (on the VM)
sudo crontab -e
15 3 * * * /opt/stock-automate/infrastructure/scripts/backup-db.sh >> /var/log/db-backup.log 2>&1
15 4 * * 0 /opt/stock-automate/infrastructure/scripts/backup-db.sh --verify >> /var/log/db-backup.log 2>&1
```

Set `S3_BUCKET` (or use `gsutil` to a Cloud Storage bucket — 5 GB is free in the
same regions) so the backups leave the machine. A backup that only exists on the
instance it protects survives a bad deploy, not a lost instance.

Deploying a change:

```bash
ssh stock-automate 'cd /opt/stock-automate && git pull && \
  docker compose -f docker-compose.prod.yml -f docker-compose.gcp.yml \
  --env-file .env.prod up -d --build'
```

Celery does not hot-reload; `up -d` recreates the containers, which is what
picks the change up.

---

## Troubleshooting

`gcloud` errors are terse and rarely name the actual cause. The four you are
most likely to meet:

**`The required property [project] is not currently set`**
The project *name* is not the project *ID*. `gcloud projects list`, take the
first column, `gcloud config set project <ID>`. Also worth re-running after a
Cloud Shell session times out — the config persists, but a fresh shell in a
different project does not inherit it.

**`PERMISSION_DENIED: Permission denied to enable service [...]`** with
`PreconditionFailure`, `subject: '110002'`
Billing is not linked to this project, whatever the message says. A genuine
permissions failure does not carry a precondition violation, and `110002` is
that precondition.

**Fix it in the Console, not the CLI.** This error appears for *every* service,
including the billing API itself, so there is no CLI route out: checking or
linking billing needs an API that cannot be enabled until billing is linked.

→ [Link a billing account to the project](https://console.cloud.google.com/billing/linkedaccount)

Then retry `gcloud services enable compute.googleapis.com`.

If the Console shows billing already linked and this persists, it is genuinely
permissions and you need **Service Usage Admin** or Owner on the project —
unlikely if you created it yourself.

**`does not have permission to access projects instance [X] (or it may not exist)`**
Usually the second half. GCP returns permission-denied rather than not-found so
the error cannot be used to probe which project names exist, and
`gcloud config set project X` never validates `X` — so a typo or a project you
never created fails exactly like one you cannot see.

[Check what actually exists](https://console.cloud.google.com/cloud-resource-manager),
or `gcloud projects list`. If the list is empty, the project was never created.

**`Quota 'IN_USE_ADDRESSES' exceeded`** or an instance-count quota error
A fresh project sometimes ships with a zero or very low quota until the first
billing cycle. Console → IAM & Admin → Quotas, filter for the named quota, and
request an increase — for these defaults it is usually granted in minutes.

**`The zone ... does not have enough resources`**
Capacity, not your account. Try `us-central1-b` or `us-central1-c` — but stay
inside `us-west1` / `us-central1` / `us-east1`, or the VM stops being free.

---

## Cost check after a week

```bash
gcloud billing accounts list
```

Then Console → Billing → **Reports**, grouped by SKU.

Expect the `E2 instance core/ram` charge to be matched by a **free-tier
discount**, netting to £0.00. A gross charge on its own is fine and expected;
what matters is the offsetting line.

Anything genuinely unrecovered is almost certainly one of three things, in order
of likelihood:

1. **Boot disk type** — `pd-balanced` instead of `pd-standard`
2. **Region** — anything outside `us-west1` / `us-central1` / `us-east1`
3. **A second `e2-micro`** — the allowance is one per *billing account*, not per
   project, so an instance left running in another project consumes it

```bash
gcloud compute instances list --format="table(name, zone, machineType.basename())"
gcloud compute disks list --format="table(name, zone, type.basename(), sizeGb)"
```

On the $300 trial the discount is masked by credit consumption until the trial
ends — see the note in step 3.
