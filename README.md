# Trading Platform

A stock scanner and autonomous trading platform for Trading 212, built around
transparent heuristics, explicit data provenance and fail-closed safety.

> **Nothing in this application is investment advice.** No screening result
> asserts that an instrument is a good investment, and no strategy is claimed to
> be profitable. Results describe configured criteria and nothing more.

> **This system defaults to paper trading.** One toggle in Settings selects the
> venue: **paper** (the Trading 212 demo account) or **live**. Live can only be
> selected when the server itself permits it — `LIVE_TRADING_ENABLED` plus live
> credentials — a deployment-level gate a browser session cannot change.

---

## Build status

This is a phased build. **Phase 1 (Foundation) is complete and working.**
Later phases are specified but not yet implemented, and the code says so
explicitly rather than pretending otherwise — where a phase is missing, the
relevant factory raises `NotImplementedError` with a pointer, rather than
silently substituting something that looks like it works.

| Phase | Scope | Status |
|---|---|---|
| 1 | Monorepo, auth, Postgres/Redis, Trading 212 demo, instrument sync, yfinance ingestion, mapping, dashboard, audit logging | **Done** |
| 2 | Rotating scanner, heuristic scoring, candidates, trade proposals, approval workflow | **Done** (push-notification *delivery* deferred — VAPID keygen + preferences exist, service-worker send does not) |
| 3 | Internal paper broker, risk engine, position sizing, stops, reconciliation, EOD summary | **Done** — DB-backed internal paper broker; risk engine (sizing + caps + correlation + halts, fail-closed); approve→execute wiring; broker-side ATR stops with trailing, time and emergency exits; scheduled reconciliation (halts on divergence, clears when clean); and a persisted EOD account summary. Live execution is Phase 5 |
| 4 | S&P 15m mean reversion, gold/oil trend, pie strategy, correlation filter | **Done** — strategy framework (config / run / decision) with signals routed through the risk engine; S&P 500 15-minute mean reversion, gold/oil trend, and a target-allocation pie rebalancer; the intraday (15m) data pipeline and a real Twelve Data provider (keyed; offline via mock); and the correlation filter refined to per-position benchmark exposure. Strategies are seeded inactive and paper-only |
| 5 | Live adapter, venue toggle, autonomous live, risk halts | **Done** — a single Paper/Live toggle picks the venue for the whole product (paper = Trading 212 demo, live = real money), so there is exactly one balance in the UI. Live is gated by the server flag + credentials and re-checked by a depth-in-defence preflight; sizing is bounded by persistent `max_live_capital` / `max_daily_loss`, and a daily-loss breach halts and reverts to paper. Autonomous strategy filling stays behind its own server toggle (default off). No test places a real order — the suite blanks broker credentials and injects venues. Backtesting/optimisation (a former separate phase) was dropped by decision |

### What Phase 2 adds

The whole of product #1, the **scanner**, working on real market data:

- Pure, tested technical indicators (`app/indicators`) — trend, momentum, risk,
  liquidity, positioning, ATR, correlation. Reused by Phase 4 strategies.
- A configurable 100-point score (Trend 25 / Momentum 20 / Risk 20 / Liquidity
  20 / Positioning 15) with the §6 classification bands. **Missing optional data
  never lowers the core score** (acceptance criterion 7), and no output asserts
  an instrument is a good investment — both enforced by tests.
- Rotating universe selection, so a budgeted catalogue is covered over days.
- The candidate → proposal → approval workflow: volatility-sized proposals with
  ATR stops, a duplicate-proposal guard, expiry, and explicit authenticated
  approval. Verified against 275+ real yfinance candles per instrument.
- A `/scanner` page ranking candidates with a per-signal breakdown.

### What Phase 1 actually does today

With a Trading 212 demo key configured (or by explicitly requesting the mock
broker, which is how the tests run keyless):

1. Signs you in (server-side sessions, Argon2id, CSRF-protected).
2. Synchronises the Trading 212 instrument catalogue.
3. Resolves canonical instrument identity by ISIN → MIC+ticker → provider
   symbol → manual confirmation.
4. Maps an instrument to a market-data symbol, keeping **signal** and
   **execution** instruments separate.
5. Backfills daily candles once, then refreshes only a ~10-day overlapping tail.
6. Normalises GBX→GBP, rejects incoherent and unclosed candles, records data
   quality events.
7. Writes every meaningful action to a hash-chained, database-enforced
   append-only audit log.

---

## Quick start

### Prerequisites

- Docker + Docker Compose
- Node 20+ and pnpm (`corepack enable pnpm`)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (`brew install uv`)

### Run it

```bash
git clone <this repo> && cd Stock-Automate

cp .env.example .env
# Generate a real secret (the development default is refused in production):
#   openssl rand -base64 32   -> SECRETS_ENCRYPTION_KEY

docker compose up -d postgres redis

pnpm install
pnpm db:migrate      # apply schema (includes the audit immutability triggers)
pnpm db:seed         # exchanges, settings, and a dev user

pnpm api:dev         # http://localhost:8000  (API docs at /docs)
pnpm dev             # http://localhost:3000
```

> **Do not run `pnpm build` / `next build` while `pnpm dev` is running.** Both
> write to `apps/web/.next`. A production build overwrites the dev server's
> webpack chunks, and the still-running dev server then dies with
> `Cannot find module './<n>.js'` (a stale-cache error, not a code error). If it
> happens: stop the dev server, `rm -rf apps/web/.next`, and restart `pnpm dev`.
> To validate web changes *without* disturbing a running dev server, type-check
> only — `cd apps/web && pnpm typecheck` (`tsc --noEmit`) — which touches nothing
> on disk. Only run a real build when no dev server is up.

Sign in with the credentials the seed script prints
(`dev@example.com` / `development-password`). **Change them before exposing this
to a network.**

The broker catalogue syncs automatically every day; to force it now, press
**Re-sync now** on the dashboard. This needs `TRADING212_DEMO_API_KEY` set — see
[docs/trading212-setup.md](docs/trading212-setup.md). Without it you get a **503**
naming the variable, not mock data.

To work offline, ask for the mock broker explicitly:

```bash
curl -X POST 'localhost:8000/instruments/sync?broker=mock' ...
```

Or bring the whole stack up in Docker:

```bash
docker compose up -d      # postgres, redis, api (migrates on boot), worker, web
```

### Ports

Host ports deliberately avoid the defaults so this stack coexists with another
local Postgres/Redis:

| Service | Host port |
|---|---|
| web | 3000 |
| api | 8000 |
| postgres | **5433** |
| redis | **6380** |

---

## Commands

```bash
# Tests
pnpm api:test                    # unit + integration (needs postgres up)
pnpm --filter web test           # web unit tests
pnpm test:e2e                    # Playwright

# Quality
pnpm api:lint                    # ruff
pnpm api:format                  # ruff format
pnpm api:typecheck               # mypy (strict)
pnpm typecheck                   # tsc
pnpm format                      # prettier

# Database
pnpm db:migrate                  # alembic upgrade head
pnpm db:revision -- "message"    # autogenerate a migration
pnpm db:seed

# Worker
pnpm worker:dev                  # celery worker + beat
```

The integration tests require PostgreSQL, deliberately: the schema depends on
Postgres behaviour (audit triggers, `nextval`, advisory locks, `ON CONFLICT`,
JSONB) that SQLite does not have, so a suite passing on SQLite would be testing
a different system than the one that runs.

---

## Architecture

```
apps/
  api/      FastAPI + SQLAlchemy — domain logic lives here
  worker/   Celery jobs; imports the API's domain layer rather than duplicating it
  web/      Next.js dashboard (never holds a credential, never touches the DB)
packages/   Shared TS: ui, config, types, indicators, api-client
infrastructure/
  docker/   Dockerfiles
```

Integrations sit behind interfaces (`app.broker.Broker`,
`app.data.MarketDataProvider`). Provider-native shapes — Trading 212 JSON,
yfinance DataFrames — never escape their adapter.

Further reading:

- [docs/architecture.md](docs/architecture.md) — layering and the seams that matter
- [docs/data-providers.md](docs/data-providers.md) — free-tier budgets, GBX, candle rules
- [docs/trading212-setup.md](docs/trading212-setup.md) — getting demo/live keys safely
- [docs/risk-model.md](docs/risk-model.md) — sizing and controls (Phase 3 design)
- [docs/pwa.md](docs/pwa.md) — install and push notifications (Phase 2)

---

## Safety model

The properties this codebase treats as non-negotiable:

- **Paper by default.** `default_paper_broker_kind()` cannot return a live
  broker, whatever the configuration.
- **A venue is never substituted.** A missing or rejected credential raises;
  nothing falls back to mock data. The mock broker and provider remain
  available, but only when explicitly named — which is what the test suite and
  CI do. Believing you are on Trading 212 while running invented prices is
  worse than an outage.
- **One venue at a time, and live is server-gated.** A single Paper/Live toggle
  decides where every order goes and which balance the UI shows. Selecting live
  needs the server flag, live credentials and no active risk halt — reported
  together so the UI can show every blocker at once. The execution preflight
  re-checks them before an order is built, so the gate holds even if the toggle
  is stale.
- **Going back to paper is always allowed** and never blocked. Stopping must
  never be harder than starting.
- **Live sizing is bounded by persistent ceilings** — `max_live_capital` caps the
  equity all live sizing scales against, and breaching `max_daily_loss` halts
  trading and drops the venue back to paper.
- **Order submission is never retried.** A timeout raises
  `BrokerAmbiguousResponseError`: the order may exist, so it is reconciled, not
  resent.
- **Fail closed.** Stale data, an unreachable provider or an uncertain broker
  state block trading rather than proceeding on assumption.
- **Credentials never reach the browser.** They are encrypted at rest (Fernet),
  redacted from logs by a processor, and scrubbed from audit payloads.
- **Audit is append-only in the database.** UPDATE, DELETE and TRUNCATE are all
  rejected by triggers, and each row hashes its predecessor.

---

## Notable engineering decisions

A few choices that look odd until you know why:

**Money is `Decimal`, never `float`; JSON carries it as a string.** Binary
floating point cannot represent decimal cash exactly, and JSON numbers are IEEE
doubles — serialising a price as a number would reintroduce the drift the schema
exists to prevent.

**Price unit is tracked separately from currency.** The LSE quotes some
instruments in pence and others in pounds *on the same exchange* — verified
against the live API: `SGLN.L` returns 5758.84 GBp while `VUAG.L` returns 108.36
GBP. Inferring the unit from the `.L` suffix would size every VUAG order 100×
wrong. The reported currency wins; normalisation happens once, at the adapter
boundary.

**Daily candles keep their session date rather than being tz-converted.**
yfinance stamps the LSE daily bar for 16 July as `2026-07-16 00:00+01:00`;
converting that to UTC yields `2026-07-15 23:00Z` and silently shifts the whole
series back a day. A daily bar is a label for a session, not an instant.

**Enum columns use a `StrEnumType` decorator.** A `Mapped[PriceUnit]` backed by
plain `String` round-trips to a `str`. Since these are `StrEnum`, `==` still
passes — so the bug hides until code uses `is`, which then silently never
matches. Caught by an integration test.

**The audit writer inserts once, never updates.** The obvious design (insert,
flush for the sequence, update with the hash) is impossible here: the
immutability trigger rejects the update. It caught exactly that during
development, which is the argument for enforcing it in the database rather than
in application code.

---

## Licence

Private project. No warranty. Use at your own risk — this software can be
configured to place real orders with real money.
