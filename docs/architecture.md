# Architecture

## Shape

Three deployables and one shared domain layer.

```
┌──────────┐   HTTPS + cookie    ┌──────────┐
│   web    │ ──────────────────► │   api    │
│ Next.js  │ ◄────────────────── │ FastAPI  │
└──────────┘   never a secret    └────┬─────┘
                                      │ shares app.*
                                 ┌────┴─────┐
                                 │  worker  │
                                 │  Celery  │
                                 └────┬─────┘
                                      │
                        ┌─────────────┴──────────────┐
                        │                            │
                  ┌─────▼─────┐              ┌───────▼──────┐
                  │ PostgreSQL│              │    Redis     │
                  │ truth     │              │ locks, rate  │
                  └───────────┘              └──────────────┘
```

**The worker imports the API's domain layer** (`app.services`, `app.broker`,
`app.data`) rather than reimplementing it. A scheduled sync and an
API-triggered sync must be the same code, or they will diverge — and the one
that diverges is the one nobody is watching.

**The web app holds nothing sensitive.** It has no database connection and no
broker credential. Its only privilege is a session cookie, which is HttpOnly, so
even an XSS bug cannot read it.

## Layers

```
routes/          thin: parse, authorise, delegate, serialise
  ↓
services/        orchestration; owns transactions and audit
  ↓
broker/  data/   integrations behind interfaces
  ↓
models/          SQLAlchemy; Decimal money, explicit enums
```

### The seams that matter

Two abstractions are load-bearing, and they exist for the same reason: the
things behind them are *interchangeable and untrustworthy*.

**`app.broker.Broker`** — Trading 212 demo, Trading 212 live, mock, and (Phase 3)
an internal simulator are peers. Nothing outside `app.broker` may import a
concrete adapter or see broker JSON. This is what makes the internal simulator a
first-class execution venue rather than a test double.

**`app.data.MarketDataProvider`** — yfinance, Twelve Data, EODHD and mock are
peers. yfinance DataFrames stop at the adapter; what crosses is
`app.data.types.Candle`, already unit-normalised.

If provider shapes leaked past these boundaries, every consumer would quietly
grow a dependency on one vendor's field names and the interface would be
decorative.

## Data flow

```
Trading 212 ──sync──► BrokerInstrument ──resolve──► Instrument (canonical)
                                                        │
yfinance ──────────► MarketDataMapping ◄────────────────┘
    │                       │
    └──ingest──► Candle ◄───┘   (local store = working source of truth)
                   │
                   ▼
            strategies read HERE, never a provider
```

Once data is stored, providers are upstream of the truth, not consulted at
decision time. A decision is always made against data that has been persisted,
quality-checked and versioned.

## Instrument identity

Three tables, deliberately not one:

| Table | Answers |
|---|---|
| `Instrument` | What security is this, canonically? |
| `BrokerInstrument` | What will Trading 212 accept in an order? |
| `MarketDataMapping` | What will this provider answer to? |

They disagree in practice. Trading 212 calls it `VUAGl_EQ`; yfinance calls it
`VUAG.L`; the exchange calls it `VUAG`. Collapsing them is how an order gets
placed against the wrong security.

Identity resolves ISIN → MIC+ticker → provider symbol → name+currency+exchange →
manual confirmation. Anything below the ISIN tier is marked
`requires_confirmation` and held at `MAPPING_REQUIRED`, which blocks progression
toward tradability.

### Signal vs execution

`MarketDataMapping.is_signal_source` marks the series a strategy *reads*; orders
still route to the instrument's broker ticker. That indirection is what lets a
strategy trade VUAG on 15-minute SPY signals. Exactly one mapping per instrument
may be the signal source — two would make "the" signal ambiguous.

## Transactions and audit

Services own transactions; routes commit. An audit record and the change it
describes land in the same transaction — an order that committed without its
audit row, or vice versa, is worse than either failing.

Audit appends serialise through a PostgreSQL advisory lock. That is a real cost,
accepted deliberately: two concurrent appends reading the same chain tip would
both claim the same predecessor and fork the chain, which is indistinguishable
from tampering. Audit volume is low; if it ever isn't, the answer is a dedicated
append worker, not a weaker chain.

## Idempotency

Jobs are retried; the broker's order endpoint is not idempotent. Two defences:

1. **Idempotent by construction** — sync and ingestion upsert, so re-running
   converges. This is the primary defence: a job safe to run twice needs no lock.
2. **Distributed locks** where duplication could reach a broker. Redis, fenced by
   a token so an overrun task cannot delete a lock someone else now holds, and
   always TTL'd so a dead worker cannot block a job forever.

## Failure posture

Fail closed, everywhere:

| Condition | Response |
|---|---|
| Provider unreachable | Mark unavailable; do not trade |
| Candle stale | Block signals for that instrument |
| Candle unclosed | Never served for decisions |
| Order submission times out | `BrokerAmbiguousResponseError` → reconcile, never retry |
| Broker state uncertain | Block further orders for that instrument |
| Redis unreachable (rate limit) | Deny the request |
| Live requested, no credential | Raise — never substitute a simulator |

## Testing

- **Unit** — pure logic, no I/O. Normalisation, adapters against fixture frames,
  broker safety rules.
- **Integration** — real PostgreSQL, per-test throwaway database. The schema
  depends on Postgres behaviour SQLite lacks; testing on SQLite would test a
  different system.
- **E2E** — Playwright, browser through the real API.

No automated test submits a real-money order, and none can: the mock and demo
brokers are the only ones reachable without a live key, and `resolve_broker`
raises rather than falling back.
