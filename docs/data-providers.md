# Market data

Free tiers are the binding constraint on this system. The strategy is: fetch
once, store locally, refresh a small tail, and treat the database as the working
source of truth.

## Provider roles

| Provider | Role | Key | Status |
|---|---|---|---|
| **yfinance** | Broad daily candles, long history, LSE ETFs, dividends/splits | None | Implemented |
| **Twelve Data** | US 15-minute signal candles only (SPY + a few) | Free Basic | Phase 4 |
| **EODHD** | Verification and gap-fill only | Free | Phase 2 |
| **Mock** | Deterministic fixtures for tests and offline use | None | Implemented |

### Broad daily priority

```
1. Local database
2. yfinance batch retrieval
3. EODHD fallback / validation
4. Mark unavailable or stale
```

The chain running out marks data **unavailable**. It never substitutes something
unverified — that is the whole point of the ordering.

## Budgets

Configured in `.env`, overridable at runtime via `system_settings`. Never
hard-coded.

```
TWELVE_DATA_DAILY_MAX=800                  # the plan's limit
TWELVE_DATA_DAILY_OPERATIONAL_LIMIT=720    # where we stop non-critical work
TWELVE_DATA_DAILY_EMERGENCY_RESERVE=80     # kept for open positions
TWELVE_DATA_PER_MINUTE_OPERATIONAL_LIMIT=7 # of a max of 8

EODHD_DAILY_OPERATIONAL_LIMIT=18           # of ~20
EODHD_DAILY_EMERGENCY_RESERVE=2
```

### Why two stores

- **Daily budget → PostgreSQL**, incremented in the same transaction that
  authorises the request. A crash between "decided to call" and "called"
  over-counts by at most one, which is the safe direction. In Redis, an eviction
  would silently restore spend already made.
- **Per-minute limit → Redis**, 60s expiry. High-frequency, disposable, shared
  across API and worker.

Redis being unreachable **denies** the request. We cannot know the recent rate,
and guessing "probably fine" is how a free tier gets suspended.

### Spend priority

```
1. Open live positions          ─┐ may draw on the
2. Protective stops / exits     ─┘ emergency reserve
3. Core strategy signals
4. Approval revalidation
5. Scanner verification
6. Background backfill
```

This is the order in which work *stops* as the budget depletes. The reserve
exists so an open position can always have its exit checked — a budget that can
strand a position is worse than no budget.

## Incremental refresh

Full history is downloaded **once**. Afterwards only a ~10-bar overlapping tail
is re-requested and upserted.

The overlap is not redundancy: providers revise recent bars (late prints,
consolidated corrections, corporate-action adjustment). Without it we would
preserve whatever we read first, forever. Upserts overwrite on conflict for the
same reason — a refetch is usually a *correction*.

This is what makes rotating a large catalogue inside a free tier possible at all.

## GBX vs GBP — the expensive one

The London Stock Exchange quotes some instruments in **pence** and others in
**pounds**. Verified against the live API:

```
SGLN.L   currency=GBp   close=5758.84    ← pence
VUAG.L   currency=GBP   close=108.36     ← pounds
LLOY.L   currency=GBp   close=110.75     ← pence
```

So the `.L` suffix does **not** determine the unit. Inferring from the suffix
would divide VUAG by 100 and size every order 100× wrong — a failure that does
not raise, it just quietly trades the wrong amount.

Rules:

- The provider's reported currency wins (`GBp` is yfinance's pence marker).
- Normalisation happens **exactly once**, at the adapter boundary.
- `Instrument.currency` is what it *settles* in (GBP); `Instrument.price_unit` is
  what it is *quoted* in (GBX). These are different columns because they are
  different facts.
- Volume is a share count and is never scaled.

`normalise_price` is deliberately **not idempotent** — there is a test asserting
that. If someone later "makes it safe" to call twice, they have hidden a
double-conversion bug rather than fixed one.

## Candle rules

**Never trade on an unclosed candle.** A forming bar's `close` is just the last
trade so far; it will change. `is_closed` is set honestly by adapters and
`CandleStore.get_candles` filters unclosed bars out **by default**. Charting is
the only legitimate exception.

**Daily bars keep their session date.** yfinance stamps the LSE bar for 16 July
as `2026-07-16 00:00+01:00`. Tz-converting to UTC gives `2026-07-15 23:00Z` —
the bar moves to the previous day and the series misaligns against everything it
is compared to. A daily bar labels a session; it is not an instant. Intraday bars
*are* instants and are converted normally.

**NaN never reaches a Decimal column.** yfinance pads holidays and halted
sessions with NaN. `Decimal('nan')` is constructible and false-y in every
comparison, so a NaN close would make an instrument look permanently below every
threshold rather than raising. Rows without full OHLC are dropped; the gap
detector notices the hole.

**Incoherent bars are dropped.** `high < low`, or a close outside the range,
means the provider sent something uninterpretable. Trading on it is worse than
having no bar.

## Uniqueness

```
instrument_id + interval + timestamp + data_series_type
```

`data_series_type` (raw vs adjusted) is in the key because a raw and an adjusted
bar for the same minute are *different facts*, not a conflict to deduplicate.

## Quality states

| Status | Meaning | Tradable |
|---|---|---|
| `ok` | Verified | Yes |
| `stale` | Older than the freshness gate | No |
| `suspect` | Failed a sanity check | No |
| `conflicted` | Providers disagree | No |
| `incomplete` | Bar still forming | No |
| `unavailable` | Never fetched | No |

Only `ok` may originate a signal. Absent data counts as stale: "never fetched"
and "fetched but old" are both reasons not to trade.
