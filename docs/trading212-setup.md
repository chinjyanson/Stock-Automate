# Trading 212 setup

> **Read this before generating a live key.** A live API key can place real
> orders. Nothing in this document should be followed on a live account until
> you have run the full workflow on demo.

## A demo key is required

`TRADING212_DEMO_API_KEY` must be set for any broker call to work. Without it
the API returns **503 `not_configured`** naming the variable. It does *not* fall
back to mock data: seeing filled orders and moving cash against invented prices,
while believing you are exercising Trading 212's real tickers and rate limits,
is worse than a clear failure.

The mock broker still exists for offline work, but you must ask for it by name —
`?broker=mock` on the API, or `BrokerKind.MOCK` in code. That is how the test
suite and CI run without credentials.

## Authentication: key *and* secret

Trading 212 authenticates with **HTTP Basic auth over two credentials**, not a
single token:

```
Authorization: Basic base64("<API_KEY>:<API_SECRET>")
```

Generating a key gives you both an **API key** and an **API secret**. The secret
is displayed **once**, at generation. A key on its own returns `401` — which is
the most common reason "my key doesn't work": only the key was saved.

If you have lost the secret, you cannot recover it. Regenerate the key to get a
fresh pair.

## Getting a demo credential

1. Open the Trading 212 app or web platform.
2. Switch to a **Practice / Demo** account. Confirm the UI says Practice before
   continuing.
3. **Settings → API (Beta) → Generate API key.**
4. Grant the minimum scopes you need. For Phase 1, read-only is sufficient:
   - Instrument metadata
   - Account data
   - Portfolio
   - Orders (read)

   Do **not** grant order-placement scope until Phase 3.
5. Copy **both values** into `.env` immediately — the secret is shown only once:

```bash
TRADING212_DEMO_API_KEY=your_demo_key_here
TRADING212_DEMO_API_SECRET=your_demo_secret_here
TRADING212_DEMO_BASE_URL=https://demo.trading212.com/api/v0
```

Then confirm the credential actually works before starting the app:

```bash
pnpm check:keys
```

This probes the account endpoint read-only and reports `PASS`/`FAIL` per
provider, naming the exact fix on failure. It never places an order and never
touches the live account unless you pass `--include-live`.

## Live keys

**Do not do this yet.** Live trading additionally requires the Phase 3 risk
engine, which does not exist. `/live/arm` will refuse with a 409 listing that as
a blocker. This section documents the eventual process.

Live and demo are **separate credentials, separate accounts, separate config**,
each a key + secret pair:

```bash
TRADING212_LIVE_API_KEY=your_live_key_here
TRADING212_LIVE_API_SECRET=your_live_secret_here
TRADING212_LIVE_BASE_URL=https://live.trading212.com/api/v0
LIVE_TRADING_ENABLED=false      # keep false until you mean it
```

They are separate classes in code (`Trading212DemoBroker`,
`Trading212LiveBroker`), not one class with a flag. `is_live` derives from the
class, so no configuration mistake can make a demo adapter place a real order or
the reverse.

### What live actually requires

Setting the key is **not** enough. All five must hold:

1. `LIVE_TRADING_ENABLED=true` on the server
2. Live credentials configured
3. A recent, clean broker reconciliation
4. No active risk halt
5. A re-authenticated user typing the exact phrase, with capital and loss
   ceilings, at `/live/arm`

Arming **expires** (default 60 minutes) and can be revoked instantly. Disarming
needs no password and no phrase — stopping must never be harder than starting.

## Rate limits

Trading 212 publishes tight per-endpoint limits; instrument metadata is roughly
one request per minute. We throttle client-side rather than discovering limits
via 429s, because **a 429 on an order submission is an ambiguous outcome** — the
order may or may not exist — and we would rather never provoke one.

```bash
TRADING212_MAX_REQUESTS_PER_MINUTE=30
TRADING212_TIMEOUT_SECONDS=20
```

This is also why instrument sync runs daily, not hourly: the catalogue changes
slowly and the limit is the scarce resource.

## What Trading 212 is and is not used for

**Used for:** instrument catalogue, account cash, positions, pending and
historical orders, placing/cancelling orders, reconciliation, and whether an
instrument is currently tradable.

**Never used for:** historical OHLCV candles or scanner market data. Its rate
limits make that infeasible, and its candles are not the series strategies are
validated against. Candles come from yfinance and Twelve Data — see
[data-providers.md](data-providers.md).

## Ticker vocabulary

Trading 212 tickers are their own namespace:

| Trading 212 | Exchange | yfinance | ISIN |
|---|---|---|---|
| `VUAGl_EQ` | VUAG (XLON) | `VUAG.L` | IE00BFMXXD54 |
| `AAPL_US_EQ` | AAPL (XNAS) | `AAPL` | US0378331005 |

The sync service strips the suffix to recover the exchange ticker and infers a
MIC, but **an unrecognised suffix yields no exchange rather than a plausible
guess**. Wrong identity is worse than absent identity.

Trading 212 does not report MICs, and yfinance does not expose ISINs for
non-US listings — so an LSE mapping usually resolves below the ISIN tier and is
flagged `requires_confirmation`. That is working as intended: confirm it in the
UI before trading.

## Security

- Keys go in `.env` (gitignored). Never commit them.
- Keys are sent in an `Authorization` header, never a query string — query
  strings end up in proxy logs.
- Stored credentials are Fernet-encrypted at rest.
- Logs redact anything key-shaped via a structlog processor.
- Audit events record the *fact* of a credential change, never the value.
- The browser never receives a key; only the API process holds one, transiently.
- Consider IP-restricting the key in Trading 212 if your deployment has a static
  egress address.

If you suspect a key is exposed: revoke it in Trading 212 first, then rotate
`.env`. Revoking upstream is what actually stops it.
