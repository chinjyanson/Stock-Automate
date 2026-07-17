# PWA and push notifications

> **Status: partially implemented.** The manifest and app shell exist. The
> service worker, push subscriptions and notification delivery are **Phase 2**,
> shipping alongside the scanner candidates and approvals they exist to deliver.
> Building the notification pipeline before there is anything to notify about
> would produce a system that can only send test messages.

## Installing

Once served over HTTPS (or `localhost`):

**iOS** — Safari → Share → Add to Home Screen. iOS requires the app be installed
to the Home Screen before it will accept web push at all.

**Android** — Chrome → menu → Install app.

**Desktop** — Chrome/Edge → install icon in the address bar.

## VAPID keys (Phase 2)

```bash
pnpm --filter web exec web-push generate-vapid-keys
```

```bash
VAPID_PUBLIC_KEY=...              # safe to expose
VAPID_PRIVATE_KEY=...             # never exposed; server-side only
VAPID_SUBJECT=mailto:you@example.com
NEXT_PUBLIC_VAPID_PUBLIC_KEY=...  # the public key again, for the browser
```

Only the public key carries the `NEXT_PUBLIC_` prefix. The private key must never
reach the browser — it authenticates *us* to the push service.

## Notification types (Phase 2+)

**Scanner** — new screening candidate; score materially improved; no longer
passes; approval requested; approval expiring; proposal rejected by risk.

**Trading** — signal generated; order submitted/filled/partially filled/
cancelled/rejected; stop placed/triggered; position closed; reconciliation
required.

**Risk and operations** — daily loss limit reached; drawdown kill switch;
strategy or instrument suspended; data stale; provider quota nearly exhausted;
broker unavailable; duplicate-order risk; live trading disabled.

**Scheduled** — market-open and market-close summaries, driven by the relevant
exchange's schedule rather than one hard-coded time.

## Approval notifications

```
Scanner candidate: [Instrument]
Score: [Score]
Proposed action: Buy/Sell
Proposed value: [Amount]
Risk: [Amount and percentage]
Reason: [Top signals]
Approval expires: [Time]
```

The action **deep-links to the approval page**. It does not execute anything.

A notification action is not an authenticated context — the tap arrives with no
proof of who is holding the phone. Executing a trade from it would mean anyone
with the lock screen could place an order. Approval requires an authenticated
session, always.

## Quiet hours

Users can configure categories and quiet hours, with one exception: **critical
risk notifications stay enabled while live trading is active.** Silencing a
drawdown kill switch is not a preference this application offers.

## Deduplication

Notifications are deduplicated and retried on failure. Delivery is recorded in
`NotificationEvent` — "did they actually get told" is a question that matters
after a loss, and it needs an answer that is not a guess.
