"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Account,
  type AuditEvent,
  type Health,
  type LiveStatus,
  type Position,
  type SyncResult,
  api,
  formatMoney,
} from "@/lib/api";
import { ModeBanner } from "@/components/ModeBanner";

export default function DashboardPage() {
  const router = useRouter();

  const [health, setHealth] = useState<Health | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null);

  const load = useCallback(async () => {
    // `finally` rather than a call at each exit: an early return or an
    // unforeseen throw must still clear the spinner. A dashboard stuck on
    // "Loading…" tells the user nothing and offers no way out.
    try {
      try {
        await api.me();
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.push("/login");
          return;
        }
        setError(
          err instanceof ApiUnreachableError
            ? err.message
            : "Could not reach the API. Is it running on port 8000?",
        );
        return;
      }

      // Each panel resolves independently: a broker outage must not blank the
      // whole dashboard, since the audit log and live status are exactly what a
      // user needs during an outage.
      const [healthR, accountR, positionsR, auditR, liveR] = await Promise.allSettled([
        api.health(),
        api.activeAccount(),
        api.activePositions(),
        api.audit(15),
        api.liveStatus(),
      ]);

      if (healthR.status === "fulfilled") setHealth(healthR.value);
      if (accountR.status === "fulfilled") setAccount(accountR.value);
      if (positionsR.status === "fulfilled") setPositions(positionsR.value);
      if (auditR.status === "fulfilled") setAudit(auditR.value.items);
      if (liveR.status === "fulfilled") setLiveStatus(liveR.value);

      // Surface the broker's own reason verbatim — "set TRADING212_DEMO_API_KEY"
      // and "Trading 212 rejected the key" need different actions from the user,
      // and a generic "unavailable" hides which one it is.
      if (accountR.status === "rejected") {
        const reason = accountR.reason;
        setError(
          reason instanceof ApiError || reason instanceof ApiUnreachableError
            ? reason.message
            : "Broker is unreachable. Account and positions are unavailable.",
        );
      }
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  async function onSync() {
    setSyncing(true);
    setSyncResult(null);
    try {
      const result = await api.syncInstruments();
      setSyncResult(result);
      await load();
    } catch (err) {
      setError(
        err instanceof ApiError || err instanceof ApiUnreachableError
          ? err.message
          : "Instrument sync failed",
      );
    } finally {
      setSyncing(false);
    }
  }

  if (loading) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-10">
        <p className="text-sm text-[var(--color-ink-muted)]">Loading…</p>
      </main>
    );
  }

  const currency = account?.currency ?? "GBP";

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-6 py-10">
      <header>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">
          {health ? `${health.environment} · v${health.version}` : "—"}
        </p>
      </header>

      <ModeBanner account={account} liveStatus={liveStatus} />

      {account?.is_stale && (
        <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)] px-4 py-2 text-sm text-[var(--color-ink-muted)]">
          Broker data is delayed (last updated {account.age_seconds}s ago). Trading 212 is
          rate-limiting; this refreshes automatically.
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm">
          {error}
        </div>
      )}

      <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Account value" value={formatMoney(account?.total, currency)} />
        <Stat label="Cash" value={formatMoney(account?.cash, currency)} />
        <Stat label="Invested" value={formatMoney(account?.invested, currency)} />
        <Stat label="Open positions" value={String(positions.length)} />
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <div className="flex items-center justify-between border-b border-[var(--color-border-subtle)] px-4 py-3">
          <h2 className="font-medium">Instruments</h2>
          <button
            onClick={onSync}
            disabled={syncing}
            title="The broker catalogue is re-synced automatically every day; this forces it now."
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
          >
            {syncing ? "Syncing…" : "Re-sync now"}
          </button>
        </div>
        <div className="px-4 py-3 text-sm">
          {syncResult ? (
            <p>
              Synced {syncResult.total_from_broker} instruments from{" "}
              {syncResult.broker.replace(/_/g, " ")} — {syncResult.instruments_created} new,{" "}
              {syncResult.broker_instruments_updated} updated
              {syncResult.instruments_needing_confirmation > 0 && (
                <>
                  {", "}
                  <span className="text-[var(--color-warn)]">
                    {syncResult.instruments_needing_confirmation} need identity confirmation
                  </span>
                </>
              )}
              .
            </p>
          ) : (
            <p className="text-[var(--color-ink-muted)]">
              The broker catalogue re-syncs automatically every day — you don&apos;t need to
              do this by hand. It keeps up with delistings, availability and trade-size
              changes; it never makes anything tradable on its own (instruments join the Bot
              Universe explicitly).
            </p>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
          Positions
        </h2>
        {positions.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
            No open positions.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-[var(--color-ink-muted)]">
                <tr className="border-b border-[var(--color-border-subtle)]">
                  <th className="px-4 py-2 font-medium">Instrument</th>
                  <th className="px-4 py-2 text-right font-medium">Quantity</th>
                  <th className="px-4 py-2 text-right font-medium">Avg price</th>
                  <th className="px-4 py-2 text-right font-medium">Current</th>
                  <th className="px-4 py-2 text-right font-medium">Unrealised</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.broker_ticker} className="border-b border-[var(--color-border-subtle)]">
                    <td className="px-4 py-2 font-mono text-xs">{p.broker_ticker}</td>
                    <td className="tabular px-4 py-2 text-right">{p.quantity}</td>
                    <td className="tabular px-4 py-2 text-right">{p.average_price}</td>
                    <td className="tabular px-4 py-2 text-right">{p.current_price ?? "—"}</td>
                    <td className="tabular px-4 py-2 text-right">{p.unrealised_pnl ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
          Audit log
        </h2>
        {audit.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">No events yet.</p>
        ) : (
          <ul className="divide-y divide-[var(--color-border-subtle)]">
            {audit.map((event) => (
              <li key={event.id} className="flex gap-3 px-4 py-2 text-sm">
                <span className="tabular w-10 shrink-0 text-[var(--color-ink-muted)]">
                  #{event.sequence}
                </span>
                <span className="w-40 shrink-0 text-[var(--color-ink-muted)]">
                  {new Date(event.occurred_at).toLocaleString()}
                </span>
                <span className="flex-1">{event.summary}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <footer className="pt-2 text-xs text-[var(--color-ink-muted)]">
        Screening results describe configured criteria only. Nothing here is investment
        advice, and no result implies an instrument is a good investment.
      </footer>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)] px-4 py-3">
      <p className="text-xs text-[var(--color-ink-muted)]">{label}</p>
      <p className="tabular mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}
