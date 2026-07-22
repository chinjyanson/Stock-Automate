"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Account,
  type Halt,
  type Health,
  type Instrument,
  type Order,
  type Position,
  type SyncResult,
  api,
  formatMoney,
} from "@/lib/api";
import { Stat } from "@/components/Stat";
import { PositionsTable } from "@/components/PositionsTable";
import { useToast } from "@/components/Toast";

/**
 * The single account view: monitor (account, positions, protective stops) and
 * operate (re-sync, emergency controls) in one place. Formerly split across the
 * Dashboard and Portfolio pages. The paper/live indicator lives in the nav bar;
 * the audit log and EOD summaries live under Settings.
 */
export default function DashboardPage() {
  const router = useRouter();
  const toast = useToast();

  const [health, setHealth] = useState<Health | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [halts, setHalts] = useState<Halt[]>([]);
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    // `finally` clears the spinner on every exit: an early return or an
    // unforeseen throw must still leave the user with a usable page.
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
      // whole page — the halts and emergency controls are exactly what a user
      // needs during one.
      const [healthR, accountR, positionsR, ordersR, haltsR, instrumentsR] =
        await Promise.allSettled([
          api.health(),
          api.activeAccount(),
          api.activePositions(),
          api.activeOrders(false),
          api.halts(true),
          api.instruments({ limit: 200 }),
        ]);

      if (healthR.status === "fulfilled") setHealth(healthR.value);
      if (accountR.status === "fulfilled") setAccount(accountR.value);
      if (positionsR.status === "fulfilled") setPositions(positionsR.value);
      if (ordersR.status === "fulfilled") setOrders(ordersR.value);
      if (haltsR.status === "fulfilled") setHalts(haltsR.value);
      if (instrumentsR.status === "fulfilled") setInstruments(instrumentsR.value.items);

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

  // Paper positions/orders key on the canonical instrument id, so map it back
  // to a human name where the instrument is loaded.
  const nameFor = useMemo(() => {
    const byId = new Map(instruments.map((i) => [i.id, i.name]));
    return (id: string) => byId.get(id) ?? id;
  }, [instruments]);

  async function onSync() {
    setSyncing(true);
    setSyncResult(null);
    try {
      const result = await api.syncInstruments();
      setSyncResult(result);
      toast.success(
        `Synced ${result.total_from_broker} instruments — ${result.instruments_created} new, ` +
          `${result.broker_instruments_updated} updated.`,
      );
      await load();
    } catch (err) {
      toast.error(
        err instanceof ApiError || err instanceof ApiUnreachableError
          ? err.message
          : "Instrument sync failed",
      );
    } finally {
      setSyncing(false);
    }
  }

  async function onKillSwitch() {
    const reason = window.prompt(
      "Activate the kill switch? This halts all trading until cleared. Reason:",
      "manual kill switch",
    );
    if (reason == null) return;
    try {
      await api.killSwitch(reason);
      toast.success("Kill switch activated — all trading is halted until cleared.");
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not activate the kill switch");
    }
  }

  async function onClearHalt(id: string) {
    try {
      await api.clearHalt(id);
      toast.success("Halt cleared.");
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not clear the halt");
    }
  }

  async function onFlatten() {
    const reason = window.prompt("Flatten every open position at market? Reason:", "manual flatten");
    if (reason == null) return;
    setBusy(true);
    try {
      const result = await api.flatten(reason);
      toast.success(`Flattened ${result.positions_closed} position(s) at market.`);
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not flatten positions");
    } finally {
      setBusy(false);
    }
  }

  async function onRunStops() {
    setBusy(true);
    try {
      const r = await api.runStops();
      toast.success(
        `Stop monitor ran — ${r.triggered} triggered, ${r.closed} closed, ${r.trailed} trailed.`,
      );
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not run stop management");
    } finally {
      setBusy(false);
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
  const stops = orders.filter((o) => o.order_type === "stop");

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-6 py-10">
      <header>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">
          {health ? `${health.environment} · v${health.version}` : "—"} · positions, protective
          stops and account controls. Switch venue and set risk limits in Settings.
        </p>
      </header>

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

      {halts.length > 0 && (
        <section className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)]">
          <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium text-[var(--color-warn)]">
            Active halts — trading is blocked
          </h2>
          <ul className="divide-y divide-[var(--color-border-subtle)]">
            {halts.map((h) => (
              <li key={h.id} className="flex items-center gap-3 px-4 py-2 text-sm">
                <span className="font-mono text-xs">{h.kind}</span>
                <span className="text-[var(--color-ink-muted)]">({h.scope})</span>
                <span className="flex-1">{h.reason}</span>
                <button
                  onClick={() => onClearHalt(h.id)}
                  className="rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-xs"
                >
                  Clear
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <Stat label="Account value" value={formatMoney(account?.total, currency)} />
        <Stat label="Cash" value={formatMoney(account?.cash, currency)} />
        <Stat label="Invested" value={formatMoney(account?.invested, currency)} />
        <Stat label="P/L" value={formatMoney(account?.result, currency)} />
        <Stat label="Open positions" value={String(positions.length)} />
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
          Positions
        </h2>
        <PositionsTable
          positions={positions}
          nameFor={nameFor}
          emptyMessage="No open positions. Approve a scanner proposal to open one."
        />
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
          Working orders — protective stops
        </h2>
        {stops.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
            No resting stops. Each filled entry places one broker-side.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-[var(--color-ink-muted)]">
                <tr className="border-b border-[var(--color-border-subtle)]">
                  <th className="px-4 py-2 font-medium">Instrument</th>
                  <th className="px-4 py-2 font-medium">Side</th>
                  <th className="px-4 py-2 text-right font-medium">Quantity</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {stops.map((o) => (
                  <tr
                    key={o.broker_order_id}
                    className="border-b border-[var(--color-border-subtle)]"
                  >
                    <td className="px-4 py-2 font-medium">{nameFor(o.broker_ticker)}</td>
                    <td className="px-4 py-2 uppercase">{o.side}</td>
                    <td className="tabular px-4 py-2 text-right">{o.quantity}</td>
                    <td className="px-4 py-2 text-[var(--color-ink-muted)]">{o.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
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
              The broker catalogue re-syncs automatically every day — you don&apos;t need to do
              this by hand. It keeps up with delistings, availability and trade-size changes; it
              never makes anything tradable on its own (instruments join the Bot Universe
              explicitly).
            </p>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-[var(--color-warn)] px-4 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-medium">Emergency controls</h2>
            <p className="text-sm text-[var(--color-ink-muted)]">
              Run the stop monitor now, halt all trading, or flatten every open position at
              market. A halt stays active until explicitly cleared.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onRunStops}
              disabled={busy}
              className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
            >
              Run stops
            </button>
            <button
              onClick={onKillSwitch}
              className="rounded-md border border-[var(--color-warn)] px-3 py-1.5 text-sm font-medium text-[var(--color-warn)]"
            >
              Kill switch
            </button>
            <button
              onClick={onFlatten}
              disabled={busy}
              className="rounded-md bg-[var(--color-warn)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              Flatten all
            </button>
          </div>
        </div>
      </section>

      <footer className="pt-2 text-xs text-[var(--color-ink-muted)]">
        Paper positions are simulated fills against stored daily candles. Nothing here is
        investment advice, and no result implies an instrument is a good investment.
      </footer>
    </main>
  );
}
