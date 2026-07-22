"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Account,
  type DailySummary,
  type Halt,
  type Instrument,
  type Order,
  type Position,
  api,
  formatMoney,
} from "@/lib/api";

export default function PortfolioPage() {
  const router = useRouter();

  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [halts, setHalts] = useState<Halt[]>([]);
  const [summaries, setSummaries] = useState<DailySummary[]>([]);
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      try {
        await api.me();
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.push("/login");
          return;
        }
      }
      // Each panel resolves independently: one panel failing must not blank
      // the others.
      const [accountR, positionsR, ordersR, haltsR, summariesR, instrumentsR] =
        await Promise.allSettled([
          api.activeAccount(),
          api.activePositions(),
          api.activeOrders(false),
          api.halts(true),
          api.activeSummaries(14),
          api.instruments({ limit: 200 }),
        ]);

      if (accountR.status === "fulfilled") setAccount(accountR.value);
      if (positionsR.status === "fulfilled") setPositions(positionsR.value);
      if (ordersR.status === "fulfilled") setOrders(ordersR.value);
      if (haltsR.status === "fulfilled") setHalts(haltsR.value);
      if (summariesR.status === "fulfilled") setSummaries(summariesR.value);
      if (instrumentsR.status === "fulfilled") setInstruments(instrumentsR.value.items);

      if (accountR.status === "rejected") {
        const reason = accountR.reason;
        setError(
          reason instanceof ApiError || reason instanceof ApiUnreachableError
            ? reason.message
            : "Could not load the portfolio.",
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

  async function onKillSwitch() {
    const reason = window.prompt(
      "Activate the kill switch? This halts all trading until cleared. Reason:",
      "manual kill switch",
    );
    if (reason == null) return;
    try {
      await api.killSwitch(reason);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not activate the kill switch");
    }
  }

  async function onClearHalt(id: string) {
    try {
      await api.clearHalt(id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not clear the halt");
    }
  }

  async function onFlatten() {
    const reason = window.prompt(
      "Flatten every open position at market? Reason:",
      "manual flatten",
    );
    if (reason == null) return;
    setBusy(true);
    try {
      await api.flatten(reason);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not flatten positions");
    } finally {
      setBusy(false);
    }
  }

  async function onRunStops() {
    setBusy(true);
    try {
      await api.runStops();
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not run stop management");
    } finally {
      setBusy(false);
    }
  }

  async function onGenerateSummary() {
    setBusy(true);
    try {
      await api.runEodSummary();
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not generate the summary");
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
        <h1 className="text-2xl font-semibold">Portfolio</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">
          <span
            className="font-medium"
            style={{
              color: account?.is_live ? "var(--color-live)" : "var(--color-paper)",
            }}
          >
            {account?.is_live ? "LIVE — real money" : "Paper — Trading 212 demo"}
          </span>{" "}
          · positions, protective stops and recent activity. Switch venue and set risk
          limits in Settings.
        </p>
      </header>

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

      <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Account value" value={formatMoney(account?.total, currency)} />
        <Stat label="Cash" value={formatMoney(account?.cash, currency)} />
        <Stat label="Invested" value={formatMoney(account?.invested, currency)} />
        <Stat label="P/L" value={formatMoney(account?.result, currency)} />
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
          Positions
        </h2>
        {positions.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
            No open positions. Approve a scanner proposal to open one.
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
                  <tr
                    key={p.broker_ticker}
                    className="border-b border-[var(--color-border-subtle)]"
                  >
                    <td className="px-4 py-2 font-medium">{nameFor(p.broker_ticker)}</td>
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
          <h2 className="font-medium">End-of-day summaries</h2>
          <button
            onClick={onGenerateSummary}
            disabled={busy}
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
          >
            Generate today
          </button>
        </div>
        {summaries.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
            No summaries yet. One is written after each close, or generate today&apos;s now.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-[var(--color-ink-muted)]">
                <tr className="border-b border-[var(--color-border-subtle)]">
                  <th className="px-4 py-2 font-medium">Date</th>
                  <th className="px-4 py-2 text-right font-medium">Equity</th>
                  <th className="px-4 py-2 text-right font-medium">Change</th>
                  <th className="px-4 py-2 text-right font-medium">Realised</th>
                  <th className="px-4 py-2 text-right font-medium">Unrealised</th>
                  <th className="px-4 py-2 text-right font-medium">Open</th>
                </tr>
              </thead>
              <tbody>
                {summaries.map((s) => (
                  <tr key={s.id} className="border-b border-[var(--color-border-subtle)]">
                    <td className="px-4 py-2">{s.summary_date}</td>
                    <td className="tabular px-4 py-2 text-right">
                      {formatMoney(s.equity, s.currency)}
                    </td>
                    <td className="tabular px-4 py-2 text-right">
                      {s.equity_change == null ? "—" : formatMoney(s.equity_change, s.currency)}
                    </td>
                    <td className="tabular px-4 py-2 text-right">
                      {formatMoney(s.realised_pnl, s.currency)}
                    </td>
                    <td className="tabular px-4 py-2 text-right">
                      {formatMoney(s.unrealised_pnl, s.currency)}
                    </td>
                    <td className="tabular px-4 py-2 text-right">{s.open_positions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)] px-4 py-3">
      <p className="text-xs text-[var(--color-ink-muted)]">{label}</p>
      <p className="tabular mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}
