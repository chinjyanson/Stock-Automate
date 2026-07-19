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
  type RiskConfig,
  api,
  formatMoney,
} from "@/lib/api";

export default function PortfolioPage() {
  const router = useRouter();

  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [config, setConfig] = useState<RiskConfig | null>(null);
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
      // Each panel resolves independently: a missing risk config (fresh install)
      // must not blank the positions, and vice versa.
      const [accountR, positionsR, ordersR, configR, haltsR, summariesR, instrumentsR] =
        await Promise.allSettled([
          api.paperAccount(),
          api.paperPositions(),
          api.paperOrders(false),
          api.riskConfig(),
          api.halts(true),
          api.paperSummaries(14),
          api.instruments({ limit: 200 }),
        ]);

      if (accountR.status === "fulfilled") setAccount(accountR.value);
      if (positionsR.status === "fulfilled") setPositions(positionsR.value);
      if (ordersR.status === "fulfilled") setOrders(ordersR.value);
      if (configR.status === "fulfilled") setConfig(configR.value);
      if (haltsR.status === "fulfilled") setHalts(haltsR.value);
      if (summariesR.status === "fulfilled") setSummaries(summariesR.value);
      if (instrumentsR.status === "fulfilled") setInstruments(instrumentsR.value.items);

      if (accountR.status === "rejected") {
        const reason = accountR.reason;
        setError(
          reason instanceof ApiError || reason instanceof ApiUnreachableError
            ? reason.message
            : "Could not load the paper portfolio.",
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
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Paper portfolio</h1>
          <p className="text-sm text-[var(--color-ink-muted)]">
            The internal paper venue — positions, protective stops, and the risk limits every
            order passes through. Nothing here trades real money.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <a
            href="/scanner"
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm"
          >
            Scanner
          </a>
          <a
            href="/dashboard"
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm"
          >
            Dashboard
          </a>
        </div>
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

      <RiskConfigCard config={config} onSaved={load} onError={setError} />

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

function RiskConfigCard({
  config,
  onSaved,
  onError,
}: {
  config: RiskConfig | null;
  onSaved: () => Promise<void>;
  onError: (message: string) => void;
}) {
  const [saving, setSaving] = useState(false);
  // Editable copies of the primary sizing limits; the rest are shown read-only.
  const [riskPct, setRiskPct] = useState("");
  const [atrMult, setAtrMult] = useState("");
  const [maxPositionPct, setMaxPositionPct] = useState("");
  const [maxOpen, setMaxOpen] = useState("");
  const [maxTrades, setMaxTrades] = useState("");
  const [trailingEnabled, setTrailingEnabled] = useState(true);
  const [maxHoldingDays, setMaxHoldingDays] = useState("");

  useEffect(() => {
    if (!config) return;
    setRiskPct(config.risk_per_trade_pct);
    setAtrMult(config.atr_stop_multiplier);
    setMaxPositionPct(config.max_position_pct);
    setMaxOpen(String(config.max_open_positions));
    setMaxTrades(String(config.max_trades_per_day));
    setTrailingEnabled(config.trailing_stop_enabled);
    setMaxHoldingDays(String(config.max_holding_days));
  }, [config]);

  if (!config) {
    return (
      <section className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-6 text-sm text-[var(--color-ink-muted)]">
        No active risk configuration. Seed one (<span className="font-mono">python -m app.seed</span>)
        before trading — the engine fails closed without it.
      </section>
    );
  }

  async function onSave() {
    setSaving(true);
    try {
      await api.updateRiskConfig({
        risk_per_trade_pct: riskPct,
        atr_stop_multiplier: atrMult,
        max_position_pct: maxPositionPct,
        max_open_positions: Number(maxOpen),
        max_trades_per_day: Number(maxTrades),
        trailing_stop_enabled: trailingEnabled,
        max_holding_days: Number(maxHoldingDays),
      });
      await onSaved();
    } catch (err) {
      onError(err instanceof ApiError ? err.message : "Could not update the risk configuration");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="rounded-lg border border-[var(--color-border-subtle)]">
      <div className="flex items-center justify-between border-b border-[var(--color-border-subtle)] px-4 py-3">
        <h2 className="font-medium">Risk configuration — {config.name}</h2>
        <button
          onClick={onSave}
          disabled={saving}
          className="rounded-md bg-[var(--color-paper)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save changes"}
        </button>
      </div>
      <div className="grid gap-4 px-4 py-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Risk per trade" hint="Fraction of equity risked to the stop">
          <input
            value={riskPct}
            onChange={(e) => setRiskPct(e.target.value)}
            className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
          />
        </Field>
        <Field label="ATR stop multiplier" hint="Stop distance = ATR × this">
          <input
            value={atrMult}
            onChange={(e) => setAtrMult(e.target.value)}
            className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
          />
        </Field>
        <Field label="Max position %" hint="Cap on a single position's value">
          <input
            value={maxPositionPct}
            onChange={(e) => setMaxPositionPct(e.target.value)}
            className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
          />
        </Field>
        <Field label="Max open positions" hint="Bounds concentration">
          <input
            value={maxOpen}
            onChange={(e) => setMaxOpen(e.target.value)}
            className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
          />
        </Field>
        <Field label="Max trades / day" hint="Bounds runaway logic">
          <input
            value={maxTrades}
            onChange={(e) => setMaxTrades(e.target.value)}
            className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
          />
        </Field>
        <Field label="Max holding days" hint="Time stop; 0 disables it">
          <input
            value={maxHoldingDays}
            onChange={(e) => setMaxHoldingDays(e.target.value)}
            className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
          />
        </Field>
        <label className="flex items-center gap-2 self-end pb-1">
          <input
            type="checkbox"
            checked={trailingEnabled}
            onChange={(e) => setTrailingEnabled(e.target.checked)}
          />
          <span className="text-xs font-medium">Trailing stop enabled</span>
        </label>
      </div>
      <div className="grid gap-3 border-t border-[var(--color-border-subtle)] px-4 py-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
        <ReadOnly label="Max total open risk" value={config.max_total_open_risk_pct} />
        <ReadOnly label="Max instrument %" value={config.max_instrument_pct} />
        <ReadOnly label="Max daily realised loss" value={config.max_daily_realised_loss_pct} />
        <ReadOnly label="Max portfolio drawdown" value={config.max_portfolio_drawdown_pct} />
        <ReadOnly
          label="Correlation benchmark"
          value={`${config.correlation_benchmark_symbol} (>${config.correlation_threshold})`}
        />
        <ReadOnly label="Max S&P exposure" value={config.max_portfolio_sp500_pct} />
      </div>
    </section>
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

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium">{label}</span>
      <span className="mb-1 block text-xs text-[var(--color-ink-muted)]">{hint}</span>
      {children}
    </label>
  );
}

function ReadOnly({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-xs text-[var(--color-ink-muted)]">{label}</span>
      <p className="tabular font-medium">{value}</p>
    </div>
  );
}
