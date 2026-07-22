"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Account,
  type LiveStatus,
  type RiskConfig,
  api,
  formatMoney,
} from "@/lib/api";

export default function SettingsPage() {
  const router = useRouter();
  const [account, setAccount] = useState<Account | null>(null);
  const [live, setLive] = useState<LiveStatus | null>(null);
  const [risk, setRisk] = useState<RiskConfig | null>(null);
  const [scanAuto, setScanAuto] = useState<boolean | null>(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      try {
        const session = await api.me();
        setIsAdmin(session.user.is_admin);
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.push("/login");
          return;
        }
      }
      const [accountR, liveR, riskR, scanR] = await Promise.allSettled([
        api.activeAccount(),
        api.liveStatus(),
        api.riskConfig(),
        api.scannerSettings(),
      ]);
      if (accountR.status === "fulfilled") setAccount(accountR.value);
      if (liveR.status === "fulfilled") setLive(liveR.value);
      if (riskR.status === "fulfilled") setRisk(riskR.value);
      if (scanR.status === "fulfilled") setScanAuto(scanR.value.auto_run_enabled);
      if (liveR.status === "rejected") {
        const r = liveR.reason;
        setError(
          r instanceof ApiError || r instanceof ApiUnreachableError
            ? r.message
            : "Could not load settings.",
        );
      }
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <main className="mx-auto max-w-4xl px-6 py-10">
        <p className="text-sm text-[var(--color-ink-muted)]">Loading…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-4xl space-y-8 px-6 py-10">
      <header>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">
          Account, live-trading controls, and the risk limits every order passes through.
        </p>
      </header>

      {error && (
        <div className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm whitespace-pre-wrap">
          {error}
        </div>
      )}

      <AccountSection account={account} />
      <ScannerSection enabled={scanAuto} onChange={load} onError={setError} />
      <LiveSection live={live} isAdmin={isAdmin} onChange={load} onError={setError} />
      <RiskSection risk={risk} onSaved={load} onError={setError} />
    </main>
  );
}

/* -- Scanner ------------------------------------------------------------- */

function ScannerSection({
  enabled,
  onChange,
  onError,
}: {
  enabled: boolean | null;
  onChange: () => Promise<void>;
  onError: (m: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  async function toggle() {
    setBusy(true);
    onError("");
    try {
      await api.setScannerAutoRun(!enabled);
      await onChange();
    } catch (err) {
      onError(err instanceof ApiError ? err.message : "Could not update the scanner setting");
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-medium">Scanner</h2>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[var(--color-border-subtle)] px-4 py-3">
        <div>
          <p className="font-medium">Run automatically</p>
          <p className="text-sm text-[var(--color-ink-muted)]">
            Runs the rotating scan on a daily schedule. Currently{" "}
            <span className="font-medium">{enabled ? "on" : "off"}</span>. Requires the
            background worker to be running; manual scans always work.
          </p>
        </div>
        <button
          onClick={toggle}
          disabled={busy || enabled === null}
          className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
        >
          {enabled ? "Turn off" : "Turn on"}
        </button>
      </div>
    </section>
  );
}

/* -- Account ------------------------------------------------------------- */

function AccountSection({ account }: { account: Account | null }) {
  const currency = account?.currency ?? "GBP";
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-medium">Account</h2>
      {account ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Broker" value={account.broker.replace(/_/g, " ")} />
          <Stat label="Value" value={formatMoney(account.total, currency)} />
          <Stat label="Cash" value={formatMoney(account.cash, currency)} />
          <Stat label="Result" value={formatMoney(account.result, currency)} />
        </div>
      ) : (
        <p className="text-sm text-[var(--color-ink-muted)]">
          No broker account available. Configure a Trading 212 key to connect one.
        </p>
      )}
    </section>
  );
}

/* -- Live trading -------------------------------------------------------- */

function LiveSection({
  live,
  isAdmin,
  onChange,
  onError,
}: {
  live: LiveStatus | null;
  isAdmin: boolean;
  onChange: () => Promise<void>;
  onError: (m: string) => void;
}) {
  const [busy, setBusy] = useState(false);

  function isReauthError(err: unknown): boolean {
    return (
      err instanceof ApiError &&
      err.status === 403 &&
      typeof err.detail === "object" &&
      err.detail !== null &&
      (err.detail as { code?: string }).code === "reauthentication_required"
    );
  }

  async function withReauth(action: () => Promise<unknown>) {
    setBusy(true);
    onError("");
    try {
      await action();
      await onChange();
    } catch (err) {
      if (isReauthError(err)) {
        const pw = window.prompt("Confirm your password to continue:");
        if (!pw) return;
        try {
          await api.reauthenticate(pw);
          await action();
          await onChange();
        } catch (err2) {
          onError(err2 instanceof ApiError ? err2.message : "Action failed");
        }
      } else {
        onError(err instanceof ApiError ? err.message : "Action failed");
      }
    } finally {
      setBusy(false);
    }
  }

  const isLive = live?.live_mode === true;
  const blocked = (live?.blockers.length ?? 0) > 0;

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-medium">Trading venue</h2>

      <div
        className={
          "rounded-lg border-2 px-4 py-4 " +
          (isLive
            ? "border-[var(--color-live)] bg-[var(--color-live-soft)]"
            : "border-[var(--color-paper)] bg-[var(--color-paper-soft)]")
        }
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p
              className="font-semibold"
              style={{ color: isLive ? "var(--color-live)" : "var(--color-paper)" }}
            >
              {isLive ? "LIVE — orders place real money" : "PAPER — Trading 212 demo"}
            </p>
            <p className="mt-1 text-sm">
              {isLive
                ? "The portfolio and dashboard show your live account."
                : "Everything trades the demo account. No real money is at risk."}
            </p>
          </div>
          <button
            onClick={() => withReauth(() => api.setLiveMode(!isLive))}
            disabled={busy || (!isLive && blocked)}
            title={!isLive && blocked ? "Resolve the blockers below first" : undefined}
            className={
              "rounded-md px-4 py-2 text-sm font-semibold text-white disabled:opacity-50 " +
              (isLive ? "bg-[var(--color-paper)]" : "bg-[var(--color-live)]")
            }
          >
            {isLive ? "Switch to paper" : "Switch to live"}
          </button>
        </div>

        {!isLive && blocked && (
          <div className="mt-3 rounded-md border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-3 py-2 text-sm">
            <p className="font-medium text-[var(--color-warn)]">
              Live is unavailable until:
            </p>
            <ul className="mt-1 list-disc pl-5">
              {live?.blockers.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {isAdmin && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[var(--color-border-subtle)] px-4 py-3">
          <div>
            <p className="font-medium">Autonomous mode (admin)</p>
            <p className="text-sm text-[var(--color-ink-muted)]">
              Whether strategies may place live orders with no per-trade approval. Currently{" "}
              <span
                className="font-medium"
                style={{
                  color: live?.autonomous_enabled_on_server
                    ? "var(--color-live)"
                    : "var(--color-ink-muted)",
                }}
              >
                {live?.autonomous_enabled_on_server ? "enabled" : "disabled"}
              </span>
              . Requires a fresh password.
            </p>
          </div>
          <button
            onClick={() =>
              withReauth(() => api.setAutonomousEnabled(!live?.autonomous_enabled_on_server))
            }
            disabled={busy}
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
          >
            {live?.autonomous_enabled_on_server ? "Disable" : "Enable"}
          </button>
        </div>
      )}

      <p className="text-xs text-[var(--color-ink-muted)]">
        Live also requires the server to permit it (LIVE_TRADING_ENABLED plus live
        credentials) — that gate cannot be turned on from here. Live position sizing is
        bounded by the max live capital and max daily loss set under Risk configuration.
      </p>
    </section>
  );
}

/* -- Risk configuration -------------------------------------------------- */

function RiskSection({
  risk,
  onSaved,
  onError,
}: {
  risk: RiskConfig | null;
  onSaved: () => Promise<void>;
  onError: (m: string) => void;
}) {
  const [saving, setSaving] = useState(false);
  const [riskPct, setRiskPct] = useState("");
  const [atrMult, setAtrMult] = useState("");
  const [maxPositionPct, setMaxPositionPct] = useState("");
  const [maxOpen, setMaxOpen] = useState("");
  const [maxTrades, setMaxTrades] = useState("");
  const [maxHoldingDays, setMaxHoldingDays] = useState("");
  const [trailing, setTrailing] = useState(true);
  const [maxLiveCapital, setMaxLiveCapital] = useState("");
  const [maxDailyLoss, setMaxDailyLoss] = useState("");

  useEffect(() => {
    if (!risk) return;
    setRiskPct(risk.risk_per_trade_pct);
    setAtrMult(risk.atr_stop_multiplier);
    setMaxPositionPct(risk.max_position_pct);
    setMaxOpen(String(risk.max_open_positions));
    setMaxTrades(String(risk.max_trades_per_day));
    setMaxHoldingDays(String(risk.max_holding_days));
    setTrailing(risk.trailing_stop_enabled);
    setMaxLiveCapital(risk.max_live_capital ?? "");
    setMaxDailyLoss(risk.max_daily_loss ?? "");
  }, [risk]);

  if (!risk) {
    return (
      <section className="space-y-3">
        <h2 className="text-lg font-medium">Risk configuration</h2>
        <p className="text-sm text-[var(--color-ink-muted)]">
          No active risk configuration. Seed one (
          <span className="font-mono">python -m app.seed</span>) before trading.
        </p>
      </section>
    );
  }

  async function onSave() {
    setSaving(true);
    onError("");
    try {
      await api.updateRiskConfig({
        risk_per_trade_pct: riskPct,
        atr_stop_multiplier: atrMult,
        max_position_pct: maxPositionPct,
        max_open_positions: Number(maxOpen),
        max_trades_per_day: Number(maxTrades),
        max_holding_days: Number(maxHoldingDays),
        trailing_stop_enabled: trailing,
        // Blank means "no ceiling" — send null rather than 0.
        max_live_capital: maxLiveCapital.trim() === "" ? null : maxLiveCapital,
        max_daily_loss: maxDailyLoss.trim() === "" ? null : maxDailyLoss,
      });
      await onSaved();
    } catch (err) {
      onError(err instanceof ApiError ? err.message : "Could not update risk configuration");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium">Risk configuration</h2>
        <button
          onClick={onSave}
          disabled={saving}
          className="rounded-md bg-[var(--color-paper)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save changes"}
        </button>
      </div>
      <div className="grid gap-4 rounded-lg border border-[var(--color-border-subtle)] px-4 py-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Risk per trade" hint="Fraction of equity risked to the stop">
          <TextInput value={riskPct} onChange={setRiskPct} />
        </Field>
        <Field label="ATR stop multiplier" hint="Stop distance = ATR × this">
          <TextInput value={atrMult} onChange={setAtrMult} />
        </Field>
        <Field label="Max position %" hint="Cap on a single position">
          <TextInput value={maxPositionPct} onChange={setMaxPositionPct} />
        </Field>
        <Field label="Max open positions">
          <TextInput value={maxOpen} onChange={setMaxOpen} />
        </Field>
        <Field label="Max trades / day">
          <TextInput value={maxTrades} onChange={setMaxTrades} />
        </Field>
        <Field label="Max holding days" hint="Time stop; 0 disables">
          <TextInput value={maxHoldingDays} onChange={setMaxHoldingDays} />
        </Field>
        <Field label="Max live capital" hint="Live only; blank = no ceiling">
          <TextInput value={maxLiveCapital} onChange={setMaxLiveCapital} />
        </Field>
        <Field label="Max daily loss" hint="Live only; breach halts and reverts to paper">
          <TextInput value={maxDailyLoss} onChange={setMaxDailyLoss} />
        </Field>
        <label className="flex items-center gap-2 self-end pb-1">
          <input type="checkbox" checked={trailing} onChange={(e) => setTrailing(e.target.checked)} />
          <span className="text-xs font-medium">Trailing stop enabled</span>
        </label>
      </div>
    </section>
  );
}

/* -- Small building blocks ---------------------------------------------- */

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
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium">{label}</span>
      {hint && (
        <span className="mb-1 block font-mono text-xs text-[var(--color-ink-muted)]">{hint}</span>
      )}
      {children}
    </label>
  );
}

function TextInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
    />
  );
}
