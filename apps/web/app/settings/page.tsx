"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Account,
  type AuditEvent,
  type DailySummary,
  type LiveStatus,
  type RiskConfig,
  type User,
  api,
  formatMoney,
} from "@/lib/api";
import { Stat } from "@/components/Stat";
import { useToast } from "@/components/Toast";

type SectionId =
  | "account"
  | "trading"
  | "risk"
  | "scanner"
  | "notifications"
  | "audit";

const SECTIONS: { id: SectionId; label: string }[] = [
  { id: "account", label: "Account" },
  { id: "trading", label: "Trading venue" },
  { id: "risk", label: "Risk" },
  { id: "scanner", label: "Scanner" },
  { id: "notifications", label: "Notifications" },
  { id: "audit", label: "Audit log" },
];

export default function SettingsPage() {
  const router = useRouter();
  const toast = useToast();
  const [section, setSection] = useState<SectionId>("account");

  const [user, setUser] = useState<User | null>(null);
  const [account, setAccount] = useState<Account | null>(null);
  const [live, setLive] = useState<LiveStatus | null>(null);
  const [risk, setRisk] = useState<RiskConfig | null>(null);
  const [scanAuto, setScanAuto] = useState<boolean | null>(null);
  const [eodDigest, setEodDigest] = useState<boolean | null>(null);
  const [summaries, setSummaries] = useState<DailySummary[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      try {
        const session = await api.me();
        setUser(session.user);
        setIsAdmin(session.user.is_admin);
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          router.push("/login");
          return;
        }
      }
      const [accountR, liveR, riskR, scanR, digestR, summariesR, auditR] =
        await Promise.allSettled([
          api.activeAccount(),
          api.liveStatus(),
          api.riskConfig(),
          api.scannerSettings(),
          api.notificationSettings(),
          api.activeSummaries(14),
          api.audit(25),
        ]);
      if (accountR.status === "fulfilled") setAccount(accountR.value);
      if (liveR.status === "fulfilled") setLive(liveR.value);
      if (riskR.status === "fulfilled") setRisk(riskR.value);
      if (scanR.status === "fulfilled") setScanAuto(scanR.value.auto_run_enabled);
      if (digestR.status === "fulfilled") setEodDigest(digestR.value.eod_digest_enabled);
      if (summariesR.status === "fulfilled") setSummaries(summariesR.value);
      if (auditR.status === "fulfilled") setAudit(auditR.value.items);
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

  // Re-auth wrapper shared by the account form and the live controls: run the
  // action, and if the API demands a fresh password, prompt and retry once.
  // Reports the outcome as a toast — success (when a message is given) or error.
  const withReauth = useCallback(
    async (action: () => Promise<unknown>, successMessage?: string) => {
      try {
        await action();
        if (successMessage) toast.success(successMessage);
        await load();
      } catch (err) {
        if (isReauthError(err)) {
          const pw = window.prompt("Confirm your password to continue:");
          if (!pw) return;
          try {
            await api.reauthenticate(pw);
            await action();
            if (successMessage) toast.success(successMessage);
            await load();
          } catch (err2) {
            toast.error(err2 instanceof ApiError ? err2.message : "Action failed");
          }
        } else {
          toast.error(err instanceof ApiError ? err.message : "Action failed");
        }
      }
    },
    [load, toast],
  );

  if (loading) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-10">
        <p className="text-sm text-[var(--color-ink-muted)]">Loading…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">
          Account, live-trading controls, risk limits, and notifications.
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm whitespace-pre-wrap">
          {error}
        </div>
      )}

      <div className="flex flex-col gap-8 sm:flex-row">
        <nav className="shrink-0 sm:w-48">
          <ul className="flex gap-1 overflow-x-auto sm:sticky sm:top-20 sm:flex-col sm:overflow-visible">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <button
                  onClick={() => setSection(s.id)}
                  aria-current={section === s.id ? "true" : undefined}
                  className={
                    "w-full whitespace-nowrap rounded-md px-3 py-2 text-left text-sm transition-colors " +
                    (section === s.id
                      ? "bg-[var(--color-surface-muted)] font-medium text-[var(--color-ink)]"
                      : "text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]")
                  }
                >
                  {s.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="min-w-0 flex-1 space-y-8">
          {section === "account" && (
            <AccountSection user={user} account={account} onChange={load} withReauth={withReauth} />
          )}
          {section === "trading" && (
            <LiveSection live={live} isAdmin={isAdmin} withReauth={withReauth} />
          )}
          {section === "risk" && <RiskSection risk={risk} onSaved={load} />}
          {section === "scanner" && <ScannerSection enabled={scanAuto} onChange={load} />}
          {section === "notifications" && (
            <NotificationsSection enabled={eodDigest} summaries={summaries} onChange={load} />
          )}
          {section === "audit" && <AuditSection events={audit} />}
        </div>
      </div>
    </main>
  );
}

function isReauthError(err: unknown): boolean {
  return (
    err instanceof ApiError &&
    err.status === 403 &&
    typeof err.detail === "object" &&
    err.detail !== null &&
    (err.detail as { code?: string }).code === "reauthentication_required"
  );
}

/* -- Account ------------------------------------------------------------- */

function AccountSection({
  user,
  account,
  onChange,
  withReauth,
}: {
  user: User | null;
  account: Account | null;
  onChange: () => Promise<void>;
  withReauth: (action: () => Promise<unknown>, successMessage?: string) => Promise<void>;
}) {
  const toast = useToast();
  const [name, setName] = useState(user?.display_name ?? "");
  const [email, setEmail] = useState(user?.email ?? "");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const currency = account?.currency ?? "GBP";

  async function saveName() {
    setBusy(true);
    try {
      await api.updateDisplayName(name.trim() === "" ? null : name.trim());
      toast.success("Display name saved.");
      await onChange();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update the display name");
    } finally {
      setBusy(false);
    }
  }

  async function saveEmail() {
    setBusy(true);
    await withReauth(() => api.updateEmail(email.trim()), "Email updated.");
    setBusy(false);
  }

  async function savePassword() {
    if (password.length < 12) {
      toast.error("Password must be at least 12 characters.");
      return;
    }
    setBusy(true);
    await withReauth(() => api.changePassword(password), "Password changed.");
    setPassword("");
    setBusy(false);
  }

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-medium">Account</h2>

      <div className="space-y-4 rounded-lg border border-[var(--color-border-subtle)] px-4 py-4">
        <Field label="Display name">
          <div className="flex gap-2">
            <TextInput value={name} onChange={setName} />
            <button
              onClick={saveName}
              disabled={busy}
              className="shrink-0 rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </Field>

        <Field label="Email" hint="Your sign-in address — the EOD digest is sent here. Needs your password.">
          <div className="flex gap-2">
            <TextInput value={email} onChange={setEmail} type="email" />
            <button
              onClick={saveEmail}
              disabled={busy}
              className="shrink-0 rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
            >
              Change
            </button>
          </div>
        </Field>

        <Field label="New password" hint="At least 12 characters. Needs your current password.">
          <div className="flex gap-2">
            <TextInput value={password} onChange={setPassword} type="password" />
            <button
              onClick={savePassword}
              disabled={busy || password.length === 0}
              className="shrink-0 rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
            >
              Update
            </button>
          </div>
        </Field>
      </div>

      <h3 className="text-sm font-medium text-[var(--color-ink-muted)]">Connected broker</h3>
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

/* -- Notifications ------------------------------------------------------- */

function NotificationsSection({
  enabled,
  summaries,
  onChange,
}: {
  enabled: boolean | null;
  summaries: DailySummary[];
  onChange: () => Promise<void>;
}) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);

  async function toggle() {
    setBusy(true);
    try {
      await api.setEodDigest(!enabled);
      toast.success(`Daily email digest turned ${!enabled ? "on" : "off"}.`);
      await onChange();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update the notification setting");
    } finally {
      setBusy(false);
    }
  }

  async function generateNow() {
    setBusy(true);
    try {
      const summary = await api.runEodSummary();
      toast.success(`Summary generated for ${summary.summary_date}.`);
      await onChange();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not generate the summary");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-medium">Notifications</h2>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[var(--color-border-subtle)] px-4 py-3">
        <div>
          <p className="font-medium">Daily end-of-day email digest</p>
          <p className="text-sm text-[var(--color-ink-muted)]">
            Emails the day&apos;s summary — equity, P/L and open positions — to your account
            address after the close. Currently{" "}
            <span className="font-medium">{enabled ? "on" : "off"}</span>. Requires the background
            worker and configured mail credentials.
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

      <div className="rounded-lg border border-[var(--color-border-subtle)]">
        <div className="flex items-center justify-between border-b border-[var(--color-border-subtle)] px-4 py-3">
          <h3 className="font-medium">End-of-day summaries</h3>
          <button
            onClick={generateNow}
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
      </div>
    </section>
  );
}

/* -- Audit log ----------------------------------------------------------- */

function AuditSection({ events }: { events: AuditEvent[] }) {
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-medium">Audit log</h2>
      <div className="rounded-lg border border-[var(--color-border-subtle)]">
        {events.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">No events yet.</p>
        ) : (
          <ul className="divide-y divide-[var(--color-border-subtle)]">
            {events.map((event) => (
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
      </div>
    </section>
  );
}

/* -- Scanner ------------------------------------------------------------- */

function ScannerSection({
  enabled,
  onChange,
}: {
  enabled: boolean | null;
  onChange: () => Promise<void>;
}) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  async function toggle() {
    setBusy(true);
    try {
      await api.setScannerAutoRun(!enabled);
      toast.success(`Scheduled scanning turned ${!enabled ? "on" : "off"}.`);
      await onChange();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update the scanner setting");
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

/* -- Live trading -------------------------------------------------------- */

function LiveSection({
  live,
  isAdmin,
  withReauth,
}: {
  live: LiveStatus | null;
  isAdmin: boolean;
  withReauth: (action: () => Promise<unknown>, successMessage?: string) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);

  async function run(action: () => Promise<unknown>, successMessage: string) {
    setBusy(true);
    await withReauth(action, successMessage);
    setBusy(false);
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
                ? "The dashboard shows your live account."
                : "Everything trades the demo account. No real money is at risk."}
            </p>
          </div>
          <button
            onClick={() =>
              run(
                () => api.setLiveMode(!isLive),
                isLive ? "Switched to paper trading." : "Switched to LIVE — real money.",
              )
            }
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
            <p className="font-medium text-[var(--color-warn)]">Live is unavailable until:</p>
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
              run(
                () => api.setAutonomousEnabled(!live?.autonomous_enabled_on_server),
                `Autonomous mode ${live?.autonomous_enabled_on_server ? "disabled" : "enabled"}.`,
              )
            }
            disabled={busy}
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
          >
            {live?.autonomous_enabled_on_server ? "Disable" : "Enable"}
          </button>
        </div>
      )}

      <p className="text-xs text-[var(--color-ink-muted)]">
        Live also requires the server to permit it (LIVE_TRADING_ENABLED plus live credentials) —
        that gate cannot be turned on from here. Live position sizing is bounded by the max live
        capital and max daily loss set under Risk.
      </p>
    </section>
  );
}

/* -- Risk configuration -------------------------------------------------- */

function RiskSection({
  risk,
  onSaved,
}: {
  risk: RiskConfig | null;
  onSaved: () => Promise<void>;
}) {
  const toast = useToast();
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
      toast.success("Risk configuration saved.");
      await onSaved();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update risk configuration");
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

function TextInput({
  value,
  onChange,
  type = "text",
}: {
  value: string;
  onChange: (v: string) => void;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-[var(--color-border-subtle)] px-2 py-1 text-sm"
    />
  );
}
