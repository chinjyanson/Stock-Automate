"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Instrument,
  type StrategyConfig,
  type StrategyDecision,
  api,
} from "@/lib/api";
import { useToast } from "@/components/Toast";

const OUTCOME_COLOUR: Record<string, string> = {
  executed: "var(--color-ok)",
  proposed: "var(--color-warn)",
  rejected_by_risk: "var(--color-warn)",
  skipped: "var(--color-ink-muted)",
  signalled: "var(--color-ink-muted)",
};

/** Parameter names in the order they are worth reading, with what each means. */
const PARAM_LABEL: Record<string, string> = {
  bb_period: "Bollinger period",
  bb_std: "Bollinger width (σ)",
  rsi_period: "RSI period",
  rsi_oversold: "RSI entry threshold",
  atr_period: "ATR period",
  min_atr_pct: "Minimum ATR (% of price)",
  insider_sell_veto: "Insider exit threshold",
  insider_exit_max_drop_atr: "Insider exit — max drop already seen (ATR)",
};

/**
 * The mean-reversion strategy: what it is watching, what it decided, and a way
 * to run it now.
 *
 * Deliberately not a list with an editor. There is one strategy, and its
 * universe is rewritten nightly from the scanner's ranking — so a universe
 * editor here would offer edits that the next sync silently discards, which is
 * worse than offering nothing. The universe is shown read-only, and the place
 * to influence it is the scanner (score, or pin to the watchlist).
 */
export default function StrategyPage() {
  const router = useRouter();
  const toast = useToast();
  const [strategy, setStrategy] = useState<StrategyConfig | null>(null);
  const [decisions, setDecisions] = useState<StrategyDecision[]>([]);
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
      const [strategiesR, decisionsR] = await Promise.allSettled([
        api.strategies(),
        api.strategyDecisions({ limit: 40 }),
      ]);
      if (strategiesR.status === "fulfilled") {
        setStrategy(
          strategiesR.value.find((s) => s.kind === "mean_reversion") ?? null,
        );
        setError(null);
      } else {
        const reason = strategiesR.reason;
        setError(
          reason instanceof ApiError || reason instanceof ApiUnreachableError
            ? reason.message
            : "Could not load the strategy.",
        );
      }
      if (decisionsR.status === "fulfilled") setDecisions(decisionsR.value);
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  // The universe holds instrument ids, and they are scattered through a 15k
  // catalogue — any single page of /instruments would mostly miss them, which
  // is why these are resolved by id rather than paged through.
  const universeIds = useMemo(() => {
    const raw = (strategy?.universe ?? {}) as { instrument_ids?: unknown[] };
    return (raw.instrument_ids ?? []).map(String);
  }, [strategy]);

  useEffect(() => {
    const missing = universeIds.filter((id) => !instruments.some((i) => i.id === id));
    if (missing.length === 0) return;
    let cancelled = false;
    void (async () => {
      try {
        for (let i = 0; i < missing.length; i += 100) {
          const page = await api.instruments({ ids: missing.slice(i, i + 100), limit: 100 });
          if (cancelled) return;
          setInstruments((prev) => {
            const seen = new Set(prev.map((p) => p.id));
            return [...prev, ...page.items.filter((it) => !seen.has(it.id))];
          });
        }
      } catch {
        // Non-fatal: rows fall back to a short id, which is what they showed
        // before names were resolved at all.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [universeIds, instruments]);

  const labelFor = useMemo(() => {
    const byId = new Map(instruments.map((i) => [i.id, i]));
    return (id: string) => {
      const i = byId.get(id);
      if (!i) return { name: null as string | null, ticker: null as string | null };
      return { name: i.name, ticker: i.exchange_ticker };
    };
  }, [instruments]);

  const nameFor = useMemo(
    () => (id: string) => {
      const l = labelFor(id);
      if (!l.name) return `${id.slice(0, 8)}…`;
      return l.ticker ? `${l.name} (${l.ticker})` : l.name;
    },
    [labelFor],
  );

  async function onToggleActive() {
    if (!strategy) return;
    setBusy(true);
    try {
      await api.updateStrategy(strategy.id, { is_active: !strategy.is_active });
      toast.success(strategy.is_active ? "Strategy deactivated." : "Strategy activated.");
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update the strategy");
    } finally {
      setBusy(false);
    }
  }

  async function onRun() {
    if (!strategy) return;
    setBusy(true);
    try {
      const result = await api.runStrategy(strategy.id);
      const outcome =
        `${result.signals} signal(s), ${result.executed} executed, ` +
        `${result.rejected} rejected.`;
      // A quiet run has two very different causes, and the counts cannot tell
      // them apart: the strategy looked and declined, or it never had usable
      // data. Say which, rather than let "0 signals" imply the first.
      if (result.skipped > 0) {
        toast.error(
          `${outcome} ${result.skipped} instrument(s) had too little history to ` +
            `evaluate — nothing was actually assessed.`,
        );
      } else if (result.stale > 0) {
        toast.error(
          `${outcome} Evaluated on stale data: ${result.stale} of ${result.considered} ` +
            `instrument(s) have no recent bars. Start the worker (pnpm worker:dev) to ` +
            `refresh candles — this result reflects old prices.`,
        );
      } else {
        toast.success(outcome);
      }
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not run the strategy");
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

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-6 py-10">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Mean reversion</h1>
          <p className="max-w-2xl text-sm text-[var(--color-ink-muted)]">
            The one strategy this product runs. The scanner decides <em>what</em> is worth
            owning; this decides <em>when</em> to buy and sell it, on daily candles using
            Bollinger Bands, RSI and ATR together. The risk engine still sizes and gates
            every entry.
          </p>
        </div>
        {strategy && (
          <div className="flex items-center gap-2">
            <button
              onClick={onToggleActive}
              disabled={busy}
              className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
            >
              {strategy.is_active ? "Deactivate" : "Activate"}
            </button>
            <button
              onClick={onRun}
              disabled={busy || universeIds.length === 0}
              title={
                universeIds.length === 0
                  ? "The universe is empty — run a scan, or wait for the nightly sync"
                  : "Evaluate now"
              }
              className="rounded-md bg-[var(--color-paper)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {busy ? "Running…" : "Run now"}
            </button>
          </div>
        )}
      </header>

      {error && (
        <div className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {!strategy ? (
        <div className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-10 text-center text-sm text-[var(--color-ink-muted)]">
          No mean-reversion strategy configured. Seed it with{" "}
          <span className="font-mono">python -m app.seed</span>.
        </div>
      ) : (
        <>
          <section className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-3">
              <h2 className="text-sm font-medium">Status</h2>
              <dl className="mt-2 space-y-1 text-sm">
                <Row
                  label="State"
                  value={strategy.is_active ? "Active" : "Inactive"}
                  tone={strategy.is_active ? "ok" : "muted"}
                />
                <Row label="Candles" value={strategy.interval} />
                <Row
                  label="Execution"
                  value={strategy.auto_execute ? "auto-executes (paper)" : "approval required"}
                />
                <Row
                  label="Universe"
                  value={`${universeIds.length} instrument${universeIds.length === 1 ? "" : "s"}`}
                />
              </dl>
            </div>

            <div className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-3">
              <h2 className="text-sm font-medium">Parameters</h2>
              <dl className="mt-2 space-y-1 text-sm">
                {Object.entries(strategy.params ?? {}).map(([k, v]) => (
                  <Row key={k} label={PARAM_LABEL[k] ?? k.replace(/_/g, " ")} value={String(v)} />
                ))}
              </dl>
            </div>
          </section>

          <section className="rounded-lg border border-[var(--color-border-subtle)]">
            <div className="border-b border-[var(--color-border-subtle)] px-4 py-3">
              <h2 className="font-medium">Universe</h2>
              <p className="text-xs text-[var(--color-ink-muted)]">
                Rewritten nightly from the scanner&rsquo;s top-ranked tradable names, plus
                anything currently held so its exit can still fire. Not editable here — to
                influence it, pin a stock on the{" "}
                <a href="/scanner" className="underline">
                  scanner
                </a>
                .
              </p>
            </div>
            {universeIds.length === 0 ? (
              <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
                Empty. It fills after the next scan and universe sync.
              </p>
            ) : (
              <ul className="flex flex-wrap gap-2 px-4 py-3 text-sm">
                {universeIds.map((id, i) => (
                  <li
                    key={id}
                    className="rounded-full border border-[var(--color-border-subtle)] px-2.5 py-0.5"
                  >
                    <span className="mr-1 text-xs text-[var(--color-ink-muted)]">{i + 1}</span>
                    {nameFor(id)}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-lg border border-[var(--color-border-subtle)]">
            <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
              Recent decisions
            </h2>
            {decisions.length === 0 ? (
              <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
                No decisions yet. Activate the strategy and run it.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-[var(--color-ink-muted)]">
                    <tr className="border-b border-[var(--color-border-subtle)]">
                      <th className="px-4 py-2 font-medium">When</th>
                      <th className="px-4 py-2 font-medium">Instrument</th>
                      <th className="px-4 py-2 font-medium">Side</th>
                      <th className="px-4 py-2 font-medium">Outcome</th>
                      <th className="px-4 py-2 font-medium">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decisions.map((d) => (
                      <tr key={d.id} className="border-b border-[var(--color-border-subtle)]">
                        <td className="px-4 py-2 text-[var(--color-ink-muted)]">
                          {new Date(d.created_at).toLocaleString()}
                        </td>
                        <td className="px-4 py-2 font-medium">{nameFor(d.instrument_id)}</td>
                        <td className="px-4 py-2 uppercase">
                          {d.side ?? <span className="text-[var(--color-ink-muted)]">&mdash;</span>}
                        </td>
                        <td
                          className="px-4 py-2"
                          style={{ color: OUTCOME_COLOUR[d.outcome] ?? "var(--color-ink)" }}
                        >
                          {d.outcome.replace(/_/g, " ")}
                        </td>
                        <td className="px-4 py-2 text-[var(--color-ink-muted)]">{d.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      <footer className="pt-2 text-xs text-[var(--color-ink-muted)]">
        Strategy signals are gated by the risk engine and filled on the internal paper venue.
        Nothing here is investment advice.
      </footer>
    </main>
  );
}

function Row({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "ok" | "muted" | "default";
}) {
  const colour =
    tone === "ok"
      ? "var(--color-ok)"
      : tone === "muted"
        ? "var(--color-ink-muted)"
        : "var(--color-ink)";
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-[var(--color-ink-muted)]">{label}</dt>
      <dd className="text-right" style={{ color: colour }}>
        {value}
      </dd>
    </div>
  );
}
