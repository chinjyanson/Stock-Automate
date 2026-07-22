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

export default function StrategiesPage() {
  const router = useRouter();
  const toast = useToast();
  const [strategies, setStrategies] = useState<StrategyConfig[]>([]);
  const [decisions, setDecisions] = useState<StrategyDecision[]>([]);
  const [instruments, setInstruments] = useState<Instrument[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
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
      const [strategiesR, decisionsR, instrumentsR] = await Promise.allSettled([
        api.strategies(),
        api.strategyDecisions({ limit: 40 }),
        api.instruments({ limit: 200 }),
      ]);
      if (strategiesR.status === "fulfilled") setStrategies(strategiesR.value);
      if (decisionsR.status === "fulfilled") setDecisions(decisionsR.value);
      if (instrumentsR.status === "fulfilled") setInstruments(instrumentsR.value.items);
      if (strategiesR.status === "rejected") {
        const reason = strategiesR.reason;
        setError(
          reason instanceof ApiError || reason instanceof ApiUnreachableError
            ? reason.message
            : "Could not load strategies.",
        );
      }
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  const nameFor = useMemo(() => {
    const byId = new Map(instruments.map((i) => [i.id, i.name]));
    return (id: string) => byId.get(id) ?? id;
  }, [instruments]);

  async function onToggleActive(strategy: StrategyConfig) {
    setBusyId(strategy.id);
    try {
      await api.updateStrategy(strategy.id, { is_active: !strategy.is_active });
      toast.success(`${strategy.name} ${strategy.is_active ? "deactivated" : "activated"}.`);
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update the strategy");
    } finally {
      setBusyId(null);
    }
  }

  async function onRun(strategy: StrategyConfig) {
    setBusyId(strategy.id);
    try {
      const result = await api.runStrategy(strategy.id);
      toast.success(
        `${strategy.name}: ${result.signals} signal(s), ${result.executed} executed, ` +
          `${result.rejected} rejected.`,
      );
      await load();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not run the strategy");
    } finally {
      setBusyId(null);
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
      <header>
        <h1 className="text-2xl font-semibold">Strategies</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">
          Automated strategies propose trades; the risk engine still sizes and gates every
          one. Inactive by default — nothing trades until you map a universe and turn it on.
        </p>
      </header>

      {error && (
        <div className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm">
          {error}
        </div>
      )}

      <section className="space-y-3">
        {strategies.length === 0 ? (
          <div className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-10 text-center text-sm text-[var(--color-ink-muted)]">
            No strategies configured. Seed the defaults with{" "}
            <span className="font-mono">python -m app.seed</span>.
          </div>
        ) : (
          strategies.map((s) => {
            const universeCount =
              (Array.isArray(s.universe?.instrument_ids)
                ? (s.universe?.instrument_ids as unknown[]).length
                : 0) +
              (s.universe?.weights ? Object.keys(s.universe.weights).length : 0);
            return (
              <div
                key={s.id}
                className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="font-medium">{s.name}</h2>
                      <span
                        className="rounded px-2 py-0.5 text-xs"
                        style={{
                          color: s.is_active ? "var(--color-ok)" : "var(--color-ink-muted)",
                          border: "1px solid var(--color-border-subtle)",
                        }}
                      >
                        {s.is_active ? "Active" : "Inactive"}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--color-ink-muted)]">
                      {s.kind.replace(/_/g, " ")} · {s.interval} · {universeCount} instrument
                      {universeCount === 1 ? "" : "s"} ·{" "}
                      {s.auto_execute ? "auto-executes (paper)" : "approval required"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setEditingId(editingId === s.id ? null : s.id)}
                      className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm"
                    >
                      {editingId === s.id ? "Close" : "Edit universe"}
                    </button>
                    <button
                      onClick={() => onToggleActive(s)}
                      disabled={busyId === s.id}
                      className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm disabled:opacity-50"
                    >
                      {s.is_active ? "Deactivate" : "Activate"}
                    </button>
                    <button
                      onClick={() => onRun(s)}
                      disabled={busyId === s.id || universeCount === 0}
                      title={universeCount === 0 ? "Map a universe first" : "Evaluate now"}
                      className="rounded-md bg-[var(--color-paper)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
                    >
                      {busyId === s.id ? "Running…" : "Run now"}
                    </button>
                  </div>
                </div>
                {editingId === s.id && (
                  <UniverseEditor
                    strategy={s}
                    nameFor={nameFor}
                    onSaved={load}
                    onClose={() => setEditingId(null)}
                  />
                )}
              </div>
            );
          })
        )}
      </section>

      <section className="rounded-lg border border-[var(--color-border-subtle)]">
        <h2 className="border-b border-[var(--color-border-subtle)] px-4 py-3 font-medium">
          Recent decisions
        </h2>
        {decisions.length === 0 ? (
          <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">
            No decisions yet. Activate a strategy with a mapped universe and run it.
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
                    <td className="px-4 py-2 uppercase">{d.side}</td>
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

      <footer className="pt-2 text-xs text-[var(--color-ink-muted)]">
        Strategy signals are gated by the risk engine and filled on the internal paper venue.
        Nothing here is investment advice.
      </footer>
    </main>
  );
}

type UniverseEntry = { id: string; name: string; weight: string };

function initialEntries(strategy: StrategyConfig, nameFor: (id: string) => string): UniverseEntry[] {
  const universe = strategy.universe ?? {};
  if (strategy.kind === "pie_rebalance") {
    const weights = (universe.weights as Record<string, unknown> | undefined) ?? {};
    return Object.entries(weights).map(([id, w]) => ({
      id,
      name: nameFor(id),
      weight: String(w),
    }));
  }
  const ids = Array.isArray(universe.instrument_ids)
    ? (universe.instrument_ids as unknown[]).map(String)
    : [];
  return ids.map((id) => ({ id, name: nameFor(id), weight: "" }));
}

function UniverseEditor({
  strategy,
  nameFor,
  onSaved,
  onClose,
}: {
  strategy: StrategyConfig;
  nameFor: (id: string) => string;
  onSaved: () => Promise<void>;
  onClose: () => void;
}) {
  const toast = useToast();
  const isPie = strategy.kind === "pie_rebalance";
  const [entries, setEntries] = useState<UniverseEntry[]>(() =>
    initialEntries(strategy, nameFor),
  );
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Instrument[]>([]);
  const [searching, setSearching] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    const t = setTimeout(async () => {
      setSearching(true);
      try {
        const r = await api.instruments({ search: query, limit: 20 });
        setResults(r.items);
      } catch {
        setResults([]);
      } finally {
        setSearching(false);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [query]);

  function add(inst: Instrument) {
    if (entries.some((e) => e.id === inst.id)) return;
    setEntries([...entries, { id: inst.id, name: inst.name, weight: isPie ? "0.1" : "" }]);
    setQuery("");
    setResults([]);
  }

  function remove(id: string) {
    setEntries(entries.filter((e) => e.id !== id));
  }

  function setWeight(id: string, weight: string) {
    setEntries(entries.map((e) => (e.id === id ? { ...e, weight } : e)));
  }

  const weightTotal = isPie
    ? entries.reduce((sum, e) => sum + (Number(e.weight) || 0), 0)
    : 0;

  async function save() {
    const universe = isPie
      ? { weights: Object.fromEntries(entries.map((e) => [e.id, Number(e.weight) || 0])) }
      : { instrument_ids: entries.map((e) => e.id) };
    setSaving(true);
    try {
      await api.updateStrategy(strategy.id, { universe });
      toast.success(`Universe saved for ${strategy.name}.`);
      await onSaved();
      onClose();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not save the universe");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mt-3 space-y-3 border-t border-[var(--color-border-subtle)] pt-3">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">
          Universe {isPie && `· weights total ${weightTotal.toFixed(2)}`}
        </p>
        <button
          onClick={save}
          disabled={saving}
          className="rounded-md bg-[var(--color-paper)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save universe"}
        </button>
      </div>

      {entries.length === 0 ? (
        <p className="text-sm text-[var(--color-ink-muted)]">
          Nothing yet — search below to add instruments.
        </p>
      ) : (
        <ul className="space-y-1">
          {entries.map((e) => (
            <li
              key={e.id}
              className="flex items-center justify-between gap-3 rounded-md bg-[var(--color-surface-muted)] px-3 py-1.5 text-sm"
            >
              <span className="truncate">{e.name}</span>
              <span className="flex items-center gap-2">
                {isPie && (
                  <input
                    value={e.weight}
                    onChange={(ev) => setWeight(e.id, ev.target.value)}
                    title="Target weight (fraction, e.g. 0.25)"
                    className="w-20 rounded-md border border-[var(--color-border-subtle)] px-2 py-0.5 text-right text-sm"
                  />
                )}
                <button
                  onClick={() => remove(e.id)}
                  className="text-[var(--color-warn)] hover:underline"
                >
                  Remove
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}

      <div className="relative">
        <input
          value={query}
          onChange={(ev) => setQuery(ev.target.value)}
          placeholder="Search instruments to add (name or ticker)…"
          className="w-full rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm"
        />
        {query.trim().length >= 2 && (
          <div className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface)] shadow">
            {searching ? (
              <p className="px-3 py-2 text-sm text-[var(--color-ink-muted)]">Searching…</p>
            ) : results.length === 0 ? (
              <p className="px-3 py-2 text-sm text-[var(--color-ink-muted)]">No matches.</p>
            ) : (
              results.map((inst) => (
                <button
                  key={inst.id}
                  onClick={() => add(inst)}
                  disabled={entries.some((e) => e.id === inst.id)}
                  className="flex w-full items-center justify-between px-3 py-1.5 text-left text-sm hover:bg-[var(--color-surface-muted)] disabled:opacity-40"
                >
                  <span className="truncate">{inst.name}</span>
                  <span className="ml-2 shrink-0 text-xs text-[var(--color-ink-muted)]">
                    {inst.exchange_ticker ?? ""}
                  </span>
                </button>
              ))
            )}
          </div>
        )}
      </div>

      <p className="text-xs text-[var(--color-ink-muted)]">
        An instrument only produces signals once it has stored market data (daily candles, or
        15-minute for mean reversion). Adding it here assigns it; ingestion is separate.
      </p>
    </div>
  );
}
