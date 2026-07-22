"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type ScannerResult,
  type ScannerResultDetail,
  api,
} from "@/lib/api";

const CLASSIFICATION_LABEL: Record<string, string> = {
  screening_candidate: "Screening candidate",
  watchlist_candidate: "Watchlist candidate",
  does_not_pass: "Does not pass the screen",
};

function scoreColour(classification: string): string {
  if (classification === "screening_candidate") return "var(--color-ok)";
  if (classification === "watchlist_candidate") return "var(--color-warn)";
  return "var(--color-ink-muted)";
}

// Number of body columns, so an expanded detail row can span the full table.
const COLUMN_COUNT = 9;

export default function ScannerPage() {
  const router = useRouter();
  const [results, setResults] = useState<ScannerResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Which row is expanded, and the detail for it (fetched on expand, cached).
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, ScannerResultDetail>>({});

  const load = useCallback(async () => {
    try {
      await api.me();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        router.push("/login");
        return;
      }
    }
    try {
      setResults(await api.scannerResults({ limit: 100 }));
    } catch (err) {
      setError(
        err instanceof ApiError || err instanceof ApiUnreachableError
          ? err.message
          : "Could not load scanner results",
      );
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  async function onRun() {
    setRunning(true);
    setError(null);
    try {
      await api.runScanner({ limit: 200 });
      setDetails({});
      setExpandedId(null);
      await load();
    } catch (err) {
      setError(
        err instanceof ApiError || err instanceof ApiUnreachableError ? err.message : "Scan failed",
      );
    } finally {
      setRunning(false);
    }
  }

  async function toggleRow(id: string) {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    // Fetch detail lazily, once per row.
    if (!details[id]) {
      try {
        const detail = await api.scannerResult(id);
        setDetails((prev) => ({ ...prev, [id]: detail }));
      } catch {
        // Non-fatal: the row stays expanded but empty.
      }
    }
  }

  if (loading) {
    return (
      <main className="mx-auto max-w-6xl px-6 py-10">
        <p className="text-sm text-[var(--color-ink-muted)]">Loading…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-6xl space-y-6 px-6 py-10">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Scanner</h1>
          <p className="text-sm text-[var(--color-ink-muted)]">
            Transparent, basic heuristics over stored daily candles. Results describe a
            configured screen — nothing here is investment advice.
          </p>
        </div>
        <button
          onClick={onRun}
          disabled={running}
          className="rounded-md bg-[var(--color-paper)] px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {running ? "Scanning…" : "Run scan"}
        </button>
      </header>

      {error && (
        <div className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {results.length === 0 ? (
        <div className="rounded-lg border border-[var(--color-border-subtle)] px-4 py-10 text-center text-sm text-[var(--color-ink-muted)]">
          No results yet. Sync a broker catalogue, ingest daily candles, then run a scan.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-[var(--color-border-subtle)]">
          <table className="w-full text-sm">
            <thead className="text-left text-[var(--color-ink-muted)]">
              <tr className="border-b border-[var(--color-border-subtle)]">
                <th className="px-4 py-2 font-medium">Stock</th>
                <th className="px-4 py-2 font-medium">Exchange</th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="The score that leads under the current configuration (value, momentum, or a blend)"
                >
                  Score
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Valuation lens: higher = cheaper / more pulled back"
                >
                  Value
                </th>
                <th className="px-4 py-2 text-right font-medium" title="Momentum core score">
                  Mom.
                </th>
                <th className="px-4 py-2 text-right font-medium">Risk</th>
                <th className="px-4 py-2 text-right font-medium">Liq.</th>
                <th className="px-4 py-2 font-medium">Tradable</th>
                <th className="px-4 py-2" aria-label="Expand" />
              </tr>
            </thead>
            <tbody>
              {results.map((r) => {
                const isOpen = expandedId === r.id;
                return (
                  <FragmentRow
                    key={r.id}
                    result={r}
                    isOpen={isOpen}
                    detail={details[r.id]}
                    onToggle={() => toggleRow(r.id)}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <footer className="pt-2 text-xs text-[var(--color-ink-muted)]">
        Screening results describe configured criteria only. No result implies an instrument is
        a good investment.
      </footer>
    </main>
  );
}

function FragmentRow({
  result: r,
  isOpen,
  detail,
  onToggle,
}: {
  result: ScannerResult;
  isOpen: boolean;
  detail: ScannerResultDetail | undefined;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-muted)]"
        aria-expanded={isOpen}
      >
        <td className="px-4 py-2 font-medium">{r.instrument_name ?? "—"}</td>
        <td className="px-4 py-2 text-[var(--color-ink-muted)]">
          {r.exchange_name ?? "—"}
          {r.exchange_mic ? ` (${r.exchange_mic})` : ""}
        </td>
        <td
          className="tabular px-4 py-2 text-right font-semibold"
          style={{ color: scoreColour(r.classification) }}
          title={CLASSIFICATION_LABEL[r.classification] ?? r.classification}
        >
          {Number(r.primary_score).toFixed(1)}
        </td>
        <td className="tabular px-4 py-2 text-right">
          {r.value_score == null ? "—" : Number(r.value_score).toFixed(0)}
        </td>
        <td className="tabular px-4 py-2 text-right">{Number(r.core_score).toFixed(0)}</td>
        <td className="tabular px-4 py-2 text-right">{Number(r.risk_score).toFixed(0)}</td>
        <td className="tabular px-4 py-2 text-right">{Number(r.liquidity_score).toFixed(0)}</td>
        <td className="px-4 py-2 text-xs">
          {r.is_trading212_tradable ? (
            <span className="text-[var(--color-ok)]">Tradable</span>
          ) : (
            <span className="text-[var(--color-ink-muted)]">Scanner only</span>
          )}
        </td>
        <td className="px-4 py-2 text-right text-[var(--color-ink-muted)]">{isOpen ? "▲" : "▼"}</td>
      </tr>
      {isOpen && (
        <tr className="border-b border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)]">
          <td colSpan={COLUMN_COUNT} className="px-4 py-4">
            {detail ? (
              <ExpandedDetail detail={detail} />
            ) : (
              <p className="text-sm text-[var(--color-ink-muted)]">Loading details…</p>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function ExpandedDetail({ detail }: { detail: ScannerResultDetail }) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-6 text-sm">
        <span>
          <span className="text-[var(--color-ink-muted)]">Classification: </span>
          <span style={{ color: scoreColour(detail.classification) }}>
            {CLASSIFICATION_LABEL[detail.classification] ?? detail.classification}
          </span>
        </span>
        <span>
          <span className="text-[var(--color-ink-muted)]">Momentum </span>
          {Number(detail.core_score).toFixed(0)}
          <span className="text-[var(--color-ink-muted)]"> · Value </span>
          {detail.value_score == null ? "—" : Number(detail.value_score).toFixed(0)}
          <span className="text-[var(--color-ink-muted)]"> · confidence </span>
          {(Number(detail.confidence) * 100).toFixed(0)}%
        </span>
        <span className="text-[var(--color-ink-muted)]">{detail.candles_used} candles</span>
      </div>

      <p className="text-xs text-[var(--color-ink-muted)]">
        Momentum rewards strength (uptrend, near highs). Value rewards cheapness (pulled back,
        oversold). Two separate lenses — a high value score means potentially undervalued, not a
        recommendation to buy.
      </p>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SignalList title="Momentum — positive" items={detail.positive_signals} tone="ok" />
        <SignalList title="Momentum — negative" items={detail.negative_signals} tone="warn" />
        <SignalList title="Value — cheap signals" items={detail.value_positive_signals} tone="ok" />
        <SignalList
          title="Value — expensive signals"
          items={detail.value_negative_signals}
          tone="warn"
        />
      </div>
    </div>
  );
}

function SignalList({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[];
  tone: "ok" | "warn" | "muted";
}) {
  const colour =
    tone === "ok"
      ? "var(--color-ok)"
      : tone === "warn"
        ? "var(--color-warn)"
        : "var(--color-ink-muted)";
  return (
    <div>
      <h3 className="text-xs font-medium" style={{ color: colour }}>
        {title}
      </h3>
      {items.length === 0 ? (
        <p className="mt-1 text-xs text-[var(--color-ink-muted)]">None</p>
      ) : (
        <ul className="mt-1 space-y-1 text-xs">
          {items.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
