"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type CrashOverlayReading,
  type CrashOverlayStatus,
  api,
} from "@/lib/api";
import { Stat } from "@/components/Stat";

/**
 * The S&P 500 exposure view: should the index sleeve be fully invested today, or
 * standing aside?
 *
 * The headline is deliberately the *verdict* rather than the probability. A
 * number like 0.0023 tells nobody anything without the trigger beside it, and
 * the whole design of this model is that the trigger moves — it is a percentile
 * of the model's own recent output, not a fixed line. So the page leads with
 * "hold" or "step aside", and shows the two numbers underneath for anyone who
 * wants to check the arithmetic.
 *
 * **Advisory.** Nothing here places an order, and the page says so plainly
 * rather than leaving it ambiguous — a screen that shows a target exposure and
 * stays silent about whether anything acts on it is the kind of thing that gets
 * misread once, expensively.
 */
export default function SP500Page() {
  const router = useRouter();
  const [status, setStatus] = useState<CrashOverlayStatus | null>(null);
  const [history, setHistory] = useState<CrashOverlayReading[]>([]);
  const [loading, setLoading] = useState(true);
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
        setError(
          err instanceof ApiUnreachableError
            ? err.message
            : "Could not reach the API. Is it running on port 8000?",
        );
        return;
      }

      const [statusR, historyR] = await Promise.allSettled([
        api.crashOverlay(),
        api.crashOverlayHistory(120),
      ]);
      if (statusR.status === "fulfilled") setStatus(statusR.value);
      if (historyR.status === "fulfilled") setHistory(historyR.value);
      if (statusR.status === "rejected") {
        const r = statusR.reason;
        setError(
          r instanceof ApiError || r instanceof ApiUnreachableError
            ? r.message
            : "Could not load the S&P 500 reading.",
        );
      }
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  const latest = status?.latest ?? null;
  const exposure = latest?.target_exposure ? Number(latest.target_exposure) : null;
  const standingAside = exposure !== null && exposure < 1;

  return (
    <main className="mx-auto max-w-5xl px-6 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">S&amp;P 500 exposure</h1>
        <p className="mt-1 text-sm text-[var(--color-ink-muted)]">
          A model that watches for a sharp fall and steps aside when one looks imminent. It
          reads eleven market-wide stress measures — volatility, credit spreads, skew, breadth
          — and refits itself from the whole history every night.
        </p>
      </header>

      {error && (
        <div className="mb-6 rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-[var(--color-ink-muted)]">Loading…</p>
      ) : !latest ? (
        <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)] px-4 py-6 text-sm">
          <p className="font-medium">No reading yet.</p>
          <p className="mt-1 text-[var(--color-ink-muted)]">
            The nightly job has not run, or there is too little history to fit the model. It
            needs about two years of data before it can score a day at all, and two years of
            scores after that before it can judge whether one is unusual.
          </p>
        </div>
      ) : (
        <>
          <section
            className={`rounded-lg border px-5 py-5 ${
              standingAside
                ? "border-[var(--color-warn)] bg-[var(--color-surface-muted)]"
                : "border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)]"
            }`}
          >
            <p className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)]">
              Today&apos;s verdict — {latest.as_of}
            </p>
            <p className="mt-2 text-3xl font-semibold">
              {standingAside ? "Step aside" : "Stay invested"}
            </p>
            <p className="mt-2 text-sm text-[var(--color-ink-muted)]">{latest.reason}</p>
          </section>

          <section className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat
              label="Target exposure"
              value={exposure === null ? "—" : `${Math.round(exposure * 100)}%`}
            />
            <Stat
              label="Chance of a 2% fall"
              value={latest.probability ? `${(Number(latest.probability) * 100).toFixed(2)}%` : "—"}
            />
            <Stat
              label="Warns above"
              value={latest.trigger ? `${(Number(latest.trigger) * 100).toFixed(2)}%` : "—"}
            />
            <Stat
              label="Days of history"
              value={(status?.days_of_history ?? 0).toLocaleString()}
            />
          </section>

          <p className="mt-3 text-xs text-[var(--color-ink-muted)]">
            The warning level is not a fixed line: it is set so that roughly the most alarming
            10% of recent days trigger it. That is why a probability of a fraction of a percent
            can still be calm — what matters is where today sits against the last two years,
            not the number on its own.
          </p>

          <section className="mt-8">
            <h2 className="text-lg font-semibold">Recent days</h2>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[40rem] text-sm">
                <thead className="text-left text-xs uppercase tracking-wide text-[var(--color-ink-muted)]">
                  <tr className="border-b border-[var(--color-border-subtle)]">
                    <th className="py-2 pr-4 font-medium">Date</th>
                    <th className="py-2 pr-4 font-medium">S&amp;P close</th>
                    <th className="py-2 pr-4 font-medium">Fall risk</th>
                    <th className="py-2 pr-4 font-medium">Warns above</th>
                    <th className="py-2 pr-4 font-medium">Exposure</th>
                    <th className="py-2 font-medium">Verdict</th>
                  </tr>
                </thead>
                <tbody className="tabular">
                  {history.map((row) => {
                    const rowExposure = row.target_exposure ? Number(row.target_exposure) : null;
                    return (
                      <tr
                        key={row.as_of}
                        className="border-b border-[var(--color-border-subtle)] last:border-0"
                      >
                        <td className="py-2 pr-4">{row.as_of}</td>
                        <td className="py-2 pr-4">
                          {row.index_close ? Number(row.index_close).toFixed(2) : "—"}
                        </td>
                        <td className="py-2 pr-4">
                          {row.probability ? `${(Number(row.probability) * 100).toFixed(2)}%` : "—"}
                        </td>
                        <td className="py-2 pr-4">
                          {row.trigger ? `${(Number(row.trigger) * 100).toFixed(2)}%` : "—"}
                        </td>
                        <td className="py-2 pr-4">
                          {rowExposure === null ? "—" : `${Math.round(rowExposure * 100)}%`}
                        </td>
                        <td className="py-2">
                          {row.is_warning ? (
                            <span className="font-medium text-[var(--color-warn)]">Warning</span>
                          ) : (
                            <span className="text-[var(--color-ink-muted)]">Clear</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className="mt-8 rounded-lg border border-[var(--color-border-subtle)] px-4 py-4 text-sm">
            <p className="font-medium">This page does not trade.</p>
            <p className="mt-1 text-[var(--color-ink-muted)]">
              It records what the model thinks the exposure should be, and nothing acts on it.
              Worth knowing before you do: the model detects <em>slow, credit-driven</em>{" "}
              declines — it handled 2008 well and COVID much less well — and roughly seven in
              eight of its warnings are not followed by the fall it feared.
            </p>
          </section>
        </>
      )}
    </main>
  );
}
