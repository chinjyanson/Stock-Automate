"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ApiError,
  ApiUnreachableError,
  type Instrument,
  type ScannerResult,
  type ScannerResultDetail,
  type WatchlistEntry,
  api,
} from "@/lib/api";
import { useToast } from "@/components/Toast";

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
const COLUMN_COUNT = 13;

// Below this many days old, a score is treated as current. The rotation
// guarantees the top-ranked band is rescanned nightly, so anything materially
// older is a name the sweep has not reached recently — worth showing plainly
// rather than letting it pass as today's reading.
const STALE_SCORE_DAYS = 3;

export default function ScannerPage() {
  const router = useRouter();
  const toast = useToast();
  const [results, setResults] = useState<ScannerResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Which row is expanded, and the detail for it (fetched on expand, cached).
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, ScannerResultDetail>>({});

  // Watchlist membership, keyed by instrument id. Held as a Set because the
  // table asks "is this one pinned?" once per row on every render.
  const [watchlist, setWatchlist] = useState<WatchlistEntry[]>([]);
  const [watchlistError, setWatchlistError] = useState<string | null>(null);
  const watchedIds = useMemo(
    () => new Set(watchlist.map((w) => w.instrument_id)),
    [watchlist],
  );
  // Instruments with a request in flight, so a row cannot be double-toggled.
  const [pendingWatch, setPendingWatch] = useState<Set<string>>(new Set());
  // Instruments currently being fetched and rescored after being pinned from
  // the search box. Separate from `pendingWatch`: the pin has already landed,
  // and what is outstanding is the market-data round trip.
  const [scanningIds, setScanningIds] = useState<Set<string>>(new Set());

  /**
   * Pinned first, then by score.
   *
   * Sorted here rather than in the API because watchlist membership is per-user
   * while `/scanner/results` ranks a run's output — the ordering is a property
   * of who is looking, not of the scan. Within each group the API's own
   * primary-score ranking is preserved, so a pinned stock keeps its position
   * relative to other pinned stocks.
   */
  const orderedResults = useMemo(() => {
    const rank = (r: ScannerResult) => (watchedIds.has(r.instrument_id) ? 0 : 1);
    return [...results].sort(
      (a, b) => rank(a) - rank(b) || Number(b.primary_score) - Number(a.primary_score),
    );
  }, [results, watchedIds]);

  // Which pins the table is actually showing, so the rest can be listed once
  // above it rather than silently missing.
  const watchedInResults = useMemo(
    () =>
      new Set(
        results.map((r) => r.instrument_id).filter((id) => watchedIds.has(id)),
      ),
    [results, watchedIds],
  );
  const pinnedShown = watchedInResults.size;

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
      // Settled, not all: a failed watchlist read must not blank the results
      // table. But it must not masquerade as an empty watchlist either — that
      // renders "nothing is pinned", which is a claim about what the next scan
      // will do, and it would be wrong. Failure keeps the last known list and
      // says so.
      const [resultsR, watchlistR] = await Promise.allSettled([
        api.scannerResults({ limit: 1000 }),
        api.scannerWatchlist(),
      ]);
      if (resultsR.status === "fulfilled") {
        setResults(resultsR.value);
      } else {
        throw resultsR.reason;
      }
      if (watchlistR.status === "fulfilled") {
        setWatchlist(watchlistR.value);
        setWatchlistError(null);
      } else {
        setWatchlistError(
          watchlistR.reason instanceof ApiError ||
            watchlistR.reason instanceof ApiUnreachableError
            ? watchlistR.reason.message
            : "Could not load the watchlist",
        );
      }
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

  /**
   * Pin or unpin one instrument.
   *
   * Server-confirmed rather than optimistic: pinning is a claim about what the
   * next scan will do, so showing a filled star before the write lands would
   * assert a guarantee that may not exist. `pendingWatch` blocks the double
   * click that the resulting latency invites.
   */
  async function toggleWatch(instrumentId: string, name: string | null, watched: boolean) {
    if (pendingWatch.has(instrumentId)) return;
    setPendingWatch((prev) => new Set(prev).add(instrumentId));
    try {
      if (watched) {
        await api.removeFromWatchlist(instrumentId);
        setWatchlist((prev) => prev.filter((w) => w.instrument_id !== instrumentId));
        toast.success(`${name ?? "Instrument"} removed from the watchlist.`);
      } else {
        const entry = await api.addToWatchlist(instrumentId);
        setWatchlist((prev) => [...prev, entry]);
        toast.success(
          entry.is_scannable
            ? `${entry.instrument_name ?? "Instrument"} is pinned — the next scan will cover it.`
            : `${entry.instrument_name ?? "Instrument"} is pinned, but has too little stored ` +
                `daily history to score yet. Ingest more candles for it first.`,
        );
      }
    } catch (err) {
      toast.error(
        err instanceof ApiError || err instanceof ApiUnreachableError
          ? err.message
          : "Could not update the watchlist",
      );
    } finally {
      setPendingWatch((prev) => {
        const next = new Set(prev);
        next.delete(instrumentId);
        return next;
      });
    }
  }

  /**
   * Pin an instrument found through the search box, then score it right away.
   *
   * The two steps are separate calls on purpose. Pinning is instant and is what
   * the operator asked for; fetching candles goes out to a market-data provider
   * and can take seconds. Binding them into one request would make the pin as
   * slow as the slowest provider, and a provider failure would leave it unclear
   * whether the stock had been pinned at all.
   *
   * The rescan matters because the rotation reaches most of the catalogue only
   * over days. Without it a newly pinned stock shows either nothing or whatever
   * it scored weeks ago — at exactly the moment someone is looking at it.
   */
  async function pinAndScan(instrument: Instrument) {
    const id = instrument.id;
    if (pendingWatch.has(id) || scanningIds.has(id)) return;
    setPendingWatch((prev) => new Set(prev).add(id));
    try {
      const entry = await api.addToWatchlist(id);
      setWatchlist((prev) =>
        prev.some((w) => w.instrument_id === id) ? prev : [...prev, entry],
      );
    } catch (err) {
      toast.error(
        err instanceof ApiError || err instanceof ApiUnreachableError
          ? err.message
          : "Could not add to the watchlist",
      );
      return;
    } finally {
      setPendingWatch((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }

    setScanningIds((prev) => new Set(prev).add(id));
    try {
      const refreshed = await api.refreshInstrument(id);
      if (refreshed.result) {
        const scored = refreshed.result;
        setResults((prev) => [
          scored,
          ...prev.filter((r) => r.instrument_id !== id),
        ]);
        toast.success(
          `${instrument.name} pinned and scored ${Number(scored.primary_score).toFixed(1)} ` +
            `on ${refreshed.candles_written} fresh bar(s).`,
        );
      } else {
        // Pinned, but not scoreable. Say which, rather than leave a silent row.
        toast.error(
          refreshed.reason ??
            `${instrument.name} is pinned but could not be scored.`,
        );
      }
    } catch (err) {
      toast.error(
        `${instrument.name} is pinned, but the rescan failed: ` +
          (err instanceof ApiError || err instanceof ApiUnreachableError
            ? err.message
            : "market data could not be fetched") +
          ". It will be scored by the next scan.",
      );
    } finally {
      setScanningIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  useEffect(() => {
    void load();
  }, [load]);

  async function onRun() {
    setRunning(true);
    setError(null);
    try {
      const run = await api.runScanner({});
      toast.success(
        `Scan complete — ${run.instruments_scored} scored, ` +
          `${run.screening_candidates} screening / ${run.watchlist_candidates} watchlist candidates.`,
      );
      setDetails({});
      setExpandedId(null);
      await load();
    } catch (err) {
      toast.error(
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

      <WatchlistSearch
        onPin={pinAndScan}
        watchedIds={watchedIds}
        busyIds={new Set([...pendingWatch, ...scanningIds])}
      />

      {scanningIds.size > 0 && (
        <p className="text-xs text-[var(--color-ink-muted)]">
          Fetching market data and scoring {scanningIds.size} newly pinned
          instrument{scanningIds.size === 1 ? "" : "s"}…
        </p>
      )}

      {watchlistError && (
        <div className="rounded-lg border border-[var(--color-warn)] bg-[var(--color-surface-muted)] px-4 py-3 text-sm">
          Could not load the watchlist: {watchlistError}. Pinned instruments are still
          selected by the scan — this is a display failure, not a lost watchlist.
        </div>
      )}

      {/*
        Pins live in the table itself, sorted to the top. The only ones that
        need saying out loud are those the table cannot show — an instrument
        with too little history to score never appears in a result set, so
        without this line it would be both invisible and impossible to unpin.
      */}
      {watchlist.length > pinnedShown && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--color-ink-muted)]">
          <span>Pinned but not in these results:</span>
          {watchlist
            .filter((w) => !watchedInResults.has(w.instrument_id))
            .map((w) => (
              <span
                key={w.instrument_id}
                className="flex items-center gap-1 rounded-full border border-[var(--color-border-subtle)] px-2 py-0.5"
              >
                {w.instrument_name ?? w.instrument_id}
                {!w.is_scannable && (
                  <span
                    title="Too little stored daily history to score — a scan cannot cover this yet"
                    style={{ color: "var(--color-warn)" }}
                  >
                    (no history)
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => toggleWatch(w.instrument_id, w.instrument_name, true)}
                  disabled={pendingWatch.has(w.instrument_id)}
                  aria-label={`Remove ${w.instrument_name ?? "instrument"} from the watchlist`}
                  className="disabled:opacity-40"
                >
                  ✕
                </button>
              </span>
            ))}
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
                <th
                  className="px-3 py-2 font-medium"
                  title="Pin an instrument so the next scan covers it before the rotating sweep"
                >
                  Watch
                </th>
                <th className="px-4 py-2 font-medium">Stock</th>
                <th className="px-4 py-2 font-medium">Exchange</th>
                <th className="px-4 py-2 font-medium" title="Sector (yfinance / GICS)">
                  Sector
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Final fundamentals-first score (0-100): intrinsic value + P/E lead, with cheapness, reversal, quality and sector in support"
                >
                  Score
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Intrinsic value / P/E: Graham margin-of-safety, earnings yield, P/B, PEG (heaviest factor)"
                >
                  Intr.
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Price cheapness: pulled back / oversold vs its own history"
                >
                  Cheap
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Reversal: is it turning up from a decline?"
                >
                  Rev.
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Quality / soundness: margins, growth, low debt, low volatility, liquidity"
                >
                  Qual.
                </th>
                <th
                  className="px-4 py-2 text-right font-medium"
                  title="Sector health: is the stock's own industry (its sector ETF) strengthening?"
                >
                  Sec.
                </th>
                <th className="px-4 py-2 font-medium">Tradable</th>
                <th
                  className="px-4 py-2 font-medium"
                  title="When this score was computed. The scan rotates through the catalogue, so rows are not all from the same day."
                >
                  Scored
                </th>
                <th className="px-4 py-2" aria-label="Expand" />
              </tr>
            </thead>
            <tbody>
              {orderedResults.map((r, i) => {
                const isOpen = expandedId === r.id;
                return (
                  <FragmentRow
                    key={r.id}
                    result={r}
                    isOpen={isOpen}
                    detail={details[r.id]}
                    onToggle={() => toggleRow(r.id)}
                    // Marks where the pinned block ends and score ranking
                    // resumes. Without it the score column appears to descend,
                    // then jump back up — which reads as a broken sort.
                    startsScoreRanked={i === pinnedShown && pinnedShown > 0}
                    isWatched={watchedIds.has(r.instrument_id)}
                    watchPending={pendingWatch.has(r.instrument_id)}
                    onToggleWatch={() =>
                      toggleWatch(
                        r.instrument_id,
                        r.instrument_name,
                        watchedIds.has(r.instrument_id),
                      )
                    }
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
  isWatched,
  watchPending,
  onToggleWatch,
  startsScoreRanked,
}: {
  result: ScannerResult;
  isOpen: boolean;
  detail: ScannerResultDetail | undefined;
  onToggle: () => void;
  isWatched: boolean;
  watchPending: boolean;
  onToggleWatch: () => void;
  startsScoreRanked: boolean;
}) {
  return (
    <>
      {startsScoreRanked && (
        <tr>
          <td
            colSpan={COLUMN_COUNT}
            className="border-t-2 border-[var(--color-border-subtle)] px-4 pt-2 pb-1 text-xs text-[var(--color-ink-muted)]"
          >
            Watchlist above · ranked by score below
          </td>
        </tr>
      )}
      <tr
        onClick={onToggle}
        className="cursor-pointer border-b border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-muted)]"
        aria-expanded={isOpen}
      >
        <td className="px-3 py-2">
          <button
            type="button"
            // The row itself expands on click; the star must not do both.
            onClick={(e) => {
              e.stopPropagation();
              onToggleWatch();
            }}
            disabled={watchPending}
            aria-pressed={isWatched}
            title={
              isWatched
                ? "On the watchlist — guaranteed to be scanned next run. Click to remove."
                : "Add to the watchlist so the next scan covers it"
            }
            className="rounded px-1 text-base leading-none disabled:opacity-40"
            style={{ color: isWatched ? "var(--color-warn)" : "var(--color-ink-muted)" }}
          >
            {isWatched ? "★" : "☆"}
          </button>
        </td>
        <td className="px-4 py-2 font-medium">{r.instrument_name ?? "—"}</td>
        <td className="px-4 py-2 text-[var(--color-ink-muted)]">
          {r.exchange_name ?? "—"}
          {r.exchange_mic ? ` (${r.exchange_mic})` : ""}
        </td>
        <td className="px-4 py-2 text-xs text-[var(--color-ink-muted)]">{r.sector ?? "—"}</td>
        <td
          className="tabular px-4 py-2 text-right font-semibold"
          style={{ color: scoreColour(r.classification) }}
          title={CLASSIFICATION_LABEL[r.classification] ?? r.classification}
        >
          {Number(r.primary_score).toFixed(1)}
        </td>
        <td className="tabular px-4 py-2 text-right font-medium">
          {r.fundamental_value_score == null
            ? "—"
            : Number(r.fundamental_value_score).toFixed(0)}
        </td>
        <td className="tabular px-4 py-2 text-right">
          {r.price_value_score == null ? "—" : Number(r.price_value_score).toFixed(0)}
        </td>
        <td className="tabular px-4 py-2 text-right">
          {r.reversal_score == null ? "—" : Number(r.reversal_score).toFixed(0)}
        </td>
        <td className="tabular px-4 py-2 text-right">
          {r.quality_score == null ? "—" : Number(r.quality_score).toFixed(0)}
        </td>
        <td className="tabular px-4 py-2 text-right">
          {r.sector_score == null ? "—" : Number(r.sector_score).toFixed(0)}
        </td>
        <td className="px-4 py-2 text-xs">
          {r.is_trading212_tradable ? (
            <span className="text-[var(--color-ok)]">Tradable</span>
          ) : (
            <span className="text-[var(--color-ink-muted)]">Scanner only</span>
          )}
        </td>
        <td className="px-4 py-2 text-xs" title={new Date(r.scanned_at).toLocaleString()}>
          <ScoreAge scannedAt={r.scanned_at} />
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

/**
 * How old a score is, in words.
 *
 * The default listing shows each instrument's most recent result across runs,
 * which is the only way a rotating scan can present a whole catalogue — but it
 * means the rows are not all from the same day. Stating the age is what stops a
 * three-week-old score from reading as this morning's.
 */
function ScoreAge({ scannedAt }: { scannedAt: string }) {
  const scanned = new Date(scannedAt);
  const days = Math.floor((Date.now() - scanned.getTime()) / 86_400_000);
  const stale = days >= STALE_SCORE_DAYS;
  const label = days <= 0 ? "today" : days === 1 ? "yesterday" : `${days}d ago`;
  return (
    <span style={{ color: stale ? "var(--color-warn)" : "var(--color-ink-muted)" }}>
      {label}
    </span>
  );
}

/**
 * Find any stock in the catalogue by name or ticker and pin it.
 *
 * The star on a row can only pin what a scan already returned, which is a small
 * slice of the catalogue on any given day. This reaches the other ~15k.
 */
function WatchlistSearch({
  onPin,
  watchedIds,
  busyIds,
}: {
  onPin: (instrument: Instrument) => void | Promise<void>;
  watchedIds: Set<string>;
  busyIds: Set<string>;
}) {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Instrument[]>([]);
  const [searching, setSearching] = useState(false);
  const [failed, setFailed] = useState(false);

  // Debounced: a request per keystroke would search a 15k catalogue on every
  // letter, and the answers would arrive out of order.
  useEffect(() => {
    const term = query.trim();
    if (term.length < 2) {
      setMatches([]);
      setFailed(false);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      setSearching(true);
      try {
        const page = await api.instruments({ search: term, limit: 15 });
        if (!cancelled) {
          setMatches(page.items);
          setFailed(false);
        }
      } catch {
        // An empty dropdown would claim the catalogue holds no such stock,
        // which is a different statement from "the search did not run".
        if (!cancelled) {
          setMatches([]);
          setFailed(true);
        }
      } finally {
        if (!cancelled) setSearching(false);
      }
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  const open = query.trim().length >= 2;

  return (
    <div className="relative">
      <label htmlFor="watchlist-search" className="sr-only">
        Search for a stock to add to the watchlist
      </label>
      <input
        id="watchlist-search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Add a stock to the watchlist — search by name or ticker…"
        className="w-full rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface)] px-3 py-2 text-sm"
        autoComplete="off"
      />
      {open && (
        <div className="absolute z-20 mt-1 max-h-72 w-full overflow-auto rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-surface)] shadow-lg">
          {searching ? (
            <p className="px-3 py-2 text-sm text-[var(--color-ink-muted)]">Searching…</p>
          ) : failed ? (
            <p className="px-3 py-2 text-sm text-[var(--color-warn)]">
              The catalogue search failed. This does not mean there is no such stock —
              try again.
            </p>
          ) : matches.length === 0 ? (
            <p className="px-3 py-2 text-sm text-[var(--color-ink-muted)]">
              Nothing in the catalogue matches “{query.trim()}”.
            </p>
          ) : (
            matches.map((inst) => {
              const pinned = watchedIds.has(inst.id);
              const busy = busyIds.has(inst.id);
              return (
                <button
                  key={inst.id}
                  type="button"
                  disabled={pinned || busy}
                  onClick={() => {
                    void onPin(inst);
                    setQuery("");
                    setMatches([]);
                  }}
                  className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm hover:bg-[var(--color-surface-muted)] disabled:opacity-50"
                >
                  <span className="min-w-0 truncate">{inst.name}</span>
                  <span className="shrink-0 font-mono text-xs text-[var(--color-ink-muted)]">
                    {inst.exchange_ticker ?? "—"}
                    {inst.exchange?.mic ? ` · ${inst.exchange.mic}` : ""}
                    {pinned ? " · pinned" : busy ? " · adding…" : ""}
                  </span>
                </button>
              );
            })
          )}
        </div>
      )}
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
