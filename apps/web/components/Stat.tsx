/**
 * A single labelled stat tile. Shared by the dashboard and settings so the
 * account figures render identically wherever they appear.
 */
export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-muted)] px-4 py-3">
      <p className="text-xs text-[var(--color-ink-muted)]">{label}</p>
      <p className="tabular mt-1 text-lg font-semibold">{value}</p>
    </div>
  );
}
