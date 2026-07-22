import type { Position } from "@/lib/api";

/**
 * Open-positions table, shared so the dashboard renders positions one way.
 *
 * `nameFor` resolves a broker ticker to a human-readable instrument name. Paper
 * positions key on the canonical instrument id, which is not meaningful on its
 * own; pass a resolver to show names, omit it to fall back to the raw ticker.
 */
export function PositionsTable({
  positions,
  nameFor,
  emptyMessage = "No open positions.",
}: {
  positions: Position[];
  nameFor?: (ticker: string) => string;
  emptyMessage?: string;
}) {
  if (positions.length === 0) {
    return (
      <p className="px-4 py-6 text-sm text-[var(--color-ink-muted)]">{emptyMessage}</p>
    );
  }
  return (
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
            <tr key={p.broker_ticker} className="border-b border-[var(--color-border-subtle)]">
              <td className="px-4 py-2 font-medium">
                {nameFor ? nameFor(p.broker_ticker) : p.broker_ticker}
              </td>
              <td className="tabular px-4 py-2 text-right">{p.quantity}</td>
              <td className="tabular px-4 py-2 text-right">{p.average_price}</td>
              <td className="tabular px-4 py-2 text-right">{p.current_price ?? "—"}</td>
              <td className="tabular px-4 py-2 text-right">{p.unrealised_pnl ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
