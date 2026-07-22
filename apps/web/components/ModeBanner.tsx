import type { Account, LiveStatus } from "@/lib/api";

/**
 * The single most important element on the dashboard: which mode am I in?
 *
 * Colour alone is not sufficient — it fails for colour-blind users and in
 * screenshots — so the mode is always stated in words, and the live state is
 * additionally announced to assistive technology. Getting this wrong means a
 * user believes they are paper trading while placing real orders.
 */
export function ModeBanner({
  account,
  liveStatus,
}: {
  account: Account | null;
  liveStatus: LiveStatus | null;
}) {
  const isLive = account?.is_live === true || liveStatus?.live_mode === true;

  if (isLive) {
    return (
      <div
        role="alert"
        aria-live="assertive"
        className="rounded-lg border-2 border-[var(--color-live)] bg-[var(--color-live-soft)] px-4 py-3"
      >
        <p className="font-semibold text-[var(--color-live)]">
          LIVE — orders place real money
          {liveStatus?.autonomous_enabled_on_server
            ? " (autonomous enabled: strategies can trade with no per-order approval)"
            : ""}
        </p>
        <p className="mt-1 text-sm text-[var(--color-ink)]">
          Switch back to paper any time in Settings.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-[var(--color-paper)] bg-[var(--color-paper-soft)] px-4 py-3">
      <p className="font-semibold text-[var(--color-paper)]">
        Paper trading — no real money is at risk
      </p>
      <p className="mt-1 text-sm text-[var(--color-ink)]">
        {account
          ? `Trading the ${account.broker.replace(/_/g, " ")} account.`
          : "No broker connected."}{" "}
        Live is off by default.
      </p>
    </div>
  );
}
