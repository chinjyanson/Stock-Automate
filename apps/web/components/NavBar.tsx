"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { type Account, type Health, type LiveStatus, api } from "@/lib/api";

const LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/scanner", label: "Scanner" },
  { href: "/strategies", label: "Strategies" },
  { href: "/settings", label: "Settings" },
];

// Routes with no chrome — the auth screens stand alone.
const BARE = ["/login", "/"];

/**
 * The single top navigation, shared by every signed-in page. Rendered once in the
 * root layout so pages carry only their own content, and the active tab is always
 * unambiguous.
 *
 * The paper/live mode pill lives here rather than on individual pages because it
 * is the single most important piece of state — "am I about to move real money?"
 * — and it must be visible on every page, not just the account view.
 */
export function NavBar() {
  const pathname = usePathname();
  const router = useRouter();
  const isBare = BARE.includes(pathname);

  const [account, setAccount] = useState<Account | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    if (isBare) return;
    let cancelled = false;
    // Best-effort: the nav must still render if any of these fail (e.g. a broker
    // outage), so each is swallowed independently.
    void api.activeAccount().then((a) => !cancelled && setAccount(a)).catch(() => {});
    void api.liveStatus().then((s) => !cancelled && setLiveStatus(s)).catch(() => {});
    void api.health().then((h) => !cancelled && setHealth(h)).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [isBare, pathname]);

  if (isBare) return null;

  async function onLogout() {
    try {
      await api.logout();
    } finally {
      router.push("/login");
    }
  }

  const isLive = account?.is_live === true || liveStatus?.live_mode === true;

  return (
    <nav className="sticky top-0 z-10 border-b border-[var(--color-border-subtle)] bg-[var(--color-surface)]/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        <div className="flex items-center gap-1">
          <span className="mr-4 font-semibold tracking-tight">Stock&nbsp;Automate</span>
          {LINKS.map((link) => {
            const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <a
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={
                  "rounded-md px-3 py-1.5 text-sm transition-colors " +
                  (active
                    ? "bg-[var(--color-surface-muted)] font-medium text-[var(--color-ink)]"
                    : "text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]")
                }
              >
                {link.label}
              </a>
            );
          })}
        </div>
        <div className="flex items-center gap-3">
          <ModePill isLive={isLive} health={health} />
          <button
            onClick={onLogout}
            className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
          >
            Sign out
          </button>
        </div>
      </div>
    </nav>
  );
}

/**
 * Compact venue + environment indicator. Mode is stated in words (never colour
 * alone) and announced to assistive tech, so a user can never mistake live for
 * paper. Live additionally uses `role="alert"` given the stakes.
 */
function ModePill({ isLive, health }: { isLive: boolean; health: Health | null }) {
  return (
    <div className="flex items-center gap-2">
      <span
        role={isLive ? "alert" : undefined}
        aria-live={isLive ? "assertive" : undefined}
        className={
          "rounded-md border px-2.5 py-1 text-xs font-semibold " +
          (isLive
            ? "border-[var(--color-live)] bg-[var(--color-live-soft)] text-[var(--color-live)]"
            : "border-[var(--color-paper)] bg-[var(--color-paper-soft)] text-[var(--color-paper)]")
        }
        title={
          isLive
            ? "Live — orders place real money. Switch in Settings."
            : "Paper — Trading 212 demo. No real money is at risk."
        }
      >
        {isLive ? "● LIVE — real money" : "● PAPER — demo"}
      </span>
      {health && (
        <span className="hidden text-xs text-[var(--color-ink-muted)] sm:inline">
          {health.environment} · v{health.version}
        </span>
      )}
    </div>
  );
}
