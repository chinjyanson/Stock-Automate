"use client";

import { usePathname, useRouter } from "next/navigation";
import { api } from "@/lib/api";

const LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/scanner", label: "Scanner" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/strategies", label: "Strategies" },
  { href: "/settings", label: "Settings" },
];

// Routes with no chrome — the auth screens stand alone.
const BARE = ["/login", "/"];

/**
 * The single top navigation, shared by every signed-in page. Rendered once in the
 * root layout so pages carry only their own content, and the active tab is always
 * unambiguous.
 */
export function NavBar() {
  const pathname = usePathname();
  const router = useRouter();

  if (BARE.includes(pathname)) return null;

  async function onLogout() {
    try {
      await api.logout();
    } finally {
      router.push("/login");
    }
  }

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
        <button
          onClick={onLogout}
          className="rounded-md border border-[var(--color-border-subtle)] px-3 py-1.5 text-sm text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
        >
          Sign out
        </button>
      </div>
    </nav>
  );
}
