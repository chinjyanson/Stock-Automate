"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";

/**
 * Transient toast notifications for the outcome of user actions.
 *
 * The split is deliberate: page-LOAD failures stay as inline banners (they are
 * persistent, page-level state a user needs to keep seeing), while the result of
 * an *action* the user just took — save, toggle, run, trade — is a transient
 * toast that confirms success or surfaces failure without stealing the page.
 *
 * Success and error are announced to assistive tech (`status` / `alert`), so the
 * outcome is never conveyed by colour alone.
 */

type Variant = "success" | "error" | "info";

type ToastItem = { id: number; message: string; variant: Variant };

type ToastApi = {
  success: (message: string) => void;
  error: (message: string) => void;
  info: (message: string) => void;
};

const ToastContext = createContext<ToastApi | null>(null);

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within a <ToastProvider>");
  }
  return ctx;
}

//: Success/info clear quickly; errors linger so a failure is not missed.
const DISMISS_MS: Record<Variant, number> = {
  success: 4500,
  info: 4500,
  error: 8000,
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (message: string, variant: Variant) => {
      const id = nextId.current++;
      setToasts((prev) => [...prev, { id, message, variant }]);
      setTimeout(() => dismiss(id), DISMISS_MS[variant]);
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(
    () => ({
      success: (m) => push(m, "success"),
      error: (m) => push(m, "error"),
      info: (m) => push(m, "info"),
    }),
    [push],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <ToastViewport toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}

function ToastViewport({
  toasts,
  onDismiss,
}: {
  toasts: ToastItem[];
  onDismiss: (id: number) => void;
}) {
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex flex-col items-center gap-2 p-4 sm:items-end">
      {toasts.map((t) => (
        <ToastCard key={t.id} toast={t} onDismiss={() => onDismiss(t.id)} />
      ))}
    </div>
  );
}

const ACCENT: Record<Variant, string> = {
  success: "var(--color-ok)",
  error: "var(--color-live)",
  info: "var(--color-paper)",
};

const ICON: Record<Variant, string> = {
  success: "✓",
  error: "✕",
  info: "ℹ",
};

function ToastCard({ toast, onDismiss }: { toast: ToastItem; onDismiss: () => void }) {
  const accent = ACCENT[toast.variant];
  return (
    <div
      role={toast.variant === "error" ? "alert" : "status"}
      aria-live={toast.variant === "error" ? "assertive" : "polite"}
      className="pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface)] px-4 py-3 text-sm shadow-lg"
      style={{ borderInlineStartWidth: 4, borderInlineStartColor: accent }}
    >
      <span aria-hidden className="mt-0.5 font-semibold" style={{ color: accent }}>
        {ICON[toast.variant]}
      </span>
      <span className="min-w-0 flex-1 break-words text-[var(--color-ink)]">{toast.message}</span>
      <button
        onClick={onDismiss}
        aria-label="Dismiss notification"
        className="-mr-1 shrink-0 rounded px-1 text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
      >
        ✕
      </button>
    </div>
  );
}
