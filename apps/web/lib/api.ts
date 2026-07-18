/**
 * Typed API client.
 *
 * Two rules this module enforces on every caller:
 *
 *  1. `credentials: "include"` on every request. The session is an HttpOnly
 *     cookie, so it is never readable from JavaScript — which means it is also
 *     never *sent* unless we ask for it explicitly.
 *
 *  2. The CSRF token is echoed from its cookie into a header on every
 *     state-changing request (the double-submit check the API performs).
 *
 * Responses are validated with Zod rather than cast. A cast would let a schema
 * change reach the UI as `undefined` at render time; validation fails loudly at
 * the boundary instead.
 */

import { z } from "zod";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const CSRF_COOKIE = "trading_csrf";
const CSRF_HEADER = "X-CSRF-Token";

/**
 * Every request is bounded.
 *
 * `fetch` has no default timeout: a socket that is bound but never answers —
 * a wedged `uvicorn --reload`, a container mid-restart, a dropped VPN — leaves
 * the promise pending forever, and the UI sits on a spinner with no error and
 * no recovery. A *refused* connection fails instantly, so this only shows up in
 * the case that looks most like the app being broken.
 *
 * 15s is chosen to sit above a slow-but-real request (instrument sync walks the
 * whole broker catalogue) and well below a user's patience.
 */
const REQUEST_TIMEOUT_MS = 15_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The API did not respond in time, or was unreachable. */
export class ApiUnreachableError extends Error {
  constructor(
    message: string,
    readonly cause_?: unknown,
  ) {
    super(message);
    this.name = "ApiUnreachableError";
  }
}

function readCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(new RegExp(`(^|;\\s*)${name}=([^;]*)`));
  return match?.[2] ? decodeURIComponent(match[2]) : null;
}

async function request<T>(
  path: string,
  // Input is deliberately `unknown` rather than `T`: schemas using `.default()`
  // have an input type that differs from their output (the field is optional
  // going in, guaranteed coming out). Constraining both to `T` would reject
  // every such schema.
  schema: z.ZodType<T, z.ZodTypeDef, unknown>,
  init: RequestInit = {},
): Promise<T> {
  const method = init.method ?? "GET";
  const headers = new Headers(init.headers);

  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  // Safe methods are exempt from CSRF, matching the API's own rule.
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const token = readCookie(CSRF_COOKIE);
    if (token) headers.set(CSRF_HEADER, token);
  }

  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      method,
      headers,
      credentials: "include",
      // Bounds the wait. Without this a hung socket never settles the promise.
      signal: init.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    // A network-level failure, not an HTTP error: the API is unreachable,
    // hung, or refused the connection. Distinguished from ApiError so callers
    // can say "the API is down" rather than "the request failed".
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new ApiUnreachableError(
        `The API did not respond within ${REQUEST_TIMEOUT_MS / 1000}s (${path}). ` +
          `It may be starting, restarting, or wedged.`,
        err,
      );
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiUnreachableError(`Request to ${path} was cancelled`, err);
    }
    throw new ApiUnreachableError(
      `Could not reach the API at ${API_URL}. Is it running?`,
      err,
    );
  }

  if (response.status === 204) {
    return schema.parse(undefined);
  }

  const text = await response.text();
  let payload: unknown = undefined;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError(`Malformed response from ${path}`, response.status, text);
    }
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : payload;
    throw new ApiError(
      typeof detail === "string" ? detail : `Request to ${path} failed`,
      response.status,
      detail,
    );
  }

  return schema.parse(payload);
}

/* -- Schemas ---------------------------------------------------------------
 * Monetary values arrive as strings, deliberately: JSON numbers are IEEE
 * doubles and would reintroduce the rounding drift the API's Decimal columns
 * exist to prevent. They are parsed for display only, never for arithmetic
 * that feeds an order.
 */

export const userSchema = z.object({
  id: z.string(),
  email: z.string(),
  display_name: z.string().nullable(),
  is_admin: z.boolean(),
  last_login_at: z.string().nullable(),
});

export const sessionSchema = z.object({
  user: userSchema,
  expires_at: z.string(),
  csrf_token: z.string(),
  is_recently_reauthenticated: z.boolean(),
});

export const healthSchema = z.object({
  status: z.string(),
  version: z.string(),
  environment: z.string(),
  live_trading_enabled: z.boolean(),
  components: z
    .array(
      z.object({
        name: z.string(),
        healthy: z.boolean(),
        detail: z.string().nullable().optional(),
        latency_ms: z.number().nullable().optional(),
      }),
    )
    .default([]),
});

export const accountSchema = z.object({
  broker: z.string(),
  is_live: z.boolean(),
  account_id: z.string(),
  currency: z.string(),
  cash: z.string(),
  total: z.string(),
  free_for_trading: z.string(),
  invested: z.string().nullable().optional(),
  result: z.string().nullable().optional(),
  retrieved_at: z.string().nullable().optional(),
  is_stale: z.boolean().default(false),
  age_seconds: z.number().default(0),
});

export const positionSchema = z.object({
  broker_ticker: z.string(),
  quantity: z.string(),
  average_price: z.string(),
  current_price: z.string().nullable(),
  unrealised_pnl: z.string().nullable(),
  currency: z.string().nullable().optional(),
});

export const instrumentSchema = z.object({
  id: z.string(),
  isin: z.string().nullable(),
  exchange_ticker: z.string().nullable(),
  name: z.string(),
  kind: z.string(),
  currency: z.string(),
  price_unit: z.string(),
  lifecycle_state: z.string(),
  lifecycle_note: z.string().nullable(),
  identity_confirmed_by_user: z.boolean(),
  is_bot_universe: z.boolean(),
  is_scanner_eligible: z.boolean(),
  suspended_at: z.string().nullable(),
  last_scanned_at: z.string().nullable(),
  exchange: z
    .object({
      mic: z.string(),
      name: z.string(),
      country: z.string().nullable(),
      timezone: z.string(),
    })
    .nullable()
    .optional(),
});

export const instrumentListSchema = z.object({
  items: z.array(instrumentSchema),
  total: z.number(),
  limit: z.number(),
  offset: z.number(),
});

export const syncResultSchema = z.object({
  broker: z.string(),
  synced_at: z.string(),
  total_from_broker: z.number(),
  broker_instruments_created: z.number(),
  broker_instruments_updated: z.number(),
  instruments_created: z.number(),
  instruments_needing_confirmation: z.number(),
  delisted: z.number(),
  errors: z.array(z.string()).default([]),
});

export const scannerResultSchema = z.object({
  id: z.string(),
  instrument_id: z.string(),
  instrument_name: z.string().nullable(),
  exchange_name: z.string().nullable(),
  exchange_mic: z.string().nullable(),
  primary_score: z.string(),
  core_score: z.string(),
  trend_score: z.string(),
  momentum_score: z.string(),
  risk_score: z.string(),
  liquidity_score: z.string(),
  positioning_score: z.string(),
  fundamental_score: z.string().nullable(),
  value_score: z.string().nullable(),
  price_value_score: z.string().nullable(),
  fundamental_value_score: z.string().nullable(),
  classification: z.string(),
  data_completeness: z.string(),
  data_freshness_days: z.string().nullable(),
  confidence: z.string(),
  candles_used: z.number(),
  is_trading212_tradable: z.boolean(),
});

export const scannerResultDetailSchema = scannerResultSchema.extend({
  instrument_name: z.string().nullable(),
  positive_signals: z.array(z.string()).default([]),
  negative_signals: z.array(z.string()).default([]),
  missing_information: z.array(z.string()).default([]),
  value_positive_signals: z.array(z.string()).default([]),
  value_negative_signals: z.array(z.string()).default([]),
  metrics: z.record(z.string(), z.unknown()).default({}),
});

export const scannerRunSchema = z.object({
  id: z.string(),
  status: z.string(),
  started_at: z.string(),
  completed_at: z.string().nullable(),
  instruments_considered: z.number(),
  instruments_scored: z.number(),
  instruments_skipped: z.number(),
  screening_candidates: z.number(),
  watchlist_candidates: z.number(),
  selection_reason: z.string().nullable(),
});

export const auditEventSchema = z.object({
  id: z.string(),
  sequence: z.number(),
  occurred_at: z.string(),
  kind: z.string(),
  actor_kind: z.string(),
  actor_label: z.string().nullable(),
  subject_type: z.string().nullable(),
  subject_id: z.string().nullable(),
  summary: z.string(),
  request_id: z.string().nullable(),
});

export const auditListSchema = z.object({
  items: z.array(auditEventSchema),
  total: z.number(),
});

export const liveStatusSchema = z.object({
  live_trading_enabled_on_server: z.boolean(),
  is_armed: z.boolean(),
  armed_at: z.string().nullable().optional(),
  expires_at: z.string().nullable().optional(),
  max_live_capital: z.string().nullable().optional(),
  max_daily_loss: z.string().nullable().optional(),
  blockers: z.array(z.string()).default([]),
});

export type User = z.infer<typeof userSchema>;
export type Session = z.infer<typeof sessionSchema>;
export type Health = z.infer<typeof healthSchema>;
export type Account = z.infer<typeof accountSchema>;
export type Position = z.infer<typeof positionSchema>;
export type Instrument = z.infer<typeof instrumentSchema>;
export type SyncResult = z.infer<typeof syncResultSchema>;
export type AuditEvent = z.infer<typeof auditEventSchema>;
export type LiveStatus = z.infer<typeof liveStatusSchema>;
export type ScannerResult = z.infer<typeof scannerResultSchema>;
export type ScannerResultDetail = z.infer<typeof scannerResultDetailSchema>;
export type ScannerRun = z.infer<typeof scannerRunSchema>;

/* -- Endpoints ------------------------------------------------------------ */

export const api = {
  health: () => request("/health", healthSchema),

  login: (email: string, password: string) =>
    request("/auth/login", sessionSchema, {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request("/auth/logout", z.undefined(), { method: "POST" }),

  me: () => request("/auth/me", sessionSchema),

  account: () => request("/account", accountSchema),

  positions: () => request("/positions", z.array(positionSchema)),

  instruments: (params: { search?: string; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.search) query.set("search", params.search);
    query.set("limit", String(params.limit ?? 25));
    query.set("offset", String(params.offset ?? 0));
    return request(`/instruments?${query}`, instrumentListSchema);
  },

  syncInstruments: () => request("/instruments/sync", syncResultSchema, { method: "POST" }),

  audit: (limit = 20) => request(`/audit?limit=${limit}`, auditListSchema),

  liveStatus: () => request("/live/status", liveStatusSchema),

  disarmLive: () => request("/live/disarm", liveStatusSchema, { method: "POST" }),

  scannerResults: (params: { minScore?: number; tradableOnly?: boolean; limit?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.minScore != null) query.set("min_score", String(params.minScore));
    if (params.tradableOnly) query.set("tradable_only", "true");
    query.set("limit", String(params.limit ?? 50));
    return request(`/scanner/results?${query}`, z.array(scannerResultSchema));
  },

  scannerResult: (id: string) =>
    request(`/scanner/results/${id}`, scannerResultDetailSchema),

  runScanner: (body: { instrument_ids?: string[]; limit?: number } = {}) =>
    request("/scanner/run", scannerRunSchema, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

/** Format a decimal string for display. Never used for arithmetic. */
export function formatMoney(value: string | null | undefined, currency: string): string {
  if (value == null) return "—";
  const asNumber = Number(value);
  if (Number.isNaN(asNumber)) return value;
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency,
    minimumFractionDigits: 2,
  }).format(asNumber);
}
