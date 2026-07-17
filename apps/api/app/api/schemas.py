"""API request/response schemas.

These are the contract the TypeScript client is generated from (§1), so they
are explicit rather than inferred from ORM models: a response shape that drifts
because a column was renamed is a broken client.

Monetary values serialise as strings, not JSON numbers. JSON numbers are IEEE
doubles, and round-tripping a price through one reintroduces exactly the binary
drift the Decimal columns exist to avoid.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, PlainSerializer

#: Decimal that survives JSON as an exact string.
SerializedDecimal = Annotated[
    Decimal, PlainSerializer(lambda v: str(v), return_type=str, when_used="json")
]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# -- Auth ---------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class ReauthRequest(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


class UserResponse(ORMModel):
    id: uuid.UUID
    email: str
    display_name: str | None
    is_admin: bool
    last_login_at: datetime | None


class SessionResponse(BaseModel):
    user: UserResponse
    expires_at: datetime
    #: Mirrored into a cookie for the double-submit CSRF check.
    csrf_token: str
    is_recently_reauthenticated: bool


# -- Health -------------------------------------------------------------------


class ComponentHealth(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None
    latency_ms: int | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    #: Echoed so a browser can never be confused about which mode it is seeing.
    live_trading_enabled: bool
    components: list[ComponentHealth] = Field(default_factory=list)


# -- Account ------------------------------------------------------------------


class AccountResponse(BaseModel):
    broker: str
    is_live: bool
    #: Masked (§17). The full identifier is not needed in the browser.
    account_id: str
    currency: str
    cash: SerializedDecimal
    total: SerializedDecimal
    free_for_trading: SerializedDecimal
    invested: SerializedDecimal | None = None
    result: SerializedDecimal | None = None
    retrieved_at: datetime | None = None
    #: True when the broker was rate-limited or unreachable and this is the last
    #: known value rather than a fresh read. The UI should mark it as delayed.
    is_stale: bool = False
    #: Age of the served value in whole seconds (0 for a fresh read).
    age_seconds: int = 0


class PositionResponse(BaseModel):
    broker_ticker: str
    quantity: SerializedDecimal
    average_price: SerializedDecimal
    current_price: SerializedDecimal | None
    unrealised_pnl: SerializedDecimal | None
    currency: str | None = None


class OrderResponse(BaseModel):
    broker_order_id: str
    broker_ticker: str
    side: str
    order_type: str
    quantity: SerializedDecimal
    filled_quantity: SerializedDecimal
    status: str
    average_fill_price: SerializedDecimal | None = None
    created_at: datetime | None = None


class ReconciliationDiscrepancyResponse(BaseModel):
    kind: str
    broker_ticker: str | None
    detail: str
    local_value: str | None = None
    broker_value: str | None = None


class ReconciliationResponse(BaseModel):
    broker: str
    reconciled_at: datetime
    positions_checked: int
    orders_checked: int
    is_clean: bool
    discrepancies: list[ReconciliationDiscrepancyResponse] = Field(default_factory=list)


# -- Instruments --------------------------------------------------------------


class ExchangeResponse(ORMModel):
    mic: str
    name: str
    country: str | None
    timezone: str


class MarketDataMappingResponse(ORMModel):
    id: uuid.UUID
    provider: str
    provider_symbol: str
    price_unit: str | None
    is_signal_source: bool
    is_active: bool
    confirmed_by_user: bool
    resolution_method: str | None
    last_verified_at: datetime | None
    last_error: str | None


class BrokerInstrumentResponse(ORMModel):
    id: uuid.UUID
    broker: str
    broker_ticker: str
    broker_name: str | None
    currency: str | None
    is_currently_available: bool
    supports_fractional: bool
    last_synced_at: datetime | None


class InstrumentResponse(ORMModel):
    id: uuid.UUID
    isin: str | None
    exchange_ticker: str | None
    name: str
    kind: str
    currency: str
    price_unit: str
    lifecycle_state: str
    lifecycle_note: str | None
    identity_confirmed_by_user: bool
    is_bot_universe: bool
    is_scanner_eligible: bool
    suspended_at: datetime | None
    last_scanned_at: datetime | None
    exchange: ExchangeResponse | None = None


class InstrumentDetailResponse(InstrumentResponse):
    data_mappings: list[MarketDataMappingResponse] = Field(default_factory=list)
    broker_instruments: list[BrokerInstrumentResponse] = Field(default_factory=list)
    #: Candle coverage and freshness, so the UI can explain why an instrument
    #: is not yet tradable rather than just showing a disabled button.
    daily_candle_count: int = 0
    daily_last_timestamp: datetime | None = None
    has_sufficient_history: bool = False


class InstrumentListResponse(BaseModel):
    items: list[InstrumentResponse]
    total: int
    limit: int
    offset: int


class SyncInstrumentsResponse(BaseModel):
    broker: str
    synced_at: datetime
    total_from_broker: int
    broker_instruments_created: int
    broker_instruments_updated: int
    instruments_created: int
    instruments_needing_confirmation: int
    delisted: int
    errors: list[str] = Field(default_factory=list)


class MapInstrumentRequest(BaseModel):
    provider: str = "yfinance"
    #: Set when the automatic resolution is wrong and the user knows better.
    provider_symbol: str | None = None
    is_signal_source: bool = False


class MapInstrumentResponse(BaseModel):
    instrument_id: uuid.UUID
    provider: str
    resolved: bool
    requires_confirmation: bool
    mapping: MarketDataMappingResponse | None = None
    reason: str | None = None


class ConfirmMappingRequest(BaseModel):
    provider_symbol: str | None = None


class IngestRequest(BaseModel):
    provider: str = "yfinance"
    force_full_backfill: bool = False
    backfill_days: int = Field(default=730, ge=1, le=7300)


class IngestResponse(BaseModel):
    instrument_id: uuid.UUID
    interval: str
    provider: str | None
    candles_written: int
    was_backfill: bool
    window_start: datetime | None
    window_end: datetime | None
    skipped_reason: str | None
    errors: list[str] = Field(default_factory=list)


class CandleResponse(BaseModel):
    timestamp: datetime
    open: SerializedDecimal
    high: SerializedDecimal
    low: SerializedDecimal
    close: SerializedDecimal
    adjusted_close: SerializedDecimal | None
    volume: SerializedDecimal | None
    currency: str
    price_unit: str
    is_closed: bool
    quality_status: str
    provider: str


# -- Audit --------------------------------------------------------------------


class AuditEventResponse(ORMModel):
    id: uuid.UUID
    sequence: int
    occurred_at: datetime
    kind: str
    actor_kind: str
    actor_user_id: uuid.UUID | None
    actor_label: str | None
    subject_type: str | None
    subject_id: str | None
    summary: str
    payload: dict[str, Any] | None
    request_id: str | None


class AuditListResponse(BaseModel):
    items: list[AuditEventResponse]
    total: int


class AuditVerifyResponse(BaseModel):
    is_intact: bool
    events_checked: int
    problems: list[str] = Field(default_factory=list)


# -- Provider status ----------------------------------------------------------


class ProviderUsageResponse(BaseModel):
    provider: str
    usage_date: str
    requests_used: int
    requests_failed: int
    operational_limit: int
    emergency_reserve: int
    remaining: int


class ProviderHealthResponse(BaseModel):
    provider: str
    is_healthy: bool
    last_success_at: datetime | None
    last_failure_at: datetime | None
    consecutive_failures: int
    last_error: str | None


# -- Live arming --------------------------------------------------------------


class ArmLiveRequest(BaseModel):
    """Arming live trading (§14).

    `confirmation_phrase` must be typed exactly. A checkbox is too easy to click
    through for an action that begins risking real money.
    """

    confirmation_phrase: str
    max_live_capital: SerializedDecimal = Field(gt=0)
    max_daily_loss: SerializedDecimal = Field(gt=0)
    duration_minutes: int = Field(default=60, ge=1, le=1440)


class LiveStatusResponse(BaseModel):
    #: Server-side master switch (LIVE_TRADING_ENABLED).
    live_trading_enabled_on_server: bool
    #: Whether an unexpired arming session exists right now.
    is_armed: bool
    armed_at: datetime | None = None
    expires_at: datetime | None = None
    max_live_capital: SerializedDecimal | None = None
    max_daily_loss: SerializedDecimal | None = None
    #: Everything currently preventing live trading, for display.
    blockers: list[str] = Field(default_factory=list)
