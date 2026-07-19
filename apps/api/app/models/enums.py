"""Domain enumerations.

These are stored as strings rather than native PG enums so that adding a member
is a code change plus a data migration only where semantics actually change,
not a lock-taking ALTER TYPE on every deploy.
"""

from __future__ import annotations

from enum import StrEnum


class InstrumentKind(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    ETC = "etc"
    TRUST = "trust"
    UNKNOWN = "unknown"


class PriceUnit(StrEnum):
    """The unit a provider quotes in, which is not always the trading currency.

    LSE instruments are commonly quoted in pence (GBX) while the instrument's
    currency is GBP. Conflating the two overstates prices by 100x, so the unit
    is tracked explicitly on every candle and normalised at the adapter
    boundary (§4).
    """

    GBP = "GBP"
    GBX = "GBX"
    USD = "USD"
    EUR = "EUR"
    CHF = "CHF"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    PLN = "PLN"
    CZK = "CZK"
    HUF = "HUF"


class LifecycleState(StrEnum):
    """Bot Universe instrument lifecycle (§7).

    An instrument only becomes bot-tradable by walking this ladder. Presence in
    the broker catalogue confers nothing.
    """

    DISCOVERED = "discovered"
    MAPPING_REQUIRED = "mapping_required"
    DATA_BACKFILL = "data_backfill"
    VALIDATION_FAILED = "validation_failed"
    PAPER_ELIGIBLE = "paper_eligible"
    PAPER_ACTIVE = "paper_active"
    LIVE_ELIGIBLE = "live_eligible"
    LIVE_ACTIVE = "live_active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class OperatingMode(StrEnum):
    """Bot operating modes (§7). Default must never be a live mode."""

    DISABLED = "disabled"
    OBSERVE_ONLY = "observe_only"
    INTERNAL_PAPER = "internal_paper"
    BROKER_DEMO = "broker_demo"
    LIVE_APPROVAL_REQUIRED = "live_approval_required"
    LIVE_AUTONOMOUS = "live_autonomous"

    @property
    def is_live(self) -> bool:
        return self in {OperatingMode.LIVE_APPROVAL_REQUIRED, OperatingMode.LIVE_AUTONOMOUS}


class BrokerKind(StrEnum):
    TRADING212_DEMO = "trading212_demo"
    TRADING212_LIVE = "trading212_live"
    INTERNAL_PAPER = "internal_paper"
    MOCK = "mock"

    @property
    def is_live(self) -> bool:
        return self is BrokerKind.TRADING212_LIVE


class ProviderKind(StrEnum):
    YFINANCE = "yfinance"
    TWELVE_DATA = "twelve_data"
    EODHD = "eodhd"
    MOCK = "mock"


class Interval(StrEnum):
    """Candle intervals. Values match the canonical form stored on candles."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def is_intraday(self) -> bool:
        return self in {Interval.M1, Interval.M5, Interval.M15, Interval.H1, Interval.H4}


class DataSeriesType(StrEnum):
    """Distinguishes adjusted from raw series sharing an instrument+interval.

    Part of the candle uniqueness key (§4): a raw and a split-adjusted bar for
    the same timestamp are different facts, not a conflict to be deduplicated.
    """

    RAW = "raw"
    ADJUSTED = "adjusted"


class QualityStatus(StrEnum):
    OK = "ok"
    STALE = "stale"
    SUSPECT = "suspect"
    CONFLICTED = "conflicted"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"

    @property
    def is_tradable(self) -> bool:
        """Only pristine data may originate a signal. Fail closed (§17)."""
        return self is QualityStatus.OK


class DataQualityEventKind(StrEnum):
    MISSING_CANDLE = "missing_candle"
    STALE_CANDLE = "stale_candle"
    DUPLICATE_CANDLE = "duplicate_candle"
    PROVIDER_CONFLICT = "provider_conflict"
    SUSPICIOUS_MOVE = "suspicious_move"
    ZERO_VOLUME = "zero_volume"
    PARTIAL_SESSION = "partial_session"
    BACKFILL_GAP = "backfill_gap"
    UNIT_MISMATCH = "unit_mismatch"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }


class TradeIntentStatus(StrEnum):
    """Lifecycle of the local record that guards against duplicate orders (§10)."""

    CREATED = "created"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    RECONCILED = "reconciled"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    FAILED = "failed"
    ABANDONED = "abandoned"


class HaltKind(StrEnum):
    """Why trading is halted (§9).

    A halt is a recorded *state*, not a raised exception — see `RiskHalt`. Each
    member maps to one of the controls in `docs/risk-model.md`. The pre-trade
    engine consults active halts before sizing; the fail-closed members
    (`stale_data`, `provider_failure`, `reconciliation`) exist so uncertainty
    becomes an explicit, visible refusal rather than a silent skip.
    """

    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    OPEN_POSITIONS = "open_positions"
    TRADES_PER_DAY = "trades_per_day"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    STALE_DATA = "stale_data"
    PROVIDER_FAILURE = "provider_failure"
    RECONCILIATION = "reconciliation"
    KILL_SWITCH = "kill_switch"
    INSTRUMENT_SUSPENDED = "instrument_suspended"


class HaltScope(StrEnum):
    """How wide a halt reaches (§9).

    A `global` halt blocks every order; `instrument`/`strategy` halts are the
    surgical overrides that suspend one thing without stopping the rest.
    """

    GLOBAL = "global"
    INSTRUMENT = "instrument"
    STRATEGY = "strategy"


class AuditEventKind(StrEnum):
    """Audit events are immutable and append-only (§17)."""

    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    CREDENTIAL_CHANGED = "credential_changed"
    SETTING_CHANGED = "setting_changed"
    INSTRUMENT_SYNCED = "instrument_synced"
    INSTRUMENT_MAPPED = "instrument_mapped"
    INSTRUMENT_LIFECYCLE_CHANGED = "instrument_lifecycle_changed"
    INSTRUMENT_SUSPENDED = "instrument_suspended"
    SCANNER_RUN_STARTED = "scanner_run_started"
    SCANNER_RUN_COMPLETED = "scanner_run_completed"
    TRADE_PROPOSED = "trade_proposed"
    TRADE_APPROVED = "trade_approved"
    TRADE_REJECTED = "trade_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    RECONCILIATION_RESOLVED = "reconciliation_resolved"
    RISK_HALT_ACTIVATED = "risk_halt_activated"
    RISK_HALT_CLEARED = "risk_halt_cleared"
    LIVE_ARMED = "live_armed"
    LIVE_DISARMED = "live_disarmed"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    STOP_ADJUSTED = "stop_adjusted"
    POSITION_CLOSED = "position_closed"
    EOD_SUMMARY_GENERATED = "eod_summary_generated"


class ActorKind(StrEnum):
    """Who caused an audited action."""

    USER = "user"
    SYSTEM = "system"
    SCHEDULER = "scheduler"
    STRATEGY = "strategy"
    RISK_ENGINE = "risk_engine"
