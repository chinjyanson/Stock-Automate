"""SQLAlchemy models.

Every model must be imported here: Alembic autogenerate only sees tables that
are attached to `Base.metadata` at import time, and a model missing from this
list silently produces an empty migration.
"""

from app.models.audit import GENESIS_HASH, AuditEvent
from app.models.base import Base, Money, Price, Quantity, Ratio
from app.models.enums import (
    ActorKind,
    AuditEventKind,
    BrokerKind,
    DataQualityEventKind,
    DataSeriesType,
    InstrumentKind,
    Interval,
    LifecycleState,
    OperatingMode,
    OrderSide,
    OrderStatus,
    OrderType,
    PriceUnit,
    ProviderKind,
    QualityStatus,
    TradeIntentStatus,
)
from app.models.instrument import (
    BrokerInstrument,
    Exchange,
    Instrument,
    MarketDataMapping,
    TradingSchedule,
    Watchlist,
    WatchlistInstrument,
)
from app.models.market_data import (
    Candle,
    CorporateAction,
    DataQualityEvent,
    FundamentalSnapshot,
    ProviderHealth,
    ProviderUsage,
)
from app.models.scanner import (
    Classification,
    ProposalStatus,
    ScannerConfiguration,
    ScannerResult,
    ScannerRun,
    ScannerRunStatus,
    TradeProposal,
)
from app.models.system import (
    BrokerCredential,
    LiveArmingSession,
    ProviderCredential,
    SystemSetting,
)
from app.models.user import User, UserSession

__all__ = [
    "GENESIS_HASH",
    "ActorKind",
    "AuditEvent",
    "AuditEventKind",
    "Base",
    "BrokerCredential",
    "BrokerInstrument",
    "BrokerKind",
    "Candle",
    "Classification",
    "CorporateAction",
    "DataQualityEvent",
    "DataQualityEventKind",
    "DataSeriesType",
    "Exchange",
    "FundamentalSnapshot",
    "Instrument",
    "InstrumentKind",
    "Interval",
    "LifecycleState",
    "LiveArmingSession",
    "MarketDataMapping",
    "Money",
    "OperatingMode",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Price",
    "PriceUnit",
    "ProposalStatus",
    "ProviderCredential",
    "ProviderHealth",
    "ProviderKind",
    "ProviderUsage",
    "QualityStatus",
    "Quantity",
    "Ratio",
    "ScannerConfiguration",
    "ScannerResult",
    "ScannerRun",
    "ScannerRunStatus",
    "SystemSetting",
    "TradeIntentStatus",
    "TradeProposal",
    "TradingSchedule",
    "User",
    "UserSession",
    "Watchlist",
    "WatchlistInstrument",
]
