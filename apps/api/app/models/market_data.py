"""Local market-data store (§4).

Once fetched, this database is the working source of truth. Providers are
upstream of it, not consulted at decision time. Every candle carries its
provenance (which provider, which symbol, when retrieved) and a quality status,
because a strategy is only permitted to act on data it can vouch for.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import (
    Base,
    Money,
    Price,
    Quantity,
    Ratio,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from app.models.instrument import Instrument
from app.models.enums import (
    DataQualityEventKind,
    DataSeriesType,
    Interval,
    PriceUnit,
    ProviderKind,
    QualityStatus,
)


class Candle(UUIDPrimaryKeyMixin, Base):
    """One OHLCV bar, normalised to the instrument's price unit.

    Prices stored here are already unit-normalised (GBX divided to GBP at the
    adapter boundary); `price_unit` records what the instrument is denominated
    in after normalisation, and adapters are responsible for never letting a
    raw pence quote reach this table.

    The unique key is instrument+interval+timestamp+series_type, so a raw and an
    adjusted bar for the same minute coexist without being treated as a
    conflict, while a genuine duplicate from a re-fetch upserts in place.
    """

    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "interval",
            "timestamp",
            "data_series_type",
            name="uq_candles_instrument_interval_timestamp_series",
        ),
        # The hot path: "give me the last N closed bars for this instrument".
        Index(
            "ix_candles_instrument_interval_timestamp",
            "instrument_id",
            "interval",
            "timestamp",
        ),
        Index("ix_candles_quality", "quality_status"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[ProviderKind] = mapped_column(StrEnumType(ProviderKind, 16), nullable=False)
    #: The symbol actually requested, retained so a mapping change is traceable.
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    interval: Mapped[Interval] = mapped_column(StrEnumType(Interval, 8), nullable=False)
    data_series_type: Mapped[DataSeriesType] = mapped_column(
        StrEnumType(DataSeriesType, 12), nullable=False, default=DataSeriesType.RAW
    )

    #: Bar OPEN time, timezone-aware UTC. Using open time consistently matters:
    #: mixing open- and close-stamped bars silently shifts signals by one bar.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    open: Mapped[Decimal] = mapped_column(Price, nullable=False)
    high: Mapped[Decimal] = mapped_column(Price, nullable=False)
    low: Mapped[Decimal] = mapped_column(Price, nullable=False)
    close: Mapped[Decimal] = mapped_column(Price, nullable=False)
    adjusted_close: Mapped[Decimal | None] = mapped_column(Price)
    volume: Mapped[Decimal | None] = mapped_column(Quantity)

    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    price_unit: Mapped[PriceUnit] = mapped_column(StrEnumType(PriceUnit, 3), nullable=False)
    is_adjusted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: False while the bar is still forming. A strategy must never read one
    #: (§4): the close is not yet the close.
    is_closed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    quality_status: Mapped[QualityStatus] = mapped_column(
        StrEnumType(QualityStatus, 16), nullable=False, default=QualityStatus.OK
    )
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    instrument: Mapped[Instrument] = relationship("Instrument", viewonly=True)

    def __repr__(self) -> str:
        return f"<Candle {self.provider_symbol} {self.interval} {self.timestamp.isoformat()}>"


class CorporateAction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Splits and dividends, needed to reconcile raw against adjusted series."""

    __tablename__ = "corporate_actions"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "action_type",
            "ex_date",
            name="uq_corporate_actions_instrument_type_exdate",
        ),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: "split" or "dividend".
    action_type: Mapped[str] = mapped_column(String(16), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    #: Split ratio (2.0 == 2-for-1), null for dividends.
    ratio: Mapped[Decimal | None] = mapped_column(Ratio)
    #: Cash amount per share, null for splits.
    amount: Mapped[Decimal | None] = mapped_column(Money)
    currency: Mapped[str | None] = mapped_column(String(3))
    provider: Mapped[ProviderKind] = mapped_column(StrEnumType(ProviderKind, 16), nullable=False)


class FundamentalSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Point-in-time optional fundamentals.

    Every field is nullable by design. A missing fundamental means *unknown*,
    and the scanner must treat it as unknown rather than as zero or as a
    failure (§6, acceptance criterion 7).
    """

    __tablename__ = "fundamental_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "provider", "as_of", name="uq_fundamental_snapshots_instrument_asof"
        ),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[ProviderKind] = mapped_column(StrEnumType(ProviderKind, 16), nullable=False)
    as_of: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    market_cap: Mapped[Decimal | None] = mapped_column(Money)
    trailing_pe: Mapped[Decimal | None] = mapped_column(Ratio)
    price_to_book: Mapped[Decimal | None] = mapped_column(Ratio)
    dividend_yield: Mapped[Decimal | None] = mapped_column(Ratio)
    revenue_growth: Mapped[Decimal | None] = mapped_column(Ratio)
    earnings_growth: Mapped[Decimal | None] = mapped_column(Ratio)
    profit_margin: Mapped[Decimal | None] = mapped_column(Ratio)
    debt_to_equity: Mapped[Decimal | None] = mapped_column(Ratio)
    currency: Mapped[str | None] = mapped_column(String(3))

    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DataQualityEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A recorded defect in the local store.

    Written rather than logged so the UI can show *why* an instrument is not
    tradable right now, and so the daily summary can report incidents (§13).
    """

    __tablename__ = "data_quality_events"
    __table_args__ = (
        Index("ix_data_quality_events_instrument_kind", "instrument_id", "kind"),
        Index("ix_data_quality_events_unresolved", "resolved_at"),
    )

    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[DataQualityEventKind] = mapped_column(
        StrEnumType(DataQualityEventKind, 24), nullable=False
    )
    provider: Mapped[ProviderKind | None] = mapped_column(StrEnumType(ProviderKind, 16))
    interval: Mapped[Interval | None] = mapped_column(StrEnumType(Interval, 8))
    #: "info" | "warning" | "error". Errors block signal generation.
    severity: Mapped[str] = mapped_column(String(8), nullable=False, default="warning")
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict[str, Any] | None] = mapped_column()
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ProviderUsage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Credit ledger enforcing free-tier budgets (§4).

    One row per provider per UTC day. Counters are incremented inside the same
    transaction that authorises a request, so a crash cannot lose the spend and
    overrun the plan.
    """

    __tablename__ = "provider_usage"
    __table_args__ = (
        UniqueConstraint("provider", "usage_date", name="uq_provider_usage_provider_date"),
    )

    provider: Mapped[ProviderKind] = mapped_column(
        StrEnumType(ProviderKind, 16), nullable=False, index=True
    )
    usage_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    requests_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requests_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Snapshot of the limit in force, so historical rows stay interpretable
    #: after the budget is retuned.
    operational_limit: Mapped[int] = mapped_column(Integer, nullable=False)
    emergency_reserve: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_request_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @property
    def remaining(self) -> int:
        return max(0, self.operational_limit - self.requests_used)

    def __repr__(self) -> str:
        return f"<ProviderUsage {self.provider} {self.usage_date} {self.requests_used}>"


class ProviderHealth(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Rolling health of one provider, surfaced on the dashboard (§14)."""

    __tablename__ = "provider_health"
    __table_args__ = (UniqueConstraint("provider", name="uq_provider_health_provider"),)

    provider: Mapped[ProviderKind] = mapped_column(StrEnumType(ProviderKind, 16), nullable=False)
    is_healthy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_latency_ms: Mapped[int | None] = mapped_column(Integer)
    last_error: Mapped[str | None] = mapped_column(Text)
