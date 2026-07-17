"""Instrument identity, broker catalogue and provider mapping (§5).

A ticker is not an identifier. "VUSA" on one venue and "VUSA" on another are
different instruments; "SPY" at one provider may be a different series than
"SPY" at another. Identity resolves in this order:

    1. ISIN
    2. Exchange MIC + exchange ticker
    3. Provider symbol + exchange
    4. Name + currency + exchange
    5. Manual user confirmation

`Instrument` is the canonical entity. `BrokerInstrument` is what Trading 212
will accept in an order. `MarketDataMapping` is what a data provider will
answer to. These are deliberately three tables: they disagree in practice, and
collapsing them is how an order gets placed against the wrong security.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

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

from app.models.base import Base, StrEnumType, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    BrokerKind,
    InstrumentKind,
    LifecycleState,
    PriceUnit,
    ProviderKind,
)


class Exchange(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A trading venue, keyed by ISO 10383 MIC."""

    __tablename__ = "exchanges"

    mic: Mapped[str] = mapped_column(String(4), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str | None] = mapped_column(String(2))
    #: IANA zone, e.g. "Europe/London". Drives session boundaries and DST.
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    currency: Mapped[str | None] = mapped_column(String(3))
    #: pandas_market_calendars identifier, when one exists for this venue.
    calendar_name: Mapped[str | None] = mapped_column(String(32))

    instruments: Mapped[list[Instrument]] = relationship(back_populates="exchange")

    def __repr__(self) -> str:
        return f"<Exchange {self.mic}>"


class TradingSchedule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Resolved session times for one venue on one date.

    Materialised per-date rather than computed from a rule, because holidays,
    half-days and DST transitions are exactly the cases a rule gets wrong, and
    a wrong session boundary means trading on an unclosed candle (§4).
    """

    __tablename__ = "trading_schedules"
    __table_args__ = (
        UniqueConstraint("exchange_id", "session_date", name="uq_trading_schedules_exchange_date"),
    )

    exchange_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exchanges.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    open_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    close_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: True for half-days and other shortened sessions; such days produce fewer
    #: intraday candles than expected and must not raise a gap alert.
    is_partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(String(200))

    exchange: Mapped[Exchange] = relationship()


class Instrument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Canonical, provider-neutral and broker-neutral security."""

    __tablename__ = "instruments"
    __table_args__ = (
        UniqueConstraint("isin", "exchange_id", name="uq_instruments_isin_exchange"),
        Index("ix_instruments_name_search", "name"),
    )

    #: Preferred identity. Nullable because some instruments (notably US
    #: listings via certain providers) arrive without one and must fall back to
    #: MIC+ticker until a user confirms.
    isin: Mapped[str | None] = mapped_column(String(12), index=True)
    exchange_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("exchanges.id", ondelete="RESTRICT"), index=True
    )
    #: Ticker as the exchange lists it — not a provider's or broker's spelling.
    exchange_ticker: Mapped[str | None] = mapped_column(String(32), index=True)

    name: Mapped[str] = mapped_column(String(300), nullable=False)
    kind: Mapped[InstrumentKind] = mapped_column(
        StrEnumType(InstrumentKind, 16), nullable=False, default=InstrumentKind.UNKNOWN
    )
    #: Currency the instrument settles in (ISO 4217).
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    #: Unit prices are quoted in, which may differ from `currency` (GBX vs GBP).
    price_unit: Mapped[PriceUnit] = mapped_column(StrEnumType(PriceUnit, 3), nullable=False)

    country: Mapped[str | None] = mapped_column(String(2))
    sector: Mapped[str | None] = mapped_column(String(120))
    industry: Mapped[str | None] = mapped_column(String(120))

    #: Set when identity was resolved below the ISIN tier and a human agreed.
    identity_confirmed_by_user: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    identity_resolution_note: Mapped[str | None] = mapped_column(Text)

    lifecycle_state: Mapped[LifecycleState] = mapped_column(
        StrEnumType(LifecycleState, 24),
        nullable=False,
        default=LifecycleState.DISCOVERED,
        index=True,
    )
    #: Free-text reason for the current lifecycle state, shown in the UI so a
    #: VALIDATION_FAILED instrument explains itself.
    lifecycle_note: Mapped[str | None] = mapped_column(Text)

    is_scanner_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: Bot Universe membership. Never set implicitly by broker sync (§7).
    is_bot_universe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suspension_reason: Mapped[str | None] = mapped_column(Text)

    last_scanned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    exchange: Mapped[Exchange | None] = relationship(back_populates="instruments")
    broker_instruments: Mapped[list[BrokerInstrument]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )
    data_mappings: Mapped[list[MarketDataMapping]] = relationship(
        back_populates="instrument", cascade="all, delete-orphan"
    )

    @property
    def is_suspended(self) -> bool:
        return self.suspended_at is not None

    def __repr__(self) -> str:
        return f"<Instrument {self.isin or self.exchange_ticker} {self.name[:30]}>"


class BrokerInstrument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """What the broker will accept in an order request.

    Trading 212's ticker spelling is its own namespace and does not match the
    exchange ticker. `is_currently_available` reflects the last sync: an
    instrument can leave the catalogue, and trading it afterwards fails.
    """

    __tablename__ = "broker_instruments"
    __table_args__ = (
        UniqueConstraint("broker", "broker_ticker", name="uq_broker_instruments_broker_ticker"),
        Index("ix_broker_instruments_instrument_broker", "instrument_id", "broker"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    broker: Mapped[BrokerKind] = mapped_column(
        StrEnumType(BrokerKind, 24), nullable=False, index=True
    )
    #: Opaque broker identifier, e.g. Trading 212's "VUAGl_EQ".
    broker_ticker: Mapped[str] = mapped_column(String(64), nullable=False)
    broker_name: Mapped[str | None] = mapped_column(String(300))
    broker_isin: Mapped[str | None] = mapped_column(String(12))
    currency: Mapped[str | None] = mapped_column(String(3))

    is_currently_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: Minimum tradable quantity and step, needed to round an order safely (§9).
    min_quantity: Mapped[Any | None] = mapped_column(String(32))
    quantity_step: Mapped[Any | None] = mapped_column(String(32))
    supports_fractional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Verbatim broker payload, retained for provenance and debugging.
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column()

    instrument: Mapped[Instrument] = relationship(back_populates="broker_instruments")

    def __repr__(self) -> str:
        return f"<BrokerInstrument {self.broker}:{self.broker_ticker}>"


class MarketDataMapping(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """What a market-data provider answers to for this instrument.

    One instrument may map to several providers, and to a *different* symbol at
    each. `is_signal_source` marks the mapping a strategy reads to decide;
    execution still happens against the broker instrument. That indirection is
    the SPY-signal/VUAG-execution case in §5 and must stay visible.
    """

    __tablename__ = "market_data_mappings"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "provider",
            "provider_symbol",
            name="uq_market_data_mappings_instrument_provider_symbol",
        ),
        Index("ix_market_data_mappings_provider_symbol", "provider", "provider_symbol"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[ProviderKind] = mapped_column(
        StrEnumType(ProviderKind, 16), nullable=False, index=True
    )
    #: Provider's spelling, e.g. "VUAG.L" for yfinance, "SPY" for Twelve Data.
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Unit this provider quotes in. yfinance returns LSE prices in GBX.
    price_unit: Mapped[PriceUnit | None] = mapped_column(StrEnumType(PriceUnit, 3))

    #: Rank within a provider when several symbols could serve; lower wins.
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    #: True when strategies should read *this* series for signals even though
    #: orders route to the instrument's broker ticker.
    is_signal_source: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: How identity was established, for the provenance display (§6).
    resolution_method: Mapped[str | None] = mapped_column(String(64))
    confirmed_by_user: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument] = relationship(back_populates="data_mappings")

    def __repr__(self) -> str:
        return f"<MarketDataMapping {self.provider}:{self.provider_symbol}>"


class Watchlist(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "watchlists"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_watchlists_user_name"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    entries: Mapped[list[WatchlistInstrument]] = relationship(
        back_populates="watchlist", cascade="all, delete-orphan"
    )


class WatchlistInstrument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "watchlist_instruments"
    __table_args__ = (
        UniqueConstraint(
            "watchlist_id", "instrument_id", name="uq_watchlist_instruments_watchlist_instrument"
        ),
    )

    watchlist_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    note: Mapped[str | None] = mapped_column(Text)

    watchlist: Mapped[Watchlist] = relationship(back_populates="entries")
    instrument: Mapped[Instrument] = relationship()
