"""Internal paper broker state (§3, §7).

These tables are the *venue's* truth for the internal paper simulator — the
equivalent of what Trading 212 holds on its servers, not our domain records.
They are owned by `app.broker.internal_paper` and read/written only through it,
which is what makes the simulator a first-class execution venue rather than a
test double: its cash and positions survive a process restart, so a real
reconciliation (broker truth vs local `TradeIntent`) has two independent sides
to compare.

The ticker convention is deliberate: `broker_ticker` holds the canonical
`Instrument.id` as a string. The internal venue has no tickers of its own, so
the instrument id *is* the venue symbol, and no `BrokerInstrument` lookup is
needed to price a fill or to map a reconciliation back to an instrument.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import (
    Base,
    Money,
    Price,
    Quantity,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import OrderSide, OrderStatus, OrderType


class PaperBrokerAccount(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The simulated account's cash. One logical account per currency."""

    __tablename__ = "paper_broker_accounts"
    __table_args__ = (UniqueConstraint("currency", name="uq_paper_broker_accounts_currency"),)

    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="GBP")
    cash: Mapped[Any] = mapped_column(Money, nullable=False)
    starting_cash: Mapped[Any] = mapped_column(Money, nullable=False)

    def __repr__(self) -> str:
        return f"<PaperBrokerAccount {self.currency} cash={self.cash}>"


class PaperBrokerPosition(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An open simulated position, keyed by the venue ticker (instrument id)."""

    __tablename__ = "paper_broker_positions"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "broker_ticker", name="uq_paper_broker_positions_account_ticker"
        ),
    )

    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_broker_accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    broker_ticker: Mapped[str] = mapped_column(String(64), nullable=False)
    quantity: Mapped[Any] = mapped_column(Quantity, nullable=False)
    average_price: Mapped[Any] = mapped_column(Price, nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return f"<PaperBrokerPosition {self.broker_ticker} qty={self.quantity}>"


class PaperBrokerOrder(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A simulated order. Market orders land FILLED; stops rest until triggered."""

    __tablename__ = "paper_broker_orders"
    __table_args__ = (
        UniqueConstraint(
            "broker_order_id", name="uq_paper_broker_orders_broker_order_id"
        ),
        Index("ix_paper_broker_orders_account_status", "account_id", "status"),
    )

    account_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_broker_accounts.id", ondelete="CASCADE"), nullable=False
    )
    broker_order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    broker_ticker: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[OrderSide] = mapped_column(StrEnumType(OrderSide, 8), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(StrEnumType(OrderType, 16), nullable=False)
    quantity: Mapped[Any] = mapped_column(Quantity, nullable=False)
    status: Mapped[OrderStatus] = mapped_column(StrEnumType(OrderStatus, 20), nullable=False)
    filled_quantity: Mapped[Any] = mapped_column(Quantity, nullable=False, default=0)
    average_fill_price: Mapped[Any | None] = mapped_column(Price)
    stop_price: Mapped[Any | None] = mapped_column(Price)
    limit_price: Mapped[Any | None] = mapped_column(Price)
    client_reference: Mapped[str | None] = mapped_column(String(128))
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    terminal_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<PaperBrokerOrder {self.side} {self.broker_ticker} {self.status}>"
