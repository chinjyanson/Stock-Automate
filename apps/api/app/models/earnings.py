"""Earnings dates, for the post-announcement drift signal."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from sqlalchemy import Date, ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EarningsEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One company's report on one date.

    Only the *date* comes from a provider. The surprise is measured here, from
    candles already stored, as the stock's abnormal return around the report —
    price relative to the benchmark over the announcement window. Analyst
    estimates are the conventional input and are not reliably free; the market's
    own reaction is both free and arguably the better measure, since it is the
    thing the drift continues.
    """

    __tablename__ = "earnings_events"
    __table_args__ = (
        UniqueConstraint("instrument_id", "report_date", name="uq_earnings_events_instrument_date"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    report_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    #: Abnormal return over the announcement window: the stock's move minus the
    #: benchmark's, as a fraction. Null until it can be measured — a report
    #: dated in the future has not moved anything yet.
    surprise: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    def __repr__(self) -> str:
        return f"<EarningsEvent {self.instrument_id} {self.report_date} surprise={self.surprise}>"
