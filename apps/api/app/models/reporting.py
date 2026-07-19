"""End-of-day reporting (§16).

A `DailyAccountSummary` is the persisted end-of-day snapshot of a venue: what the
account was worth, what was open, what moved. It exists so "how did the paper
book do last Tuesday" is answerable from a row rather than reconstructed from the
audit log, and so the same numbers can drive a summary notification and a chart.

One row per (broker, day). The EOD job upserts, so re-running a day converges
rather than duplicating.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import Date, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import (
    Base,
    Money,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import BrokerKind


class DailyAccountSummary(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One venue's end-of-day state for one day."""

    __tablename__ = "daily_account_summaries"
    __table_args__ = (
        UniqueConstraint("broker", "summary_date", name="uq_daily_account_summaries_broker_date"),
    )

    broker: Mapped[BrokerKind] = mapped_column(StrEnumType(BrokerKind, 24), nullable=False)
    summary_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)

    cash: Mapped[Any] = mapped_column(Money, nullable=False)
    equity: Mapped[Any] = mapped_column(Money, nullable=False)
    invested: Mapped[Any] = mapped_column(Money, nullable=False)
    unrealised_pnl: Mapped[Any] = mapped_column(Money, nullable=False)
    #: Realised P/L from positions closed on this day (stop, time, emergency).
    realised_pnl: Mapped[Any] = mapped_column(Money, nullable=False, default=0)
    #: Change in equity since the previous day's summary, if one exists.
    equity_change: Mapped[Any | None] = mapped_column(Money)

    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trades_today: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_halts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<DailyAccountSummary {self.broker} {self.summary_date} equity={self.equity}>"
