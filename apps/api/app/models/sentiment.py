"""Stored news sentiment, one row per instrument per day (§4, §6).

A snapshot table for the same reason the regime and index-options readings have
one: the scanner and the risk engine must be able to ask "how did this company's
news read?" without a network call. News is also the one input here that is
genuinely perishable — a feed cannot be re-queried for what it said last
Tuesday — so the row is the only record there will ever be of that day's tone.

Both readings are kept side by side and neither is derived from the other. The
lexicon columns are ours, computed locally from headlines. The `provider_*`
columns are Finnhub's own scoring, which is premium-gated and usually absent.
Storing them separately is what lets a consumer say which one it acted on.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Ratio, TimestampMixin, UUIDPrimaryKeyMixin


class SentimentSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One day's news tone for one instrument."""

    __tablename__ = "sentiment_snapshots"
    __table_args__ = (
        UniqueConstraint("instrument_id", "as_of", name="uq_sentiment_snapshots_instrument_asof"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    as_of: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    #: The symbol the headlines were requested under, so a mapping change stays
    #: traceable to the tone it produced.
    provider_symbol: Mapped[str | None] = mapped_column(String(64))

    # -- Loughran-McDonald, computed locally --------------------------------
    #: -1..+1. Null when the window held no tone words at all, which is a
    #: non-observation and must not be read as neutral.
    polarity: Mapped[Decimal | None] = mapped_column(Ratio)
    uncertainty: Mapped[Decimal | None] = mapped_column(Ratio)
    positive_words: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    negative_words: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Articles that contributed tone. Zero with a non-null row means the sweep
    #: ran and found nothing to say — which is why the row exists at all.
    headline_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # -- Finnhub's own score, when the plan allows it -----------------------
    provider_score: Mapped[Decimal | None] = mapped_column(Ratio)
    provider_bullish_pct: Mapped[Decimal | None] = mapped_column(Ratio)
    provider_buzz: Mapped[Decimal | None] = mapped_column(Ratio)

    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def __repr__(self) -> str:
        return f"<SentimentSnapshot {self.instrument_id} {self.as_of} polarity={self.polarity}>"
