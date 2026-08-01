"""Insider (SEC Form 4) transactions (§4, §6).

Corporate officers and directors must report their own trades in the issuer's
stock within two business days. Those filings are public, free, and near
real-time from EDGAR — the newest is typically minutes old — which makes them a
usable scoring input rather than a curiosity.

Two things this table exists to keep straight, because getting either wrong
turns a signal into noise:

  * **Most Form 4 rows are not decisions.** Only codes `P` (open-market
    purchase) and `S` (open-market sale) reflect someone choosing to trade.
    Tax-withholding (`F`), option exercises (`M`), grants (`A`) and gifts (`G`)
    are mechanical and carry no view. They are stored anyway — knowing a filing
    was seen and dismissed is worth more than a gap — and marked non-signalling.

  * **Whether the market has already moved.** An insider filing is only an edge
    until it is priced in. The transaction and filing dates are both kept, so
    the score can compare the price then with the price now and stand down when
    the move has already happened.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

#: Form 4 transaction codes that represent a *decision* to trade at market.
#: Everything else is mechanical and carries no directional view.
SIGNALLING_CODES = frozenset({"P", "S"})


class InsiderTransaction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One transaction line from one Form 4 filing."""

    __tablename__ = "insider_transactions"
    __table_args__ = (
        # EDGAR's accession number identifies the filing; a filing can carry
        # several transaction lines, so the line index completes the key. This
        # is what makes re-polling the feed idempotent.
        UniqueConstraint(
            "accession_number", "line_index", name="uq_insider_transactions_accession_line"
        ),
        Index("ix_insider_transactions_instrument_date", "instrument_id", "transaction_date"),
        Index("ix_insider_transactions_ticker", "issuer_ticker"),
    )

    #: EDGAR accession number, e.g. "0001140361-26-029923".
    accession_number: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Which transaction row within that filing (Form 4s often report several).
    line_index: Mapped[int] = mapped_column(nullable=False, default=0)

    #: Resolved local instrument. Null when the ticker matches nothing we hold —
    #: the row is still stored so the resolution can be retried once the
    #: catalogue grows, rather than the filing being lost.
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="SET NULL"), index=True
    )
    issuer_ticker: Mapped[str | None] = mapped_column(String(16))
    issuer_name: Mapped[str | None] = mapped_column(String(200))

    insider_name: Mapped[str | None] = mapped_column(String(200))
    #: Free text from the filing, e.g. "COO and SVP, Compliance".
    officer_title: Mapped[str | None] = mapped_column(Text)
    is_officer: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_director: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_ten_percent_owner: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: Chief-level title (CEO/CFO/COO/...) or President. The strongest role tier:
    #: a chief executive's own money is the most informative variant of this
    #: signal, and the weakest is a 10% owner rebalancing a fund.
    is_c_suite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Form 4 transaction code — P, S, F, M, A, G, ...
    transaction_code: Mapped[str | None] = mapped_column(String(2))
    #: "A" acquired or "D" disposed, as filed.
    acquired_disposed: Mapped[str | None] = mapped_column(String(1))
    transaction_date: Mapped[date | None] = mapped_column(Date, index=True)
    #: When EDGAR published it. Distinct from `transaction_date`: the gap between
    #: them is dead time in which the market could already have reacted.
    filed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    shares: Mapped[Decimal | None] = mapped_column(Numeric(20, 4))
    price_per_share: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    #: shares x price, in the filing's currency (USD for Form 4).
    value_usd: Mapped[Decimal | None] = mapped_column(Numeric(20, 2))
    shares_owned_after: Mapped[Decimal | None] = mapped_column(Numeric(20, 4))

    #: A sale under a pre-arranged Rule 10b5-1 plan is scheduled, not a fresh
    #: opinion, so it must not be read as one.
    is_10b5_1_plan: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: True only for codes in `SIGNALLING_CODES` — set at ingest so scoring
    #: filters on an index rather than re-deriving it per row.
    is_signalling: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    filing_url: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return (
            f"<InsiderTransaction {self.issuer_ticker} {self.transaction_code} "
            f"{self.shares} @ {self.price_per_share}>"
        )
