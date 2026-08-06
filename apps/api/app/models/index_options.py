"""Daily record of what the index options market was pricing."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import Date, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class IndexOptionsSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One day's reading of the S&P index option chain.

    An option chain is a snapshot with no history behind it — the provider will
    happily say what is priced today and has nothing to say about last month.
    So the history is built here, one row a day, from the day this starts
    running. Anything that wants a trend in these numbers has to wait for it.
    """

    __tablename__ = "index_options_snapshots"
    __table_args__ = (UniqueConstraint("as_of", name="uq_index_options_snapshots_as_of"),)

    as_of: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    #: Which proxy produced the reading (`^SPX`, or `SPY` when it fell back).
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    spot: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    #: Days to the expiry the reading was taken from, ~30 by design.
    expiry_days: Mapped[int | None] = mapped_column(Integer)

    #: Net dealer gamma in billions per 1% move. Positive = dealers long gamma,
    #: hedging *against* moves, so volatility is dampened and dips get bought.
    #: Negative = dealers short gamma, hedging *with* moves, so they are
    #: amplified. The sign rests on an assumed dealer position — see
    #: `app.signals.spx_options`.
    gamma_exposure: Mapped[Decimal | None] = mapped_column(Numeric(16, 4))
    #: 25-delta put IV minus 25-delta call IV. Positive = downside protection
    #: costs more than upside, and a steepening is the market pricing tail risk.
    skew_25delta: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    #: Implied volatility at the money — the market's forward expectation, as
    #: against the realised volatility the regime service measures.
    atm_iv: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    #: How many strikes carried enough open interest to enter the gamma sum. A
    #: reading built from six contracts deserves less trust than one from six
    #: hundred, and that is not visible from the number alone.
    contracts_used: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<IndexOptionsSnapshot {self.as_of} {self.symbol} gex={self.gamma_exposure}>"
