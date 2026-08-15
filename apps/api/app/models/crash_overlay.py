"""The crash overlay's daily record (§9).

One row per trading day, holding **what was known and what was decided** — the
eleven features, the model's probability, the trigger it was compared against,
and the exposure that resulted. Kept together deliberately: separating inputs
from decisions makes it impossible to answer "why was it defensive that day?"
without joining two tables and trusting they agree.

**Backfilled to 2006 and appended nightly.** Unlike the market-regime table,
this one cannot start accumulating from today: the model is refitted from the
whole history every night, so the history *is* the model. Two decades of rows
exist from the first run.

`probability`, `trigger` and `target_exposure` are nullable because the early
rows genuinely have none — the trigger needs two years of model output to rank
today against, and until then the overlay holds fully invested by design.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import Boolean, Date, Integer, Numeric, Text, UniqueConstraint, false, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CrashOverlayReading(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One day of features, model output and the exposure decision."""

    __tablename__ = "crash_overlay_readings"
    __table_args__ = (UniqueConstraint("as_of", name="uq_crash_overlay_readings_as_of"),)

    as_of: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    #: The index close this row's features are computed against. Stored because
    #: the state machine compares against the price it stepped aside at, and
    #: refetching history to answer that would make the record depend on a
    #: provider still serving it.
    index_close: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # -- The eleven features -------------------------------------------------
    vix: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    vix_term_structure: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    credit_spread: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    hyg_tlt: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    hyg_lqd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    skew: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    small_cap_rs: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    realised_vol: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    vol_of_vol: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    drawdown_from_high: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    insider_rank: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))

    # -- What the model made of them -----------------------------------------
    #: Probability of a sharp fall within the horizon. Null before the model
    #: has enough history to be fitted at all.
    probability: Mapped[Decimal | None] = mapped_column(Numeric(10, 8))
    #: The rolling percentile this probability was compared against. Null until
    #: two years of model output exist to rank against — and a null trigger
    #: means "do not warn", never "warn".
    trigger: Mapped[Decimal | None] = mapped_column(Numeric(10, 8))
    #: `server_default` as well as `default`, and the server one is the load
    #: bearing half. `refresh()` writes its rows through a Core bulk upsert that
    #: sends only the close and the features, so nothing supplies this column on
    #: a first insert and a Python-side default never fires. The schema has
    #: carried the default since the table was created; the model simply did not
    #: say so, which `alembic check` reads as drift.
    is_warning: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )

    # -- What the overlay decided --------------------------------------------
    #: Fraction of the index sleeve to hold, 0..1. Advisory: nothing in this
    #: system places an order from it without a human turning execution on.
    target_exposure: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    #: The close the overlay stepped aside at, while it is standing aside.
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    #: Server default for the same reason as `is_warning` above.
    days_out: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    #: Why the exposure is what it is, in words, for the operator who has to
    #: understand a decision months later.
    reason: Mapped[str | None] = mapped_column(Text)
