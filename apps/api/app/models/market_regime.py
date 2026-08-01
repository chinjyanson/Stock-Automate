"""Market-wide risk posture (§9).

One row per daily reading of the conditions that apply to *every* position
rather than to any one instrument: volatility, market breadth, index trend, the
yield curve, and long-run valuation.

Stored rather than recomputed on demand for two reasons. Alerts fire on a
*change* of state, which needs yesterday's reading to compare against; and the
history accumulates into the record that would be needed to judge, later,
whether any of this helped — a judgement that cannot be made today with ~13
months of candles.

The columns split deliberately into two groups:

  * **Fast** — VIX, breadth, index regime. These move on a timescale where
    reacting is meaningful, and they drive `risk_factor`.
  * **Context** — yield curve, Buffett indicator. Recorded and alerted on, but
    kept out of the risk factor. Both have been at extremes for years; gating
    trading on them would mean not trading, which is a way of being wrong that
    is easy to mistake for caution.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import Boolean, Date, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class MarketRegimeSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One day's reading of market-wide conditions."""

    __tablename__ = "market_regime_snapshots"
    __table_args__ = (UniqueConstraint("as_of", name="uq_market_regime_snapshots_as_of"),)

    as_of: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # -- Fast signals: these drive the risk factor ---------------------------
    #: CBOE volatility index. The market's own priced expectation of turbulence.
    vix: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    #: Fraction of instruments trading above their own 200-day average, 0..1.
    #: Deteriorates before an index does, which is the opposite of a moving
    #: average crossover's lag.
    breadth_above_200dma: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    #: How many instruments the breadth reading is computed from — a breadth of
    #: 0.4 over 30 names means something very different from 0.4 over 4,000.
    breadth_sample: Mapped[int | None] = mapped_column()
    #: True when the 50-day average is above the 200-day (a "golden cross").
    sp500_uptrend: Mapped[bool | None] = mapped_column(Boolean)
    world_uptrend: Mapped[bool | None] = mapped_column(Boolean)
    #: S&P proxy's distance from its own 200-day average, as a fraction.
    sp500_vs_200dma: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))

    # -- Context: alerted on, deliberately excluded from the risk factor ------
    #: 10-year minus 2-year Treasury yield. Negative is an inversion.
    yield_curve_10y2y: Mapped[Decimal | None] = mapped_column(Numeric(8, 3))
    yield_curve_10y3m: Mapped[Decimal | None] = mapped_column(Numeric(8, 3))
    #: Corporate equities over GDP, as a percentage. Quarterly and lagging by
    #: construction — it comes from the Fed's Z.1 release — so it can describe
    #: valuation but never warn of anything imminent.
    buffett_indicator: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))

    # -- The verdict ---------------------------------------------------------
    #: Multiplier applied to risk-per-trade, floored well above zero. A regime
    #: reduces position size; it does not stop the strategy trading.
    risk_factor: Mapped[Decimal] = mapped_column(Numeric(6, 4), nullable=False, default=1)
    #: Short human-readable summary of what drove the factor, for the alert body
    #: and for reading the history back later.
    posture: Mapped[str | None] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<MarketRegimeSnapshot {self.as_of} {self.posture} risk={self.risk_factor}>"
