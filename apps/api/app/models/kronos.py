"""Kronos forecast snapshots (§8).

Kronos is a decoder-only foundation model over K-line sequences: it *samples*
future candles autoregressively rather than emitting a point estimate. That has
two consequences this table exists to absorb.

**It is slow and heavy.** Torch is ~250-300MB resident on import alone, against a
448MB worker on the free-tier box, and generation is per-instrument. So it can
never run inside a scan. It runs as its own job — on a machine that has the
memory, which today means locally — and writes here. Everything downstream reads
the table and never imports torch.

**It is stochastic.** `sample_count` paths are drawn and reduced to a mean, so
the same input can give slightly different answers. `path_dispersion` records how
much the samples disagreed, which is the model's own uncertainty and is worth
more than the point forecast: a confident wrong answer and an uncertain one look
identical without it.

The same shape as `sentiment_snapshots` and `index_option_snapshots`, and for the
same reason — anything needing a network, a GPU or a large dependency is a job
that writes a table, never a call inside the pipeline.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Ratio, TimestampMixin, UUIDPrimaryKeyMixin


class KronosPrediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One instrument's forecast, as of one date."""

    __tablename__ = "kronos_predictions"
    #: Keyed by model and horizon as well as instrument and date, so a run with
    #: different weights or a different horizon lands beside the existing row
    #: rather than overwriting it. Two variants disagreeing about the same day
    #: is exactly the comparison that decides which to keep; a narrower key
    #: would silently destroy it. Readers filter to the variant they want.
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "as_of",
            "model_name",
            "horizon_days",
            name="uq_kronos_predictions_instrument_date_model",
        ),
        Index("ix_kronos_predictions_as_of", "as_of"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: The date of the last *observed* bar. Everything here is a forecast made
    #: from bars up to and including this one — no later data was visible.
    as_of: Mapped[date] = mapped_column(Date, nullable=False)

    #: Forecast horizon in trading days.
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Mean predicted return over the horizon, as a fraction. The headline.
    predicted_return: Mapped[Any] = mapped_column(Ratio, nullable=False)
    #: Standard deviation of the sampled paths' terminal returns. High means the
    #: model is unsure, which is a usable signal in its own right — and the
    #: reason a bare point forecast would be the wrong thing to store.
    path_dispersion: Mapped[Any] = mapped_column(Ratio, nullable=False)
    #: Fraction of sampled paths ending above the last observed close. A
    #: probability, and the natural input to a classifier.
    prob_up: Mapped[Any] = mapped_column(Ratio, nullable=False)
    #: Deepest drawdown along the mean path, as a positive fraction. A forecast
    #: that arrives via a 30% fall is not the same trade as a smooth one.
    predicted_drawdown: Mapped[Any] = mapped_column(Ratio, nullable=False)

    #: Provenance. A prediction is only interpretable against the weights that
    #: produced it, and these models are small enough to be swapped casually.
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    context_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Wall-clock generation time. Recorded because throughput is what decides
    #: whether this can cover a universe or only a watchlist.
    generation_ms: Mapped[int | None] = mapped_column(Integer)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def as_features(self) -> dict[str, float]:
        """The prediction as model inputs, named to match the feature pipeline."""
        return {
            "kronos_return": float(Decimal(str(self.predicted_return))),
            "kronos_dispersion": float(Decimal(str(self.path_dispersion))),
            "kronos_prob_up": float(Decimal(str(self.prob_up))),
            "kronos_drawdown": float(Decimal(str(self.predicted_drawdown))),
        }
