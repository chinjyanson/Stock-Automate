"""A fitted logistic model, stored so a strategy can serve it without refitting.

Fitting happens offline — it needs the whole replay history, and for the stock
model it needs Kronos, which needs torch, which does not fit on the deployment
box. So the fit runs on a machine that has the memory and writes a row here, and
the strategy reads it. The same store-only discipline as `kronos_predictions`
and `index_options_snapshots`.

**Three things are stored with the coefficients that might look optional and
are not.**

`feature_means` / `feature_sds` are the standardisation learned at fit time.
Fitting on z-scored features and serving on raw ones is a silent, total failure
— every probability is wrong and nothing errors. Keeping the transform in the
same row as the coefficients is what makes the two impossible to separate.

`prior_means` / `prior_taus` / `shrinkage` record how much of each coefficient
came from data rather than from an assumption. Dealer gamma cannot be
backfilled, so its coefficient ships as market-structure theory and only becomes
evidence as rows accumulate. Without `shrinkage`, a number nobody measured and a
number somebody did look identical in this table.

`label_definition` says what the probability is a probability *of*. A model read
out of here in a year is otherwise a vector of numbers with no referent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import (
    Base,
    JSONBOrJSON,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import StrategyKind


class StrategyModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One fit of one strategy's model."""

    __tablename__ = "strategy_models"
    __table_args__ = (Index("ix_strategy_models_kind_active", "kind", "is_active"),)

    kind: Mapped[StrategyKind] = mapped_column(StrEnumType(StrategyKind, 32), nullable=False)

    #: Ordered, and the order is load-bearing: coefficients, means, sds, priors
    #: and shrinkage are all positional against this list.
    feature_names: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    coefficients: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    intercept: Mapped[float] = mapped_column(Float, nullable=False)

    feature_means: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    feature_sds: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    #: Per feature: whether the training data ever pinned down its scale. A
    #: false entry means the feature is carried by its prior and must not be
    #: served, because a raw value over a fabricated scale is a meaningless
    #: number that still looks like one.
    scale_known: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)

    prior_means: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    prior_taus: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    #: Fraction of each coefficient's precision that came from data, 0..1.
    shrinkage: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)
    standard_errors: Mapped[list[Any]] = mapped_column(JSONBOrJSON, nullable=False)

    #: Ceiling on the summed log-odds contributed by features that are still
    #: mostly prior. Null means unclamped, which is right for a model whose
    #: every feature was fitted.
    low_shrinkage_clamp: Mapped[float | None] = mapped_column(Float)

    n_observations: Mapped[int] = mapped_column(Integer, nullable=False)
    positive_rate: Mapped[float | None] = mapped_column(Float)
    #: Measured on the *confirm* fold. A model chosen on the confirm fold is not
    #: a confirm fold, so these are reported and never optimised against.
    auc: Mapped[float | None] = mapped_column(Float)
    brier: Mapped[float | None] = mapped_column(Float)
    log_loss: Mapped[float | None] = mapped_column(Float)

    label_definition: Mapped[str] = mapped_column(Text, nullable=False)
    #: Free-text provenance: which script, which universe, which fold split.
    notes: Mapped[str | None] = mapped_column(Text)
    fitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: Exactly one row per kind should be active. Enforced by the loader taking
    #: the most recent active row rather than by a constraint, so re-fitting is
    #: an insert plus a deactivate and never a window with no model at all.
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_version: Mapped[str | None] = mapped_column(String(64))

    def __repr__(self) -> str:
        return f"<StrategyModel {self.kind} n={self.n_observations} auc={self.auc}>"
