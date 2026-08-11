"""Load and store fitted models.

The only bridge between `models_ml.logistic.FittedModel` — the arithmetic — and
`models.strategy_model.StrategyModel` — the row. Both directions live here so
the mapping cannot drift: a field added to one and forgotten in the other is the
kind of bug that produces a model which serves plausible numbers computed from
the wrong transform.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import StrategyKind
from app.models.strategy_model import StrategyModel
from app.models_ml.logistic import FittedModel

log = structlog.get_logger(__name__)


def to_fitted(row: StrategyModel) -> FittedModel:
    """Rehydrate the arithmetic from a stored row."""
    return FittedModel(
        feature_names=tuple(row.feature_names),
        coefficients=tuple(float(v) for v in row.coefficients),
        intercept=float(row.intercept),
        means=tuple(float(v) for v in row.feature_means),
        sds=tuple(float(v) for v in row.feature_sds),
        scale_known=tuple(bool(v) for v in row.scale_known),
        prior_means=tuple(float(v) for v in row.prior_means),
        prior_taus=tuple(float(v) for v in row.prior_taus),
        shrinkage=tuple(float(v) for v in row.shrinkage),
        standard_errors=tuple(float(v) for v in row.standard_errors),
        n_observations=int(row.n_observations),
        positive_rate=float(row.positive_rate or 0.0),
        auc=float(row.auc) if row.auc is not None else float("nan"),
        brier=float(row.brier) if row.brier is not None else float("nan"),
        log_loss=float(row.log_loss) if row.log_loss is not None else float("nan"),
        label_definition=row.label_definition,
        low_shrinkage_clamp=(
            float(row.low_shrinkage_clamp) if row.low_shrinkage_clamp is not None else None
        ),
    )


class StrategyModelService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def active(self, kind: StrategyKind) -> FittedModel | None:
        """The model this strategy should serve, or None if it has none.

        None is a real answer, not an error: a strategy with no fitted model
        emits no signals rather than falling back to some default weighting. A
        silent fallback would be a different strategy wearing this one's name.
        """
        row = (
            (
                await self._session.execute(
                    select(StrategyModel)
                    .where(StrategyModel.kind == kind, StrategyModel.is_active.is_(True))
                    .order_by(StrategyModel.fitted_at.desc().nullslast())
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )
        return None if row is None else to_fitted(row)

    async def save(
        self,
        kind: StrategyKind,
        model: FittedModel,
        *,
        notes: str | None = None,
        activate: bool = True,
    ) -> StrategyModel:
        """Insert a fit, optionally making it the one that serves.

        Insert-then-deactivate rather than update-in-place, so the history of
        what was believed when is preserved, and so there is never a window with
        no active model.
        """
        row = StrategyModel(
            kind=kind,
            feature_names=list(model.feature_names),
            coefficients=list(model.coefficients),
            intercept=model.intercept,
            feature_means=list(model.means),
            feature_sds=list(model.sds),
            scale_known=list(model.scale_known),
            prior_means=list(model.prior_means),
            prior_taus=list(model.prior_taus),
            shrinkage=list(model.shrinkage),
            standard_errors=list(model.standard_errors),
            low_shrinkage_clamp=model.low_shrinkage_clamp,
            n_observations=model.n_observations,
            positive_rate=model.positive_rate,
            auc=None if model.auc != model.auc else model.auc,
            brier=None if model.brier != model.brier else model.brier,
            log_loss=None if model.log_loss != model.log_loss else model.log_loss,
            label_definition=model.label_definition,
            notes=notes,
            fitted_at=datetime.now(UTC),
            is_active=activate,
        )
        self._session.add(row)
        await self._session.flush()

        if activate:
            await self._session.execute(
                update(StrategyModel)
                .where(
                    StrategyModel.kind == kind,
                    StrategyModel.id != row.id,
                    StrategyModel.is_active.is_(True),
                )
                .values(is_active=False)
            )
            await self._session.flush()
        log.info("strategy_model.saved", kind=str(kind), n=model.n_observations, auc=model.auc)
        return row
