"""Fitted models, Kronos predictions, and a safe key for the options history.

Three changes, all groundwork for replacing the hand-weighted strategies with
fitted logistic models.

**strategy_models.** A fit needs the whole replay history, and the stock model
needs Kronos, which needs torch, which does not fit in a 448M worker. So fitting
happens offline and the strategy reads a row. The standardisation, the priors
and the per-coefficient `shrinkage` are stored beside the coefficients rather
than alongside them somewhere: serving on a different transform than the fit
used is a silent, total failure, and a coefficient carried by a prior is
indistinguishable from a measured one unless the row says so.

**kronos_predictions.** The table the model already described but no migration
ever created. Keyed by model name and horizon as well as instrument and date, so
two variants can disagree about the same day — which is the comparison that
decides which variant to keep.

**index_options_snapshots: keyed by (as_of, symbol), not as_of.** This is a
correctness fix, and it has to land before any rows accumulate. The reading
falls back from `^SPX` to `SPY`, and a gamma exposure goes as
`open interest x spot` — gamma itself goes as `1/S`, so only one of the two
factors of spot survives. SPX trades near ten times SPY's level while SPY
carries far more contracts, and those do not cancel: the reading steps by the
contract ratio over ten the day a fallback fires. Under a date-only key one
series would silently contain both, and a several-fold step is small enough to
pass for a market move rather than announce itself as a bug. Since this
history exists precisely to be fitted on, that corruption would be permanent and
undetectable. The new `gamma_tilt` and `charm_tilt` columns hold net-to-gross
ratios, which are comparable across proxies because the open interest and the
multiplier cancel; they are what a model reads.

Revision ID: 74e1367771fc
Revises: d5a92b7e4f16
Create Date: 2026-08-11 10:00:06.574884+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import app.models.base

revision: str = "74e1367771fc"
down_revision: str | None = "d5a92b7e4f16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategy_models",
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("feature_names", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("coefficients", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("intercept", sa.Float(), nullable=False),
        sa.Column("feature_means", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("feature_sds", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("scale_known", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("prior_means", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("prior_taus", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("shrinkage", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("standard_errors", app.models.base.JSONBOrJSON(), nullable=False),
        sa.Column("low_shrinkage_clamp", sa.Float(), nullable=True),
        sa.Column("n_observations", sa.Integer(), nullable=False),
        sa.Column("positive_rate", sa.Float(), nullable=True),
        sa.Column("auc", sa.Float(), nullable=True),
        sa.Column("brier", sa.Float(), nullable=True),
        sa.Column("log_loss", sa.Float(), nullable=True),
        sa.Column("label_definition", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("fitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_strategy_models")),
    )
    op.create_index(
        "ix_strategy_models_kind_active", "strategy_models", ["kind", "is_active"], unique=False
    )
    op.create_table(
        "kronos_predictions",
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("predicted_return", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("path_dispersion", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("prob_up", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("predicted_drawdown", sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("context_bars", sa.Integer(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("generation_ms", sa.Integer(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["instrument_id"],
            ["instruments.id"],
            name=op.f("fk_kronos_predictions_instrument_id_instruments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kronos_predictions")),
        sa.UniqueConstraint(
            "instrument_id",
            "as_of",
            "model_name",
            "horizon_days",
            name="uq_kronos_predictions_instrument_date_model",
        ),
    )
    op.create_index("ix_kronos_predictions_as_of", "kronos_predictions", ["as_of"], unique=False)
    op.create_index(
        op.f("ix_kronos_predictions_instrument_id"),
        "kronos_predictions",
        ["instrument_id"],
        unique=False,
    )
    op.add_column(
        "index_options_snapshots",
        sa.Column("gamma_tilt", sa.Numeric(precision=8, scale=6), nullable=True),
    )
    op.add_column(
        "index_options_snapshots",
        sa.Column("charm_exposure", sa.Numeric(precision=16, scale=4), nullable=True),
    )
    op.add_column(
        "index_options_snapshots",
        sa.Column("charm_tilt", sa.Numeric(precision=8, scale=6), nullable=True),
    )
    op.drop_constraint(
        op.f("uq_index_options_snapshots_as_of"), "index_options_snapshots", type_="unique"
    )
    op.create_unique_constraint(
        "uq_index_options_snapshots_as_of_symbol", "index_options_snapshots", ["as_of", "symbol"]
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_index_options_snapshots_as_of_symbol", "index_options_snapshots", type_="unique"
    )
    op.create_unique_constraint(
        op.f("uq_index_options_snapshots_as_of"),
        "index_options_snapshots",
        ["as_of"],
        postgresql_nulls_not_distinct=False,
    )
    op.drop_column("index_options_snapshots", "charm_tilt")
    op.drop_column("index_options_snapshots", "charm_exposure")
    op.drop_column("index_options_snapshots", "gamma_tilt")
    op.drop_index(op.f("ix_kronos_predictions_instrument_id"), table_name="kronos_predictions")
    op.drop_index("ix_kronos_predictions_as_of", table_name="kronos_predictions")
    op.drop_table("kronos_predictions")
    op.drop_index("ix_strategy_models_kind_active", table_name="strategy_models")
    op.drop_table("strategy_models")
