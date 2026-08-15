"""Remove the Kronos prediction table.

Kronos is gone from the codebase, so its table goes with it. Eight measurements
across two universes never found it beating or adding to features that cost two
lines of arithmetic:

  * Seven comparisons against `discount_sma200` alone, which it never beat.
  * As a model feature on 276 point-in-time forecasts, it moved out-of-sample
    AUC from 0.5549 to 0.5563, with every coefficient's error bar spanning zero
    several times over and the sign backwards from the obvious story.
  * As a *filter* measured in money rather than ranking, three of its four
    framings pointed the wrong way: trades it expected to rise returned -0.043R
    while trades it expected to fall returned +0.062R.

The downgrade recreates the table but cannot recreate its contents — a forecast
is only reproducible by re-running the model that no longer ships here.

Revision ID: ac4312114ea2
Revises: 74e1367771fc
Create Date: 2026-08-12 09:41:15.238584+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ac4312114ea2"
down_revision: str | None = "74e1367771fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index(op.f("ix_kronos_predictions_as_of"), table_name="kronos_predictions")
    op.drop_index(op.f("ix_kronos_predictions_instrument_id"), table_name="kronos_predictions")
    op.drop_table("kronos_predictions")


def downgrade() -> None:
    op.create_table(
        "kronos_predictions",
        sa.Column("instrument_id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column("as_of", sa.DATE(), autoincrement=False, nullable=False),
        sa.Column("horizon_days", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "predicted_return",
            sa.NUMERIC(precision=12, scale=6),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "path_dispersion",
            sa.NUMERIC(precision=12, scale=6),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "prob_up", sa.NUMERIC(precision=12, scale=6), autoincrement=False, nullable=False
        ),
        sa.Column(
            "predicted_drawdown",
            sa.NUMERIC(precision=12, scale=6),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column("model_name", sa.VARCHAR(length=64), autoincrement=False, nullable=False),
        sa.Column("context_bars", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("sample_count", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("generation_ms", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "generated_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True
        ),
        sa.Column("id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
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
            name=op.f("uq_kronos_predictions_instrument_date_model"),
            postgresql_include=[],
            postgresql_nulls_not_distinct=False,
        ),
    )
    op.create_index(
        op.f("ix_kronos_predictions_instrument_id"),
        "kronos_predictions",
        ["instrument_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_kronos_predictions_as_of"), "kronos_predictions", ["as_of"], unique=False
    )
