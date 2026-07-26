"""fundamentals-first scoring: reversal/quality scores + factor weights

Adds the two new per-result factor scores (reversal, quality) and the
configuration knobs for the absolute, fundamentals-first final score: a
`factor_weights` set and a `fundamentals_penalty`. Seeds existing configurations
with the defaults so the new blend takes effect.

Revision ID: b8e2c1f4a6d3
Revises: 7f3c9a1b2d4e
Create Date: 2026-07-23 10:15:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8e2c1f4a6d3"
down_revision: str | None = "7f3c9a1b2d4e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACTOR_WEIGHTS = (
    '{"fundamental_value": 0.35, "price_cheapness": 0.20, '
    '"reversal": 0.20, "quality": 0.15, "sector": 0.10}'
)


def upgrade() -> None:
    op.add_column(
        "scanner_results",
        sa.Column("reversal_score", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column(
        "scanner_results",
        sa.Column("quality_score", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column("scanner_configurations", sa.Column("factor_weights", sa.JSON(), nullable=True))
    op.add_column(
        "scanner_configurations",
        sa.Column(
            "fundamentals_penalty",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="0.1",
        ),
    )
    op.execute(f"UPDATE scanner_configurations SET factor_weights = '{_FACTOR_WEIGHTS}'::jsonb")


def downgrade() -> None:
    op.drop_column("scanner_configurations", "fundamentals_penalty")
    op.drop_column("scanner_configurations", "factor_weights")
    op.drop_column("scanner_results", "quality_score")
    op.drop_column("scanner_results", "reversal_score")
