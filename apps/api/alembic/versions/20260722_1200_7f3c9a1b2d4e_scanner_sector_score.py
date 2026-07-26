"""scanner sector score + rebalanced weights

Adds the `sector` category to the scanner: a new nullable `sector_score` column
on scanner_results, and a rebalanced weight set on existing scanner
configurations (the five prior categories were trimmed to make room for
`sector` while keeping the 0-100 scale, so the 75/60 thresholds still hold).

Revision ID: 7f3c9a1b2d4e
Revises: 32211713f911
Create Date: 2026-07-22 12:00:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7f3c9a1b2d4e"
down_revision: str | None = "32211713f911"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The rebalanced defaults (sum to 100) — mirror scoring.DEFAULT_WEIGHTS.
_NEW_WEIGHTS = (
    '{"trend": 20.0, "momentum": 20.0, "risk": 15.0, '
    '"liquidity": 15.0, "positioning": 10.0, "sector": 20.0}'
)
_OLD_WEIGHTS = (
    '{"trend": 25.0, "momentum": 20.0, "risk": 20.0, "liquidity": 20.0, "positioning": 15.0}'
)


def upgrade() -> None:
    op.add_column(
        "scanner_results",
        sa.Column("sector_score", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    # Give every configuration the rebalanced weights so the core score keeps
    # summing to 100 once the sector category contributes.
    op.execute(f"UPDATE scanner_configurations SET weights = '{_NEW_WEIGHTS}'::jsonb")


def downgrade() -> None:
    op.execute(f"UPDATE scanner_configurations SET weights = '{_OLD_WEIGHTS}'::jsonb")
    op.drop_column("scanner_results", "sector_score")
