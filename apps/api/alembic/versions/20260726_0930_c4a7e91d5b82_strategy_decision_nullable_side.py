"""strategy decisions: nullable side, for instruments that were never evaluated

The engine now records a SKIPPED decision when a strategy could not evaluate an
instrument for want of candle history. Such a row has no side: nothing was
decided. Previously the column was NOT NULL, which left only two options —
fabricate a side, or record nothing — and recording nothing is why a run that
saw no data looked identical to a run that found no setup.

Widening a constraint, so existing rows are untouched and the downgrade is only
safe while no null sides exist; it fails loudly if any do rather than
inventing values.

Revision ID: c4a7e91d5b82
Revises: b8e2c1f4a6d3
Create Date: 2026-07-26 09:30:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4a7e91d5b82"
down_revision: str | None = "b8e2c1f4a6d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "strategy_decisions",
        "side",
        existing_type=sa.String(length=8),
        nullable=True,
    )


def downgrade() -> None:
    # Refuse rather than guess. A null side means "not evaluated"; there is no
    # BUY or SELL that honestly stands in for it, so the operator must decide
    # what to do with those rows before narrowing the column again.
    count = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM strategy_decisions WHERE side IS NULL"))
        .scalar_one()
    )
    if count:
        raise RuntimeError(
            f"{count} strategy_decisions rows have a null side (not-evaluated records). "
            "Delete or reclassify them before downgrading; this migration will not "
            "invent a side for them."
        )
    op.alter_column(
        "strategy_decisions",
        "side",
        existing_type=sa.String(length=8),
        nullable=False,
    )
