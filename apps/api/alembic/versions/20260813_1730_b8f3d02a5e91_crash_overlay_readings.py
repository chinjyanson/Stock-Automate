"""The crash overlay's daily record.

One row per trading day: the eleven features, the model's probability, the
trigger it was ranked against, and the exposure that resulted. Backfilled to
2006 on first run and appended nightly.

Everything but `as_of`, `is_warning` and `days_out` is nullable, because the
early rows genuinely have nothing to record — the trigger needs two years of
model output before a percentile means anything, and the overlay holds fully
invested until it does.

Revision ID: b8f3d02a5e91
Revises: f2a71c9e4b83
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b8f3d02a5e91"
down_revision = "f2a71c9e4b83"
branch_labels = None
depends_on = None

_FEATURES = (
    ("vix", 12, 4),
    ("vix_term_structure", 12, 6),
    ("credit_spread", 12, 6),
    ("hyg_tlt", 12, 6),
    ("hyg_lqd", 12, 6),
    ("skew", 12, 4),
    ("small_cap_rs", 12, 6),
    ("realised_vol", 12, 6),
    ("vol_of_vol", 12, 6),
    ("drawdown_from_high", 12, 6),
    ("insider_rank", 12, 6),
)


def upgrade() -> None:
    op.create_table(
        "crash_overlay_readings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("index_close", sa.Numeric(20, 8), nullable=True),
        *(sa.Column(name, sa.Numeric(p, s), nullable=True) for name, p, s in _FEATURES),
        sa.Column("probability", sa.Numeric(10, 8), nullable=True),
        sa.Column("trigger", sa.Numeric(10, 8), nullable=True),
        sa.Column("is_warning", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("target_exposure", sa.Numeric(6, 4), nullable=True),
        sa.Column("exit_price", sa.Numeric(20, 8), nullable=True),
        sa.Column("days_out", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("as_of", name="uq_crash_overlay_readings_as_of"),
    )
    op.create_index(
        "ix_crash_overlay_readings_as_of", "crash_overlay_readings", ["as_of"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_crash_overlay_readings_as_of", table_name="crash_overlay_readings")
    op.drop_table("crash_overlay_readings")
