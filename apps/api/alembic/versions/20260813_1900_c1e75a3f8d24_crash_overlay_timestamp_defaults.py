"""Give crash_overlay_readings the timestamp defaults its model assumes.

`TimestampMixin` declares `created_at` and `updated_at` with
`server_default=func.now()`, so the *database* is expected to fill them.
b8f3d02a5e91 created the columns `NOT NULL` and without that default, which
every ORM insert survives — SQLAlchemy sends a value it got from the mixin — and
every Core bulk insert fails on, because there is nothing to supply one.

Nothing caught it: the integration tests build rows through the ORM, and the
only Core path is the twenty-year backfill, which is exactly what fell over the
first time it ran for real. Aligning the schema with the model is the fix; the
service also now sends both columns explicitly, which is belt and braces and
additionally makes `updated_at` move on re-runs (an `onupdate=` does not fire
for `ON CONFLICT DO UPDATE`).

Revision ID: c1e75a3f8d24
Revises: b8f3d02a5e91
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c1e75a3f8d24"
down_revision = "b8f3d02a5e91"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in ("created_at", "updated_at"):
        op.alter_column(
            "crash_overlay_readings",
            column,
            server_default=sa.func.now(),
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=False,
        )


def downgrade() -> None:
    for column in ("created_at", "updated_at"):
        op.alter_column(
            "crash_overlay_readings",
            column,
            server_default=None,
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=False,
        )
