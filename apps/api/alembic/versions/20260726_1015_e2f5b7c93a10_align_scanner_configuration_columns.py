"""align scanner_configurations with the model: jsonb factor_weights, no server default

Two corrections to b8e2c1f4a6d3, both drift that `alembic check` catches:

  * `factor_weights` was created as `sa.JSON()` where the model declares
    `dict[str, Any]`, which the Base's type_annotation_map maps to
    `JSONBOrJSON` — JSONB on PostgreSQL. It was the only `json` column in the
    schema; every sibling, including `thresholds` on this same table, is
    `jsonb`. Not cosmetic: `json` stores the raw text and supports none of the
    containment operators or GIN indexing the rest of the schema relies on.

  * `fundamentals_penalty` kept the `server_default='0.1'` that b8e2c1f4a6d3
    used to backfill existing rows when adding a NOT NULL column. That default
    had done its job; leaving it in place meant the database carried a default
    the model does not declare. The Python-side `default=0.1` continues to
    supply the value for new rows through the ORM.

Revision ID: e2f5b7c93a10
Revises: c4a7e91d5b82
Create Date: 2026-07-26 10:15:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e2f5b7c93a10"
down_revision: str | None = "c4a7e91d5b82"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # postgresql_using is required: there is no implicit json -> jsonb cast.
    op.alter_column(
        "scanner_configurations",
        "factor_weights",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="factor_weights::jsonb",
    )
    op.alter_column(
        "scanner_configurations",
        "fundamentals_penalty",
        existing_type=sa.Numeric(precision=12, scale=6),
        existing_nullable=False,
        server_default=None,
    )


def downgrade() -> None:
    op.alter_column(
        "scanner_configurations",
        "fundamentals_penalty",
        existing_type=sa.Numeric(precision=12, scale=6),
        existing_nullable=False,
        server_default=sa.text("0.1"),
    )
    op.alter_column(
        "scanner_configurations",
        "factor_weights",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.JSON(),
        existing_nullable=True,
        postgresql_using="factor_weights::json",
    )
