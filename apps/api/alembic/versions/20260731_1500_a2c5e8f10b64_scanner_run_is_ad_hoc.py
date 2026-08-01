"""Mark explicitly-selected scanner runs as ad hoc.

The strategy universe is synced from "the latest scanner run". Rescanning a
single instrument on demand would make that run the latest, collapsing the
mean-reversion universe to one name. This flag separates a one-off rescan from
the nightly rotating scan so only the latter defines what the strategy watches.

Existing rows default to False: every run recorded so far came from the
rotation or from a full manual scan, both of which are legitimate rankings.

Revision ID: a2c5e8f10b64
Revises: f7184b40649a
Create Date: 2026-07-31 15:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a2c5e8f10b64"
down_revision = "f7184b40649a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scanner_runs",
        sa.Column(
            "is_ad_hoc",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    # "The newest result for each instrument" is now asked on every listing, and
    # answering it by sorting the whole table costs more every night. The
    # composite serves the window's partition and order directly, and subsumes
    # the plain instrument_id index it replaces.
    op.create_index(
        "ix_scanner_results_instrument_recent",
        "scanner_results",
        ["instrument_id", sa.text("created_at DESC")],
    )
    op.drop_index("ix_scanner_results_instrument", table_name="scanner_results")


def downgrade() -> None:
    op.create_index("ix_scanner_results_instrument", "scanner_results", ["instrument_id"])
    op.drop_index("ix_scanner_results_instrument_recent", table_name="scanner_results")
    op.drop_column("scanner_runs", "is_ad_hoc")
