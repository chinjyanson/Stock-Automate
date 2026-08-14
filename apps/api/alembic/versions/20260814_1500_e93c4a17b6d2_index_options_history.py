"""Bring back the index-options history, so it can start accumulating.

`f2a71c9e4b83` dropped this table because nothing wrote to it — the strategy
that consumed it was gone, and an orphaned table with rows in it is a trap. That
was right at the time. It is being recreated because the question has changed:
dealer gamma is now the thing being *tested* rather than an input to something
else, and the test cannot start until the rows do.

The point worth understanding about this table is that it can only ever be
filled forwards. An option chain is published for today; per-strike open
interest is not retained anywhere free once the day passes. So unlike every
other series in this system, there is no backfill that recovers a missing week —
each day the job does not run is a permanent hole. That is the whole reason for
recreating the table before anything reads from it.

Two details carried over deliberately from the original, both of which cost
something to rediscover:

**Keyed by (as_of, symbol), not as_of.** The reading falls back from `^SPX` to
`SPY`, and gamma exposure scales with `open interest x spot`. SPX trades near
ten times SPY's level while SPY carries far more contracts, and those do not
cancel — the series steps by several times the day a fallback fires. Under a
date-only key one column would silently hold both scales, and a several-fold
step passes for a market move rather than announcing itself. A model fitted on
that history could never recover from it, and this table exists precisely to be
fitted on.

**Timestamps carry `server_default=now()`.** `TimestampMixin` expects the
database to fill them. `b8f3d02a5e91` created a sibling table without the
default; every ORM insert survived that and every Core bulk insert failed on it,
which is what `c1e75a3f8d24` had to go back and repair. Same mixin here, so the
same default here.

Revision ID: e93c4a17b6d2
Revises: c1e75a3f8d24
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "e93c4a17b6d2"
down_revision = "c1e75a3f8d24"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_options_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("spot", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("expiry_days", sa.Integer(), nullable=True),
        # Levels, kept for diagnosis and for reconstructing a tilt by hand.
        sa.Column("gamma_exposure", sa.Numeric(precision=16, scale=4), nullable=True),
        sa.Column("charm_exposure", sa.Numeric(precision=16, scale=4), nullable=True),
        # Net-to-gross ratios in [-1, +1]. These are what a model reads: the
        # contract multiplier and the open interest cancel out of a ratio, so
        # they mean the same thing whichever proxy produced the row.
        sa.Column("gamma_tilt", sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column("charm_tilt", sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column("skew_25delta", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("atm_iv", sa.Numeric(precision=8, scale=4), nullable=True),
        # How many strikes cleared the open-interest floor. A reading built from
        # six contracts deserves less trust than one from six hundred, and that
        # is invisible in the number itself.
        sa.Column("contracts_used", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "as_of", "symbol", name="uq_index_options_snapshots_as_of_symbol"
        ),
    )
    op.create_index(
        "ix_index_options_snapshots_as_of", "index_options_snapshots", ["as_of"]
    )


def downgrade() -> None:
    # Losing the rows is losing the history for good — see the docstring. The
    # downgrade is kept honest rather than kept safe: there is no way to make
    # dropping this table recoverable, and pretending otherwise would be worse.
    op.drop_index("ix_index_options_snapshots_as_of", table_name="index_options_snapshots")
    op.drop_table("index_options_snapshots")
