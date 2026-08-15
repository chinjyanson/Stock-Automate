"""Index options snapshots, earnings events, and the capital split.

Three additions, one per piece of tranche 2.

**index_options_snapshots.** An option chain is a snapshot with no history
behind it — a provider will say what is priced today and nothing about last
month. So the history is accumulated here from the day the job starts running,
one row per day, keyed by date so a re-run converges.

**earnings_events.** Report dates per instrument, unique on (instrument, date)
so repeated ingests are idempotent. Only the date comes from the provider; the
`surprise` column is filled in later from candles already stored, as the
abnormal return around the announcement.

**strategy_configurations.capital_allocation_pct.** Each strategy's share of the
account, 0..1. Applied before any risk cap, so every percentage limit becomes a
percentage of that sleeve rather than of the whole account. Null means the whole
of it, which is what every existing row gets — so behaviour is unchanged until a
split is actually configured.

Revision ID: a91c4f6d2e85
Revises: e6b2f47c8d13
Create Date: 2026-08-06 18:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a91c4f6d2e85"
down_revision = "e6b2f47c8d13"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_options_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("spot", sa.Numeric(precision=14, scale=4), nullable=True),
        sa.Column("expiry_days", sa.Integer(), nullable=True),
        sa.Column("gamma_exposure", sa.Numeric(precision=16, scale=4), nullable=True),
        sa.Column("skew_25delta", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("atm_iv", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("contracts_used", sa.Integer(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("as_of", name="uq_index_options_snapshots_as_of"),
    )
    op.create_index(
        "ix_index_options_snapshots_as_of", "index_options_snapshots", ["as_of"], unique=False
    )

    op.create_table(
        "earnings_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("surprise", sa.Numeric(precision=10, scale=6), nullable=True),
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
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id", "report_date", name="uq_earnings_events_instrument_date"
        ),
    )
    op.create_index("ix_earnings_events_instrument_id", "earnings_events", ["instrument_id"])
    op.create_index("ix_earnings_events_report_date", "earnings_events", ["report_date"])

    op.add_column(
        "strategy_configurations",
        sa.Column("capital_allocation_pct", sa.Numeric(precision=12, scale=6), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("strategy_configurations", "capital_allocation_pct")
    op.drop_index("ix_earnings_events_report_date", table_name="earnings_events")
    op.drop_index("ix_earnings_events_instrument_id", table_name="earnings_events")
    op.drop_table("earnings_events")
    op.drop_index("ix_index_options_snapshots_as_of", table_name="index_options_snapshots")
    op.drop_table("index_options_snapshots")
