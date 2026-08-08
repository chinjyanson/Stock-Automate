"""News sentiment snapshots, the risk gate's thresholds, and statement fields.

Three additions, the two halves of tranche 3 plus the config the gate reads.

**sentiment_snapshots.** One row per instrument per day. News is the most
perishable input in the system — a feed cannot be re-queried for what it said
last Tuesday — so the row is the only record there will ever be of that day's
tone. Both readings live side by side and neither is derived from the other:
`polarity` is ours, computed locally from headlines with the Loughran-McDonald
lexicon, while the `provider_*` columns are Finnhub's own scoring, which is
premium-gated and usually absent. Keeping them apart is what lets a consumer say
which one it acted on.

**risk_configurations sentiment thresholds.** `sentiment_reduction_threshold`
defaults to -0.35 rather than 0: ordinary news skews mildly negative (failures
are reported, functioning is not), so a threshold at zero would shrink nearly
every trade and stop carrying information. `sentiment_veto_threshold` is
nullable and defaults to null — the hard refusal is off until someone turns it
on, which is the right default for a signal that counts words and cannot read a
headline. `sentiment_max_age_days` is the staleness beyond which the gate stands
down entirely, because absence of news must never read as bad news.

**fundamental_snapshots statement fields.** The inputs a discounted cash flow
needs, all nullable. Read from the `.info` payload the enrichment sweep already
fetches, so ingesting them costs no additional provider calls. Stored ahead of
any DCF existing, deliberately: whether the free-tier coverage justifies
building one is a question only the accumulated data can answer.

Revision ID: c7e93a15b840
Revises: a91c4f6d2e85
Create Date: 2026-08-08 10:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c7e93a15b840"
down_revision = "a91c4f6d2e85"
branch_labels = None
depends_on = None

#: Cash-flow and balance-sheet columns, with the numeric type each maps to.
#: Amounts use Money (18,4), the share count uses Quantity, per-share book value
#: uses Price (18,8), and the two ratios use Ratio (12,6).
_STATEMENT_COLUMNS = (
    ("free_cash_flow", sa.Numeric(precision=18, scale=4)),
    ("operating_cash_flow", sa.Numeric(precision=18, scale=4)),
    ("total_debt", sa.Numeric(precision=18, scale=4)),
    ("total_cash", sa.Numeric(precision=18, scale=4)),
    ("shares_outstanding", sa.Numeric(precision=24, scale=8)),
    ("ebitda", sa.Numeric(precision=18, scale=4)),
    ("enterprise_value", sa.Numeric(precision=18, scale=4)),
    ("book_value_per_share", sa.Numeric(precision=18, scale=8)),
    ("return_on_equity", sa.Numeric(precision=12, scale=6)),
    ("current_ratio", sa.Numeric(precision=12, scale=6)),
)


def upgrade() -> None:
    op.create_table(
        "sentiment_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("instrument_id", sa.Uuid(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("provider_symbol", sa.String(length=64), nullable=True),
        sa.Column("polarity", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("uncertainty", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("positive_words", sa.Integer(), nullable=False),
        sa.Column("negative_words", sa.Integer(), nullable=False),
        sa.Column("headline_count", sa.Integer(), nullable=False),
        sa.Column("provider_score", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("provider_bullish_pct", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("provider_buzz", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
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
            "instrument_id", "as_of", name="uq_sentiment_snapshots_instrument_asof"
        ),
    )
    op.create_index("ix_sentiment_snapshots_instrument_id", "sentiment_snapshots", ["instrument_id"])
    op.create_index("ix_sentiment_snapshots_as_of", "sentiment_snapshots", ["as_of"])

    # Server defaults backfill the existing rows, then are dropped: the model
    # declares these Python-side, and leaving a database default behind makes
    # `alembic check` report drift against it on every subsequent run.
    op.add_column(
        "risk_configurations",
        sa.Column(
            "sentiment_reduction_threshold",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="-0.35",
        ),
    )
    op.add_column(
        "risk_configurations",
        sa.Column("sentiment_veto_threshold", sa.Numeric(precision=12, scale=6), nullable=True),
    )
    op.add_column(
        "risk_configurations",
        sa.Column(
            "sentiment_max_age_days", sa.Integer(), nullable=False, server_default="3"
        ),
    )
    op.alter_column("risk_configurations", "sentiment_reduction_threshold", server_default=None)
    op.alter_column("risk_configurations", "sentiment_max_age_days", server_default=None)

    for name, column_type in _STATEMENT_COLUMNS:
        op.add_column("fundamental_snapshots", sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    for name, _ in _STATEMENT_COLUMNS:
        op.drop_column("fundamental_snapshots", name)

    op.drop_column("risk_configurations", "sentiment_max_age_days")
    op.drop_column("risk_configurations", "sentiment_veto_threshold")
    op.drop_column("risk_configurations", "sentiment_reduction_threshold")

    op.drop_index("ix_sentiment_snapshots_as_of", table_name="sentiment_snapshots")
    op.drop_index("ix_sentiment_snapshots_instrument_id", table_name="sentiment_snapshots")
    op.drop_table("sentiment_snapshots")
