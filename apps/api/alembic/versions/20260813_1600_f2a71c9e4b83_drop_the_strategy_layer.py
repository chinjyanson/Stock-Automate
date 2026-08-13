"""Drop the strategy layer, and the tables that only fed it.

The individual-stock strategy, the index strategy and the per-stock backtest
replay are gone from the codebase, so six tables now have no model and no
writer. They are dropped rather than left behind: an orphaned table with rows in
it is a trap for the next person, who cannot tell stale data from live data
without reading the git history.

Ordered by dependency — decisions and runs reference configurations, so they go
first. `earnings_events` fed post-earnings drift, which only the stock strategy
consulted; the scanner's `earnings_growth` is a fundamentals-snapshot column and
is untouched.

**Downgrade recreates the tables but not their contents.** This is a data-losing
migration in the direction that matters, which is the honest position: the code
that gave those rows meaning no longer exists, so preserving them would preserve
bytes rather than information.

Revision ID: f2a71c9e4b83
Revises: ac4312114ea2
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f2a71c9e4b83"
down_revision = "ac4312114ea2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("strategy_decisions")
    op.drop_table("strategy_runs")
    op.drop_table("strategy_models")
    op.drop_table("strategy_configurations")
    op.drop_table("index_options_snapshots")
    op.drop_table("earnings_events")


def downgrade() -> None:
    op.create_table(
        "strategy_configurations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("interval", sa.String(length=8), nullable=False),
        sa.Column("auto_execute", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column("universe", postgresql.JSONB(), nullable=True),
        sa.Column("account_equity", sa.Numeric(20, 8), nullable=True),
        sa.Column("capital_allocation_pct", sa.Numeric(6, 4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "strategy_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "configuration_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("strategy_configurations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("considered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("signals", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_table(
        "strategy_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("strategy_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("instrument_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=True),
        sa.Column("conviction", sa.Numeric(6, 4), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("proposal_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "strategy_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("feature_names", postgresql.JSONB(), nullable=False),
        sa.Column("coefficients", postgresql.JSONB(), nullable=False),
        sa.Column("intercept", sa.Numeric(20, 10), nullable=False),
        sa.Column("feature_means", postgresql.JSONB(), nullable=False),
        sa.Column("feature_sds", postgresql.JSONB(), nullable=False),
        sa.Column("prior_means", postgresql.JSONB(), nullable=True),
        sa.Column("prior_taus", postgresql.JSONB(), nullable=True),
        sa.Column("shrinkage", postgresql.JSONB(), nullable=True),
        sa.Column("label_definition", sa.Text(), nullable=False),
        sa.Column("n_observations", sa.Integer(), nullable=False),
        sa.Column("auc", sa.Numeric(6, 4), nullable=True),
        sa.Column("brier", sa.Numeric(6, 4), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("fitted_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "index_options_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("spot", sa.Numeric(20, 8), nullable=True),
        sa.Column("gamma_exposure", sa.Numeric(30, 8), nullable=True),
        sa.Column("skew_25delta", sa.Numeric(10, 6), nullable=True),
        sa.Column("atm_iv", sa.Numeric(10, 6), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("as_of", "symbol", name="uq_index_options_as_of_symbol"),
    )
    op.create_table(
        "earnings_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("instrument_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("eps_actual", sa.Numeric(20, 8), nullable=True),
        sa.Column("eps_estimate", sa.Numeric(20, 8), nullable=True),
        sa.Column("surprise_pct", sa.Numeric(10, 4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("instrument_id", "report_date", name="uq_earnings_instrument_date"),
    )
