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

from alembic import op

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


#: The six tables exactly as they stood at `ac4312114ea2`, read back from a
#: database built by replaying the migrations up to that point rather than
#: reconstructed by hand. The previous version of this downgrade was written
#: by hand, was never executed, and was wrong in every table: renamed columns
#: (`finished_at` for `completed_at`), missing columns, missing indexes,
#: missing foreign keys and a unique constraint under the wrong name. It only
#: has to be right for `alembic downgrade base` to reach zero, which is the
#: one thing nobody had ever asked it to do.
_RESTORED_SCHEMA: tuple[str, ...] = (
    (
        "CREATE TABLE earnings_events ( id uuid NOT NULL, instrument_id uuid NOT NULL, "
        "report_date date NOT NULL, surprise numeric(10,6), created_at timestamp with time zone "
        "DEFAULT now() NOT NULL, updated_at timestamp with time zone DEFAULT now() NOT NULL );"
    ),
    (
        "CREATE TABLE index_options_snapshots ( id uuid NOT NULL, as_of date NOT NULL, symbol "
        "character varying(16) NOT NULL, spot numeric(14,4), expiry_days integer, "
        "gamma_exposure numeric(16,4), skew_25delta numeric(8,4), atm_iv numeric(8,4), "
        "contracts_used integer, created_at timestamp with time zone DEFAULT now() NOT NULL, "
        "updated_at timestamp with time zone DEFAULT now() NOT NULL, gamma_tilt numeric(8,6), "
        "charm_exposure numeric(16,4), charm_tilt numeric(8,6) );"
    ),
    (
        "CREATE TABLE strategy_configurations ( kind character varying(32) NOT NULL, name "
        'character varying(120) NOT NULL, is_active boolean NOT NULL, "interval" character '
        "varying(8) NOT NULL, operating_mode character varying(24) NOT NULL, auto_execute "
        "boolean NOT NULL, params jsonb, universe jsonb, account_equity numeric(18,4), id uuid "
        "NOT NULL, created_at timestamp with time zone DEFAULT now() NOT NULL, updated_at "
        "timestamp with time zone DEFAULT now() NOT NULL, capital_allocation_pct numeric(12,6) "
        ");"
    ),
    (
        "CREATE TABLE strategy_decisions ( run_id uuid NOT NULL, configuration_id uuid, "
        "instrument_id uuid NOT NULL, kind character varying(32) NOT NULL, side character "
        "varying(8), conviction numeric(12,6) NOT NULL, outcome character varying(24) NOT NULL, "
        "reason text NOT NULL, metrics jsonb, proposal_id uuid, id uuid NOT NULL, created_at "
        "timestamp with time zone DEFAULT now() NOT NULL, updated_at timestamp with time zone "
        "DEFAULT now() NOT NULL );"
    ),
    (
        "CREATE TABLE strategy_models ( kind character varying(32) NOT NULL, feature_names "
        "jsonb NOT NULL, coefficients jsonb NOT NULL, intercept double precision NOT NULL, "
        "feature_means jsonb NOT NULL, feature_sds jsonb NOT NULL, scale_known jsonb NOT NULL, "
        "prior_means jsonb NOT NULL, prior_taus jsonb NOT NULL, shrinkage jsonb NOT NULL, "
        "standard_errors jsonb NOT NULL, low_shrinkage_clamp double precision, n_observations "
        "integer NOT NULL, positive_rate double precision, auc double precision, brier double "
        "precision, log_loss double precision, label_definition text NOT NULL, notes text, "
        "fitted_at timestamp with time zone, is_active boolean NOT NULL, model_version "
        "character varying(64), id uuid NOT NULL, created_at timestamp with time zone DEFAULT "
        "now() NOT NULL, updated_at timestamp with time zone DEFAULT now() NOT NULL );"
    ),
    (
        "CREATE TABLE strategy_runs ( configuration_id uuid, kind character varying(32) NOT "
        "NULL, status character varying(16) NOT NULL, started_at timestamp with time zone NOT "
        "NULL, completed_at timestamp with time zone, instruments_considered integer NOT NULL, "
        "signals_generated integer NOT NULL, proposals_created integer NOT NULL, executed "
        "integer NOT NULL, rejected integer NOT NULL, selection_reason character varying(200), "
        "error text, id uuid NOT NULL, created_at timestamp with time zone DEFAULT now() NOT "
        "NULL, updated_at timestamp with time zone DEFAULT now() NOT NULL );"
    ),
    "ALTER TABLE ONLY earnings_events ADD CONSTRAINT pk_earnings_events PRIMARY KEY (id);",
    (
        "ALTER TABLE ONLY index_options_snapshots ADD CONSTRAINT pk_index_options_snapshots "
        "PRIMARY KEY (id);"
    ),
    (
        "ALTER TABLE ONLY strategy_configurations ADD CONSTRAINT pk_strategy_configurations "
        "PRIMARY KEY (id);"
    ),
    "ALTER TABLE ONLY strategy_decisions ADD CONSTRAINT pk_strategy_decisions PRIMARY KEY (id);",
    "ALTER TABLE ONLY strategy_models ADD CONSTRAINT pk_strategy_models PRIMARY KEY (id);",
    "ALTER TABLE ONLY strategy_runs ADD CONSTRAINT pk_strategy_runs PRIMARY KEY (id);",
    (
        "ALTER TABLE ONLY earnings_events ADD CONSTRAINT uq_earnings_events_instrument_date "
        "UNIQUE (instrument_id, report_date);"
    ),
    (
        "ALTER TABLE ONLY index_options_snapshots ADD CONSTRAINT "
        "uq_index_options_snapshots_as_of_symbol UNIQUE (as_of, symbol);"
    ),
    (
        "ALTER TABLE ONLY strategy_configurations ADD CONSTRAINT "
        "uq_strategy_configurations_name UNIQUE (name);"
    ),
    (
        "CREATE INDEX ix_earnings_events_instrument_id ON earnings_events USING btree "
        "(instrument_id);"
    ),
    "CREATE INDEX ix_earnings_events_report_date ON earnings_events USING btree (report_date);",
    (
        "CREATE INDEX ix_index_options_snapshots_as_of ON index_options_snapshots USING btree "
        "(as_of);"
    ),
    (
        "CREATE INDEX ix_strategy_decisions_configuration_id ON strategy_decisions USING btree "
        "(configuration_id);"
    ),
    (
        "CREATE INDEX ix_strategy_decisions_instrument ON strategy_decisions USING btree "
        "(instrument_id);"
    ),
    "CREATE INDEX ix_strategy_decisions_outcome ON strategy_decisions USING btree (outcome);",
    "CREATE INDEX ix_strategy_decisions_run ON strategy_decisions USING btree (run_id);",
    (
        "CREATE INDEX ix_strategy_models_kind_active ON strategy_models USING btree (kind, "
        "is_active);"
    ),
    (
        "CREATE INDEX ix_strategy_runs_configuration_id ON strategy_runs USING btree "
        "(configuration_id);"
    ),
    "CREATE INDEX ix_strategy_runs_kind_started ON strategy_runs USING btree (kind, started_at);",
    (
        "ALTER TABLE ONLY earnings_events ADD CONSTRAINT "
        "fk_earnings_events_instrument_id_instruments FOREIGN KEY (instrument_id) REFERENCES "
        "instruments(id) ON DELETE CASCADE;"
    ),
    (
        "ALTER TABLE ONLY strategy_decisions ADD CONSTRAINT "
        "fk_strategy_decisions_configuration_id_strategy_configurations FOREIGN KEY "
        "(configuration_id) REFERENCES strategy_configurations(id) ON DELETE SET NULL;"
    ),
    (
        "ALTER TABLE ONLY strategy_decisions ADD CONSTRAINT "
        "fk_strategy_decisions_instrument_id_instruments FOREIGN KEY (instrument_id) REFERENCES "
        "instruments(id) ON DELETE CASCADE;"
    ),
    (
        "ALTER TABLE ONLY strategy_decisions ADD CONSTRAINT "
        "fk_strategy_decisions_proposal_id_trade_proposals FOREIGN KEY (proposal_id) REFERENCES "
        "trade_proposals(id) ON DELETE SET NULL;"
    ),
    (
        "ALTER TABLE ONLY strategy_decisions ADD CONSTRAINT "
        "fk_strategy_decisions_run_id_strategy_runs FOREIGN KEY (run_id) REFERENCES "
        "strategy_runs(id) ON DELETE CASCADE;"
    ),
    (
        "ALTER TABLE ONLY strategy_runs ADD CONSTRAINT "
        "fk_strategy_runs_configuration_id_strategy_configurations FOREIGN KEY "
        "(configuration_id) REFERENCES strategy_configurations(id) ON DELETE SET NULL;"
    ),
)


def downgrade() -> None:
    """Rebuild the six tables, so the migrations below this one can drop them.

    Not because anything will use them again — the code that gave these rows
    meaning is gone, and the upgrade above is still a data-losing one in the
    direction that matters. This exists so that `alembic downgrade base`
    reaches zero, which is the check that a migration chain is a chain rather
    than a one-way door.
    """
    for statement in _RESTORED_SCHEMA:
        op.execute(statement)
