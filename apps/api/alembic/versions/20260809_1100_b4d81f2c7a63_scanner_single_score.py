"""Scanner: one score, five groups.

The scanner computed two scores for months: a 100-point momentum core over six
categories, and a six-factor blend. Only the blend ever ranked anything — the
core's classification was computed and discarded, and five of its six category
columns were written, serialized, and rendered nowhere. This finishes the
migration that `0f75d2ac25fb` started: one score, and the five groups behind it
persisted on one 0-100 scale.

Momentum moves to the strategy layer, which evaluates nightly on fresh candles.
The scanner rotates 200-2000 names against a catalogue of ~20,000, so a stored
momentum reading is 10-100 days old when it is compared against a fresh one.

Irreversible in one respect, and the downgrade says so: the dropped columns are
recreated NOT NULL with a 0 backfill, because the values themselves cannot be
recovered. Re-run a scan after downgrading to repopulate them.

Revision ID: b4d81f2c7a63
Revises: c7e93a15b840
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b4d81f2c7a63"
down_revision = "c7e93a15b840"
branch_labels = None
depends_on = None

_RATIO = sa.Numeric(precision=12, scale=6)

#: Dropped from `scanner_results`, with the nullability they had, so the
#: downgrade can recreate them faithfully in shape if not in content.
_DROPPED_RESULT_COLUMNS: tuple[tuple[str, bool], ...] = (
    # (name, nullable)
    ("core_score", False),
    ("trend_score", False),
    ("momentum_score", False),
    ("risk_score", False),
    ("liquidity_score", False),
    ("positioning_score", False),
    ("reversal_score", True),
    ("fundamental_score", True),
    ("value_score", True),
)

#: The five-group default (scoring.DEFAULT_WEIGHTS). Written over whatever the
#: old six-category `weights` held, since the key names no longer exist.
_NEW_WEIGHTS = (
    '{"value": 30.0, "cheapness": 24.0, "insider": 18.0, "quality": 18.0, "sector": 10.0}'
)

_OLD_WEIGHTS = (
    '{"trend": 20.0, "momentum": 20.0, "risk": 15.0, '
    '"liquidity": 15.0, "positioning": 10.0, "sector": 20.0}'
)

_OLD_FACTOR_WEIGHTS = (
    '{"fundamental_value": 0.30, "price_cheapness": 0.29, "reversal": 0.05, '
    '"quality": 0.12, "sector": 0.09, "insider": 0.15}'
)


def upgrade() -> None:
    # -- scanner_results ----------------------------------------------------
    # The ranking index was on `core_score`, which no longer exists and was
    # never the column anything ordered by. Point it at the score that ranks.
    op.drop_index("ix_scanner_results_run_score", table_name="scanner_results")
    for name, _ in _DROPPED_RESULT_COLUMNS:
        op.drop_column("scanner_results", name)
    op.create_index(
        "ix_scanner_results_run_score", "scanner_results", ["run_id", "primary_score"]
    )

    # `sector_score` held category *points* (max 20) while every column beside
    # it held a 0-100 factor — a stock with no sector tag stored 10.0 and showed
    # it in the results table next to numbers on a different scale. It now holds
    # the group score, so historic rows are cleared rather than left to be read
    # on the wrong scale. NULL is the honest value: those rows never carried one.
    op.execute("UPDATE scanner_results SET sector_score = NULL")

    # -- scanner_configurations --------------------------------------------
    op.drop_column("scanner_configurations", "factor_weights")
    op.drop_column("scanner_configurations", "momentum_weight")
    op.drop_column("scanner_configurations", "value_weight")
    # The old keys (trend/momentum/positioning) name groups that are gone; a
    # config still holding them would silently fall back to defaults on every
    # lookup, so rewrite it rather than leave a stale dict in place.
    op.execute(f"UPDATE scanner_configurations SET weights = '{_NEW_WEIGHTS}'::jsonb")


def downgrade() -> None:
    op.drop_index("ix_scanner_results_run_score", table_name="scanner_results")
    for name, nullable in _DROPPED_RESULT_COLUMNS:
        # Non-null columns need a server_default to survive the backfill on an
        # existing table; it is dropped immediately afterwards so the model and
        # the schema agree. The 0 is a placeholder — the real values are gone.
        op.add_column(
            "scanner_results",
            sa.Column(name, _RATIO, nullable=nullable, server_default="0" if not nullable else None),
        )
        if not nullable:
            op.alter_column("scanner_results", name, server_default=None)
    op.create_index("ix_scanner_results_run_score", "scanner_results", ["run_id", "core_score"])

    op.add_column(
        "scanner_configurations", sa.Column("factor_weights", sa.dialects.postgresql.JSONB())
    )
    op.add_column(
        "scanner_configurations",
        sa.Column("momentum_weight", _RATIO, nullable=False, server_default="1"),
    )
    op.add_column(
        "scanner_configurations",
        sa.Column("value_weight", _RATIO, nullable=False, server_default="0"),
    )
    op.alter_column("scanner_configurations", "momentum_weight", server_default=None)
    op.alter_column("scanner_configurations", "value_weight", server_default=None)
    op.execute(f"UPDATE scanner_configurations SET weights = '{_OLD_WEIGHTS}'::jsonb")
    op.execute(f"UPDATE scanner_configurations SET factor_weights = '{_OLD_FACTOR_WEIGHTS}'::jsonb")
