"""Persist the risk engine's verdict alongside the trade intent.

`RiskDecision` was transient. Only its `reason` string survived, and only on
rejection — so once a position was sized there was no record of *which* cap
bound it, or what the correlation, stress and regime readings were at the time.

That was tolerable while the caps were few and static. It is not now: the
correlation, rate-sensitivity and whole-book stress caps all need tuning against
evidence, and "why was this position small?" is a question that gets asked months
later, when nothing but the record remains.

One JSONB column rather than a table of typed columns, deliberately: the set of
caps changes as limits are added, and a schema migration per cap would guarantee
the record drifts out of date with the engine. Rejections write no trade intent
at all, so those go to the audit log's payload instead — the same dict, from the
same `RiskDecision.as_record`.

Revision ID: e6b2f47c8d13
Revises: d4a8c1e5b729
Create Date: 2026-08-06 16:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "e6b2f47c8d13"
down_revision = "d4a8c1e5b729"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trade_intents",
        sa.Column(
            "risk_evaluation",
            postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("trade_intents", "risk_evaluation")
