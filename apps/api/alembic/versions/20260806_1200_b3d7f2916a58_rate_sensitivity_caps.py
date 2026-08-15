"""Rate-sensitivity concentration cap on risk configurations.

The correlation stage already prevents the book quietly becoming one S&P bet.
It cannot see the other version of the same mistake: a portfolio of REITs,
utilities and long-duration growth names looks diversified by sector and by
pairwise correlation, and is nonetheless a single position on bond yields.

Two columns. `rate_proxy_symbol` names the reference series — a bond *price*
series, deliberately not a yield index like ^TNX, because yields move inversely
to prices and a yield series would invert the meaning of every correlation
computed against it without raising anything.
`max_portfolio_rate_sensitive_pct` mirrors `max_portfolio_sp500_pct`: the
fraction of equity that may sit in rate-sensitive holdings before a new one is
cut. Defaulted to the same 0.50, since it is the same kind of judgement.

Existing rows take the defaults, which reproduces current behaviour on any book
that is not already rate-concentrated.

Revision ID: b3d7f2916a58
Revises: c5e91a7f34b2
Create Date: 2026-08-06 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b3d7f2916a58"
down_revision = "c5e91a7f34b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "risk_configurations",
        sa.Column(
            "rate_proxy_symbol",
            sa.String(length=32),
            nullable=False,
            server_default="IEF",
        ),
    )
    op.add_column(
        "risk_configurations",
        sa.Column(
            "max_portfolio_rate_sensitive_pct",
            sa.Numeric(precision=12, scale=6),
            nullable=False,
            server_default="0.5",
        ),
    )
    # The server defaults existed only to backfill existing rows on a NOT NULL
    # add. Drop them again so the model stays the single source of truth for
    # what a new configuration starts at.
    op.alter_column("risk_configurations", "rate_proxy_symbol", server_default=None)
    op.alter_column("risk_configurations", "max_portfolio_rate_sensitive_pct", server_default=None)


def downgrade() -> None:
    op.drop_column("risk_configurations", "max_portfolio_rate_sensitive_pct")
    op.drop_column("risk_configurations", "rate_proxy_symbol")
