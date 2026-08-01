"""Normalise provider symbols, and rest symbols that never return data.

Two related fixes to the same waste.

**Notation.** The broker writes share classes as `BRK/A` and `BRK_B`; Yahoo
wants `BRK-A` and `BRK-B`. Untranslated, the symbol fetches nothing — which is
indistinguishable from a delisted company, so the instrument sat in the
catalogue with no candles and no explanation. Verified against the live API for
BRK, HEI, GEF, MKC and PBR before writing this.

Warrants (`*_WAR`) are deactivated rather than rewritten. Yahoo has no
dependable equivalent, and a warrant is not something this system should rank
or hold.

**Backoff.** Roughly a fifth of mapped instruments return nothing and always
will. Without a record of that they occupied a slot in every rotation forever,
so the nightly sweep spent that share of its budget re-learning it. The two new
columns let a symbol rest after repeated empty responses — a timestamp rather
than a flag, so a relisting is still picked up.

Revision ID: c5e91a7f34b2
Revises: a2c5e8f10b64
Create Date: 2026-07-31 18:30:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c5e91a7f34b2"
down_revision = "a2c5e8f10b64"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "market_data_mappings",
        sa.Column(
            "consecutive_empty_fetches",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "market_data_mappings",
        sa.Column("retry_after", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_market_data_mappings_retry_after",
        "market_data_mappings",
        ["retry_after"],
    )

    # Repair symbols already stored under the broker's notation. Scoped to
    # yfinance: the separator convention is that provider's, and rewriting a
    # symbol for a provider that wants the original would break a working
    # mapping to fix one that is not.
    op.execute(
        """
        UPDATE market_data_mappings
        SET provider_symbol = REPLACE(REPLACE(provider_symbol, '/', '-'), '_', '-')
        WHERE provider = 'yfinance'
          AND provider_symbol ~ '[/_]'
          AND provider_symbol NOT LIKE '%\\_WAR%'
        """
    )
    # A trailing separator carries no class; it is broker padding (`AVAV_`).
    op.execute(
        """
        UPDATE market_data_mappings
        SET provider_symbol = RTRIM(provider_symbol, '-')
        WHERE provider = 'yfinance' AND provider_symbol LIKE '%-'
        """
    )
    # Warrants: no Yahoo equivalent, so stop asking rather than ask wrongly.
    op.execute(
        """
        UPDATE market_data_mappings
        SET is_active = false,
            last_error = 'Warrant: no dependable yfinance symbol'
        WHERE provider = 'yfinance' AND provider_symbol LIKE '%\\_WAR%'
        """
    )


def downgrade() -> None:
    # The symbol rewrites are not reversed. They corrected a notation that never
    # fetched anything, so restoring them would only restore the fault — and the
    # original values are recoverable from `instruments.exchange_ticker`.
    op.drop_index("ix_market_data_mappings_retry_after", table_name="market_data_mappings")
    op.drop_column("market_data_mappings", "retry_after")
    op.drop_column("market_data_mappings", "consecutive_empty_fetches")
