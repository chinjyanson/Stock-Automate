"""Credit spread and realised volatility on market-regime snapshots.

Two additions to the fast signals that scale position size.

**Credit spread** (ICE BofA US high-yield option-adjusted spread) reprices ahead
of equity and, unlike the yield curve already recorded here, moves on a
timescale a position size can respond to — which is why it counts toward the
factor where the curve deliberately does not.

**Realised volatility** is the 20-day annualised move of the S&P proxy, computed
from candles already loaded for the 200-day trend read. VIX is the market's
*expectation* of volatility; this is what actually happened. A slow, orderly
decline can leave VIX untroubled while the tape plainly deteriorates, and that
divergence is the case this catches.

Both nullable: a failed FRED call or a proxy without enough history must record
as unknown, never as zero, since zero here would read as "calm".

Note for anyone reading the history back: the deduction constants in
`app.services.market_regime` were reduced in the same change (0.15/0.07 →
0.12/0.06) so that adding two inputs did not quietly make the system more
defensive. Risk factors recorded before this migration were produced under the
old cut sizes and are not directly comparable with later ones.

Revision ID: d4a8c1e5b729
Revises: b3d7f2916a58
Create Date: 2026-08-06 14:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "d4a8c1e5b729"
down_revision = "b3d7f2916a58"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "market_regime_snapshots",
        sa.Column("credit_spread", sa.Numeric(precision=8, scale=3), nullable=True),
    )
    op.add_column(
        "market_regime_snapshots",
        sa.Column("sp500_realised_vol", sa.Numeric(precision=8, scale=4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("market_regime_snapshots", "sp500_realised_vol")
    op.drop_column("market_regime_snapshots", "credit_spread")
