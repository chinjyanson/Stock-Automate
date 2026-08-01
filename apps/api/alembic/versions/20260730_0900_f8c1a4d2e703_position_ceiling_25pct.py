"""raise the single-position ceiling to 25%

A ceiling, not a target. Nothing in the engine aims for a position size: it is
determined by `risk_per_trade_pct / (atr_stop_multiplier * ATR)`, and this only
clips the result. For the current universe that formula lands at 14-16% of
equity, so the practical effect is a concentrated book of roughly seven names
rather than twenty — the 25% ceiling itself binds for only the calmest stock in
the universe.

Set deliberately with that trade-off understood: fewer, larger positions, and
real gap exposure on speculative small-caps where a 20-30% overnight move goes
straight through a stop only 4-7% away.

Revision ID: f8c1a4d2e703
Revises: 714ed9659c98
Create Date: 2026-07-30 09:00:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "f8c1a4d2e703"
down_revision: str | None = "714ed9659c98"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE risk_configurations SET max_position_pct = 0.25")


def downgrade() -> None:
    op.execute("UPDATE risk_configurations SET max_position_pct = 0.05")
