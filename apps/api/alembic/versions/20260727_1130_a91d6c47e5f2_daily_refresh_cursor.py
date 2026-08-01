"""instruments: a dedicated cursor for the daily candle refresh

`refresh_daily_candles` ordered its batch by `last_scanned_at NULLS FIRST` — a
column only the scanner writes, and only when a scan successfully scores an
instrument. The refresh therefore read a cursor it never advanced, and picked the
same head of the queue on every run.

That head was the worst possible set: never-scanned instruments, overwhelmingly
delisted symbols that return no data. With no candles they can never reach the
scanner's 30-bar floor, so their `last_scanned_at` stays null and they stay at
the front indefinitely. On this database that was 2,592 zero-candle instruments
jammed ahead of the 12,346 that actually have prices to top up.

`last_refresh_attempt_at` is written on every *attempt*, so a symbol that returns
nothing still advances and the sweep progresses.

Nullable with no backfill on purpose: every existing row starts unattempted, so
the first cycles after this migration sweep the whole eligible set once, oldest
first, exactly as intended.

Revision ID: a91d6c47e5f2
Revises: e2f5b7c93a10
Create Date: 2026-07-27 11:30:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a91d6c47e5f2"
down_revision: str | None = "e2f5b7c93a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "instruments",
        sa.Column("last_refresh_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_instruments_last_refresh_attempt_at"),
        "instruments",
        ["last_refresh_attempt_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_instruments_last_refresh_attempt_at"), table_name="instruments")
    op.drop_column("instruments", "last_refresh_attempt_at")
