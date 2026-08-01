"""daily mean reversion over scanner-ranked names

Three data changes, no schema change:

  * `sp500_mean_reversion` -> `mean_reversion` across configurations, runs and
    decisions. The strategy is index-agnostic; a kind naming an index it must
    not trade misleads whoever reads it next. Stored as VARCHAR, so this is a
    data update rather than an ALTER TYPE.

  * The strategy itself moves from 15-minute z-score/RSI to daily Bollinger
    Bands + RSI + ATR, and its universe is emptied — it is repopulated nightly
    from the scanner ranking by `worker.jobs.strategy.sync_strategy_universe`,
    so carrying the old single-instrument universe forward would leave a stale
    holdover until that job first runs.

  * `max_open_positions` 10 -> 20, so the top-20 candidate pool can actually be
    held. At 20 positions the existing `max_position_pct` of 10% is no longer
    the binding constraint; total exposure is.

Revision ID: d3b8f26a91c7
Revises: a91d6c47e5f2
Create Date: 2026-07-28 14:00:00.000000+00:00

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d3b8f26a91c7"
down_revision: str | None = "a91d6c47e5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEW_PARAMS = (
    '{"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, '
    '"rsi_oversold": 35.0, "atr_period": 14, "min_atr_pct": 0.02}'
)
_OLD_PARAMS = (
    '{"sma_period": 20, "zscore_entry": -2.0, "rsi_period": 14, '
    '"rsi_oversold": 30.0, "zscore_exit": 0.0}'
)


def upgrade() -> None:
    for table in ("strategy_configurations", "strategy_runs", "strategy_decisions"):
        op.execute(
            sa.text(
                f"UPDATE {table} SET kind = 'mean_reversion' "  # noqa: S608 - fixed table list
                f"WHERE kind = 'sp500_mean_reversion'"
            )
        )

    op.execute(
        sa.text(
            "UPDATE strategy_configurations SET "
            "name = 'Daily mean reversion', "
            "interval = '1d', "
            f"params = '{_NEW_PARAMS}'::jsonb, "
            "universe = '{\"instrument_ids\": []}'::jsonb "
            "WHERE kind = 'mean_reversion'"
        )
    )

    op.execute(sa.text("UPDATE risk_configurations SET max_open_positions = 20"))


def downgrade() -> None:
    op.execute(sa.text("UPDATE risk_configurations SET max_open_positions = 10"))
    op.execute(
        sa.text(
            "UPDATE strategy_configurations SET "
            "name = 'S&P 500 mean reversion', "
            "interval = '15m', "
            f"params = '{_OLD_PARAMS}'::jsonb "
            "WHERE kind = 'mean_reversion'"
        )
    )
    for table in ("strategy_configurations", "strategy_runs", "strategy_decisions"):
        op.execute(
            sa.text(
                f"UPDATE {table} SET kind = 'sp500_mean_reversion' "  # noqa: S608
                f"WHERE kind = 'mean_reversion'"
            )
        )
