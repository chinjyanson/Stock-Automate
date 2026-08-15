"""Widen the ATR stop from 2x to 5x, on measurement.

The first parameter in this system changed because a backtest said so rather
than because it was reasoned about.

A 2x ATR stop sits inside the noise the mean-reversion entry is trying to buy. A
forward-return test showed the entry's ingredients genuinely predict — RSI <= 30
precedes +3.08% over 20 days against a random-bar baseline, significant at every
horizon — while the assembled strategy earned nothing. The stop was ejecting
positions before the move it had correctly predicted arrived.

Widening it improved the profit factor monotonically on **both** halves of a
1,000-instrument replay, which almost nothing else tested has managed:

    stop     fit / confirm profit factor    win rate      worst losing streak
    2x       1.15 / 0.96                    52% / 48%     20.7R / 34.7R
    3x       1.20 / 0.96                    60% / 56%     12.5R / 24.0R
    5x       1.20 / 1.00                    63% / 60%      8.0R / 13.9R

The drawdown gain exceeds what the smaller trade count alone would produce.

**This does not put more money at risk.** Position size is
`risk_budget / stop_distance`, so a wider stop buys proportionally fewer shares
and the cash risked per trade is unchanged. What changes is that positions are
smaller and their stops are further away — fewer, larger percentage moves.

Applies to every strategy the risk engine sizes, not only mean reversion. The
index-timing sleeve has never been measured either way, so this is a considered
change for it rather than an evidenced one.

Only rows still holding the old default are moved; a value an operator has tuned
is left alone.

Revision ID: d5a92b7e4f16
Revises: b4d81f2c7a63
"""

from __future__ import annotations

from alembic import op

revision = "d5a92b7e4f16"
down_revision = "b4d81f2c7a63"
branch_labels = None
depends_on = None

_OLD = "2.0"
_NEW = "5.0"


def upgrade() -> None:
    op.alter_column("risk_configurations", "atr_stop_multiplier", server_default=None)
    # Deliberately conditional. An operator who has already tuned this away from
    # the old default made a decision, and a migration should not overrule it.
    op.execute(
        f"UPDATE risk_configurations SET atr_stop_multiplier = {_NEW} "
        f"WHERE atr_stop_multiplier = {_OLD}"
    )


def downgrade() -> None:
    op.execute(
        f"UPDATE risk_configurations SET atr_stop_multiplier = {_OLD} "
        f"WHERE atr_stop_multiplier = {_NEW}"
    )
