"""Risk-engine entities (§9, §10).

Three tables carry the risk engine's state, and each exists because the
alternative is worse:

  * `RiskConfiguration` — every limit lives here, versioned and audited. No risk
    limit is a code constant, because "who loosened the drawdown limit, and
    when" is a question that gets asked after the loss (`docs/risk-model.md`).

  * `RiskHalt` — a halt is a *row*, not a raised exception. A halt that clears
    itself on process restart is not a halt; this one persists, is visible, and
    requires an explicit clearing action.

  * `TradeIntent` — the local record that guards against duplicate orders (§10).
    Trading 212 will not store our correlation token, so the only durable link
    between "we decided to buy" and "the broker shows a fill" is this row. It
    doubles as our side of reconciliation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import (
    Base,
    Money,
    Price,
    Quantity,
    Ratio,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import BrokerKind, HaltKind, HaltScope, OrderSide, TradeIntentStatus


class RiskConfiguration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A named, versioned set of risk limits (§9).

    Exactly one row is active at a time. Editing produces an audited change; the
    engine reads the active row on every evaluation, so a tightened limit takes
    effect on the next order without a redeploy.
    """

    __tablename__ = "risk_configurations"
    __table_args__ = (UniqueConstraint("name", name="uq_risk_configurations_name"),)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # -- Position sizing ----------------------------------------------------
    #: Fraction of equity risked on a single trade (the loss to the stop).
    risk_per_trade_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.01)
    #: ATR multiple that sets the stop distance (initial and trailing).
    atr_stop_multiplier: Mapped[Any] = mapped_column(Ratio, nullable=False, default=2.0)

    # -- Stop management ----------------------------------------------------
    #: Ratchet the protective stop upward as price rises (never down).
    trailing_stop_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: Exit a position held longer than this many calendar days. 0 disables the
    #: time stop.
    max_holding_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # -- Concentration / exposure caps -------------------------------------
    max_position_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.10)
    max_instrument_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.20)
    max_total_open_risk_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.06)
    max_portfolio_exposure_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=1.0)
    #: Absolute per-position cash cap. Null = no monetary cap beyond the
    #: percentage limits above.
    monetary_position_cap: Mapped[Any | None] = mapped_column(Money)

    # -- Count / frequency caps --------------------------------------------
    max_open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    max_trades_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    consecutive_loss_cooldown: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    # -- Loss / drawdown limits (drive halts) ------------------------------
    max_daily_realised_loss_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.03)
    max_portfolio_drawdown_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.15)

    # -- Live ceilings ------------------------------------------------------
    #: Absolute capital the live venue may deploy. Bounds the equity all live
    #: sizing scales against, so you cannot risk more than you decided to — the
    #: persistent replacement for the old per-session arming ceiling. Null = the
    #: broker's own equity is the only bound.
    max_live_capital: Mapped[Any | None] = mapped_column(Money)
    #: Realised live loss in one day that halts trading and drops back to paper.
    #: Null = no automatic daily stop.
    max_daily_loss: Mapped[Any | None] = mapped_column(Money)

    # -- Correlation --------------------------------------------------------
    correlation_benchmark_symbol: Mapped[str] = mapped_column(
        String(32), nullable=False, default="SPY"
    )
    correlation_window_short: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    correlation_window_long: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    #: Above this candidate-vs-benchmark correlation, exposure is treated as more
    #: S&P beta than diversification.
    correlation_threshold: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.80)
    #: When benchmark-correlated exposure already exceeds this fraction of the
    #: portfolio, a new correlated position is reduced or rejected.
    max_portfolio_sp500_pct: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0.50)

    #: Freeform extras so a limit can be added without a migration.
    extra: Mapped[dict[str, Any] | None] = mapped_column()

    def __repr__(self) -> str:
        return f"<RiskConfiguration {self.name} active={self.is_active}>"


class RiskHalt(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An active or historical trading halt (§9).

    Halts are states, not exceptions: recorded, visible, and requiring explicit
    clearing. `is_active` plus a partial-friendly index lets the engine ask "is
    anything blocking this order right now" cheaply.
    """

    __tablename__ = "risk_halts"
    __table_args__ = (
        Index("ix_risk_halts_active", "is_active"),
        Index("ix_risk_halts_scope_instrument", "scope", "instrument_id"),
    )

    kind: Mapped[HaltKind] = mapped_column(StrEnumType(HaltKind, 24), nullable=False)
    scope: Mapped[HaltScope] = mapped_column(
        StrEnumType(HaltScope, 16), nullable=False, default=HaltScope.GLOBAL
    )
    #: Set only for instrument-scoped halts.
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), index=True
    )
    #: Set only for strategy-scoped halts (free-form until strategies land).
    strategy_key: Mapped[str | None] = mapped_column(String(64))

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    activated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cleared_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<RiskHalt {self.kind}/{self.scope} active={self.is_active}>"


class TradeIntent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The durable record of an intent to trade, and the duplicate-order guard (§10).

    Created before an order is submitted and updated as its fate is learned. The
    `client_reference` is our correlation token: unique so a retried job cannot
    create a second intent, and carried on the broker request so reconciliation
    can match a fill back to the decision that caused it — even against a broker
    (Trading 212) that will not echo it.
    """

    __tablename__ = "trade_intents"
    __table_args__ = (
        UniqueConstraint("client_reference", name="uq_trade_intents_client_reference"),
        Index("ix_trade_intents_status", "status"),
        Index("ix_trade_intents_instrument", "instrument_id"),
    )

    proposal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("trade_proposals.id", ondelete="SET NULL"), index=True
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    broker: Mapped[BrokerKind] = mapped_column(StrEnumType(BrokerKind, 24), nullable=False)
    client_reference: Mapped[uuid.UUID] = mapped_column(nullable=False, default=uuid.uuid4)

    status: Mapped[TradeIntentStatus] = mapped_column(
        StrEnumType(TradeIntentStatus, 24),
        nullable=False,
        default=TradeIntentStatus.CREATED,
    )
    side: Mapped[OrderSide] = mapped_column(StrEnumType(OrderSide, 8), nullable=False)
    quantity: Mapped[Any] = mapped_column(Quantity, nullable=False)

    #: Learned from the broker once (if) the order reaches it.
    broker_order_id: Mapped[str | None] = mapped_column(String(128))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reconciled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    filled_quantity: Mapped[Any | None] = mapped_column(Quantity)
    filled_price: Mapped[Any | None] = mapped_column(Price)

    #: The protective stop placed after entry, and its broker order id. Both are
    #: rewritten when the stop trails upward.
    stop_price: Mapped[Any | None] = mapped_column(Price)
    stop_broker_order_id: Mapped[str | None] = mapped_column(String(128))

    #: Set when the position this intent opened is closed (stop hit, time stop,
    #: or emergency exit). A closed intent is not re-managed and feeds realised
    #: P/L at end of day.
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[Any | None] = mapped_column(Price)
    exit_reason: Mapped[str | None] = mapped_column(String(32))

    note: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<TradeIntent {self.side} {self.quantity} status={self.status}>"
