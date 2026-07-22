"""Strategy entities (§8).

A strategy is a configured, reproducible opinion about when to trade. Three tables
carry it, mirroring the scanner's shape (config / run / result):

  * `StrategyConfiguration` — a named, versioned, audited strategy instance: its
    kind, the universe it watches, its parameters, and the mode it runs in. One
    row per configured strategy; editing is an audited change.

  * `StrategyRun` — one evaluation pass over the universe, with counts, so "what
    did the trend strategy do at 21:00" is answerable from a row.

  * `StrategyDecision` — one signal and what became of it. A signal is not a
    trade: it becomes a proposal, which the risk engine can still refuse. The
    outcome column records that, so "the strategy wanted to but risk said no" is
    visible rather than silent (§9).
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import (
    Base,
    Money,
    Ratio,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import (
    Interval,
    OperatingMode,
    OrderSide,
    StrategyDecisionOutcome,
    StrategyKind,
    StrategyRunStatus,
)


class StrategyConfiguration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A named, versioned strategy instance (§8)."""

    __tablename__ = "strategy_configurations"
    __table_args__ = (UniqueConstraint("name", name="uq_strategy_configurations_name"),)

    kind: Mapped[StrategyKind] = mapped_column(StrEnumType(StrategyKind, 32), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    #: Inactive by default: a fresh install must never auto-trade before a human
    #: has mapped a universe and turned it on.
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Bar size the strategy reads from the local store.
    interval: Mapped[Interval] = mapped_column(
        StrEnumType(Interval, 8), nullable=False, default=Interval.D1
    )
    #: Trading mode. Default is internal paper — never live (§7).
    operating_mode: Mapped[OperatingMode] = mapped_column(
        StrEnumType(OperatingMode, 24), nullable=False, default=OperatingMode.INTERNAL_PAPER
    )
    #: When true (paper only), an approved signal is executed immediately rather
    #: than waiting for a human. Ignored for live modes, which always require a
    #: human approval upstream.
    auto_execute: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    #: Strategy parameters (thresholds, lookbacks) — JSON so tuning needs no
    #: migration.
    params: Mapped[dict[str, Any] | None] = mapped_column()
    #: What the strategy watches: {"instrument_ids": [...]} and/or, for the pie,
    #: {"weights": {instrument_id: target_fraction}}.
    universe: Mapped[dict[str, Any] | None] = mapped_column()
    #: Capital the strategy sizes against. Null = use the broker account equity.
    account_equity: Mapped[Any | None] = mapped_column(Money)

    runs: Mapped[list[StrategyRun]] = relationship(back_populates="configuration")

    def __repr__(self) -> str:
        return f"<StrategyConfiguration {self.name} ({self.kind}) active={self.is_active}>"


class StrategyRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One evaluation pass of a strategy over its universe."""

    __tablename__ = "strategy_runs"
    __table_args__ = (Index("ix_strategy_runs_kind_started", "kind", "started_at"),)

    configuration_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("strategy_configurations.id", ondelete="SET NULL"), index=True
    )
    kind: Mapped[StrategyKind] = mapped_column(StrEnumType(StrategyKind, 32), nullable=False)
    status: Mapped[StrategyRunStatus] = mapped_column(
        StrEnumType(StrategyRunStatus, 16), nullable=False, default=StrategyRunStatus.RUNNING
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    instruments_considered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    signals_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    proposals_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    executed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    selection_reason: Mapped[str | None] = mapped_column(String(200))
    error: Mapped[str | None] = mapped_column(Text)

    configuration: Mapped[StrategyConfiguration | None] = relationship(back_populates="runs")
    decisions: Mapped[list[StrategyDecision]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class StrategyDecision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One strategy signal and its outcome, with provenance (§8)."""

    __tablename__ = "strategy_decisions"
    __table_args__ = (
        Index("ix_strategy_decisions_run", "run_id"),
        Index("ix_strategy_decisions_instrument", "instrument_id"),
        Index("ix_strategy_decisions_outcome", "outcome"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("strategy_runs.id", ondelete="CASCADE"), nullable=False
    )
    configuration_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("strategy_configurations.id", ondelete="SET NULL"), index=True
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[StrategyKind] = mapped_column(StrEnumType(StrategyKind, 32), nullable=False)
    side: Mapped[OrderSide] = mapped_column(StrEnumType(OrderSide, 8), nullable=False)
    conviction: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0)
    outcome: Mapped[StrategyDecisionOutcome] = mapped_column(
        StrEnumType(StrategyDecisionOutcome, 24), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column()

    #: Set when the signal became a proposal.
    proposal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("trade_proposals.id", ondelete="SET NULL")
    )

    run: Mapped[StrategyRun] = relationship(back_populates="decisions")

    def __repr__(self) -> str:
        return f"<StrategyDecision {self.kind} {self.side} {self.outcome}>"
