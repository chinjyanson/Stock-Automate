"""Scanner entities (§6, §15).

The vocabulary here is deliberate and constrained (§0). A result is a *screening
candidate* or *watchlist candidate* or *does not pass the screen* — never a
"good investment". The column names and enums enforce that language so it cannot
drift in the UI.

A `ScannerResult` records not just a score but its *provenance*: which candles
fed it, how complete they were, how fresh, and which signals drove the number.
A score without that is an opinion; with it, it is a reproducible measurement.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
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
    Price,
    Quantity,
    Ratio,
    StrEnumType,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.enums import OrderSide


class Classification(StrEnum):
    """Screening outcome bands (§6). The thresholds are configurable; these are
    the labels the bands map to."""

    SCREENING_CANDIDATE = "screening_candidate"  # 75-100 by default
    WATCHLIST_CANDIDATE = "watchlist_candidate"  # 60-74
    DOES_NOT_PASS = "does_not_pass"  # below 60

    @property
    def label(self) -> str:
        return {
            Classification.SCREENING_CANDIDATE: "Screening candidate",
            Classification.WATCHLIST_CANDIDATE: "Watchlist candidate",
            Classification.DOES_NOT_PASS: "Does not pass the screen",
        }[self]


class ScannerRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProposalStatus(StrEnum):
    """Lifecycle of a proposed trade awaiting a human decision (§6)."""

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    #: Approved by the user, then blocked by a pre-execution revalidation (stale
    #: data, risk limit, no longer tradable). A distinct terminal state because
    #: "the user said yes but the system said no" must be visible, not silent.
    REJECTED_BY_RISK = "rejected_by_risk"


class ScannerConfiguration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A named, versioned scanner configuration (§6).

    Universe filters, score weights and thresholds all live here so a scan is
    reproducible: re-running with the same configuration over the same candles
    yields the same result. Weights and thresholds are JSON so they can be tuned
    without a migration.
    """

    __tablename__ = "scanner_configurations"
    __table_args__ = (UniqueConstraint("name", name="uq_scanner_configurations_name"),)

    name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # -- Universe filters (§6) ---------------------------------------------
    included_exchanges: Mapped[dict[str, Any] | None] = mapped_column()  # {"mics": [...]}
    included_currencies: Mapped[dict[str, Any] | None] = mapped_column()
    include_stocks: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    include_etfs: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    min_price: Mapped[Any | None] = mapped_column(Price)
    min_avg_traded_value: Mapped[Any | None] = mapped_column(Money)
    max_instruments_per_scan: Mapped[int] = mapped_column(Integer, nullable=False, default=200)
    trading212_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # -- Scoring (§6) -------------------------------------------------------
    #: Category weights, summing to 100. Trend 25 / Momentum 20 / Risk 20 /
    #: Liquidity 20 / Positioning 15 by default.
    weights: Mapped[dict[str, Any] | None] = mapped_column()
    #: Band thresholds, e.g. {"screening": 75, "watchlist": 60}.
    thresholds: Mapped[dict[str, Any] | None] = mapped_column()
    benchmark_symbol: Mapped[str | None] = mapped_column(String(32), default="SPY")

    #: How much the momentum core score vs the value score drive the *primary*
    #: score that classification and ranking use. Default is momentum-primary
    #: (1.0 / 0.0). A "buy low" configuration sets value_weight high and
    #: momentum_weight low — value becomes the lead metric, momentum a secondary
    #: one that still shows on every result. Both scores are always computed and
    #: stored regardless; these only decide which leads.
    momentum_weight: Mapped[Any] = mapped_column(Ratio, nullable=False, default=1)
    value_weight: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0)

    runs: Mapped[list[ScannerRun]] = relationship(back_populates="configuration")


class ScannerRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One execution of the scanner over a slice of the universe."""

    __tablename__ = "scanner_runs"
    __table_args__ = (Index("ix_scanner_runs_status_started", "status", "started_at"),)

    configuration_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("scanner_configurations.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[ScannerRunStatus] = mapped_column(
        StrEnumType(ScannerRunStatus, 16), nullable=False, default=ScannerRunStatus.RUNNING
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    instruments_considered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    instruments_scored: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    instruments_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    screening_candidates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    watchlist_candidates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    #: How this slice was chosen (rotation, watchlist, explicit) — for the "why
    #: was this scanned" question the UI must answer (§6).
    selection_reason: Mapped[str | None] = mapped_column(String(200))

    configuration: Mapped[ScannerConfiguration | None] = relationship(back_populates="runs")
    results: Mapped[list[ScannerResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ScannerResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One instrument's score from one run, with full provenance."""

    __tablename__ = "scanner_results"
    __table_args__ = (
        Index("ix_scanner_results_run_score", "run_id", "core_score"),
        Index("ix_scanner_results_instrument", "instrument_id"),
        Index("ix_scanner_results_classification", "classification"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("scanner_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False
    )

    #: The score that drives this result's classification and default ranking —
    #: a weighted blend of the momentum core and the value score per the run's
    #: configuration. In a momentum-primary run this equals core_score; in a
    #: value-primary ("buy low") run it is value-led.
    primary_score: Mapped[Any] = mapped_column(Ratio, nullable=False, default=0)

    #: The 100-point momentum core score and its per-category breakdown.
    core_score: Mapped[Any] = mapped_column(Ratio, nullable=False)
    trend_score: Mapped[Any] = mapped_column(Ratio, nullable=False)
    momentum_score: Mapped[Any] = mapped_column(Ratio, nullable=False)
    risk_score: Mapped[Any] = mapped_column(Ratio, nullable=False)
    liquidity_score: Mapped[Any] = mapped_column(Ratio, nullable=False)
    positioning_score: Mapped[Any] = mapped_column(Ratio, nullable=False)

    #: Separate, optional. Never gates the core score (§6, acceptance 7).
    fundamental_score: Mapped[Any | None] = mapped_column(Ratio)

    #: The valuation lens (0-100): how *cheap* the instrument looks, computed
    #: alongside the momentum core and never folded into it. High = potentially
    #: undervalued (pulled back, low in range, below its average, oversold, and
    #: where fundamentals exist, cheap on earnings/book/yield).
    value_score: Mapped[Any | None] = mapped_column(Ratio)
    price_value_score: Mapped[Any | None] = mapped_column(Ratio)
    fundamental_value_score: Mapped[Any | None] = mapped_column(Ratio)
    value_signals: Mapped[dict[str, Any] | None] = mapped_column()

    # Indexed via the explicit Index in __table_args__; no column-level
    # index=True, which would define a second index of the same name.
    classification: Mapped[Classification] = mapped_column(
        StrEnumType(Classification, 24), nullable=False
    )

    # -- Provenance (§6) ----------------------------------------------------
    data_completeness: Mapped[Any] = mapped_column(Ratio, nullable=False)  # 0..1
    data_freshness_days: Mapped[Any | None] = mapped_column(Ratio)
    confidence: Mapped[Any] = mapped_column(Ratio, nullable=False)  # 0..1
    candles_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    provider: Mapped[str | None] = mapped_column(String(16))

    #: Human-readable explanation lists, so a score defends itself in the UI.
    positive_signals: Mapped[dict[str, Any] | None] = mapped_column()
    negative_signals: Mapped[dict[str, Any] | None] = mapped_column()
    missing_information: Mapped[dict[str, Any] | None] = mapped_column()
    #: Raw indicator values, retained for the detail page and reproducibility.
    metrics: Mapped[dict[str, Any] | None] = mapped_column()

    is_trading212_tradable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    run: Mapped[ScannerRun] = relationship(back_populates="results")


class TradeProposal(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A proposed trade generated from a scanner candidate (§6).

    A proposal is never executed on its own. It carries everything a human needs
    to decide — side, size, risk, stop, reason, expiry — and is submitted only
    after explicit authenticated approval AND a fresh pre-execution revalidation.
    """

    __tablename__ = "trade_proposals"
    __table_args__ = (
        Index("ix_trade_proposals_status", "status"),
        Index("ix_trade_proposals_expires", "expires_at"),
    )

    scanner_result_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("scanner_results.id", ondelete="SET NULL"), index=True
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status: Mapped[ProposalStatus] = mapped_column(
        StrEnumType(ProposalStatus, 24), nullable=False, default=ProposalStatus.PENDING_APPROVAL
    )
    side: Mapped[OrderSide] = mapped_column(StrEnumType(OrderSide, 8), nullable=False)

    proposed_quantity: Mapped[Any] = mapped_column(Quantity, nullable=False)
    max_position_value: Mapped[Any] = mapped_column(Money, nullable=False)
    risk_amount: Mapped[Any] = mapped_column(Money, nullable=False)
    risk_pct: Mapped[Any] = mapped_column(Ratio, nullable=False)
    indicative_entry_price: Mapped[Any] = mapped_column(Price, nullable=False)
    proposed_stop_price: Mapped[Any | None] = mapped_column(Price)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)

    reason: Mapped[str] = mapped_column(Text, nullable=False)
    #: Which broker the resulting order would target, resolved at proposal time.
    broker: Mapped[str | None] = mapped_column(String(24))

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    #: Set when a human decides. approved_by is null until then.
    approved_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[str | None] = mapped_column(Text)

    #: Set if a duplicate intent already exists, per §6's pre-exec checks.
    duplicate_of_proposal_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("trade_proposals.id", ondelete="SET NULL")
    )

    def is_expired_at(self, now: datetime) -> bool:
        return now >= self.expires_at

    @property
    def is_actionable(self) -> bool:
        """True only while a pending proposal can still be approved."""
        return self.status is ProposalStatus.PENDING_APPROVAL
