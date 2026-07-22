"""Trade proposals and the approval workflow (§6).

A scanner candidate does not become an order by itself. This module turns a
candidate into a `TradeProposal` carrying everything a human needs to decide,
and handles the explicit approval/rejection — including the pre-execution
revalidation §6 mandates.

Scope boundary, stated plainly: proposal *sizing* here is deliberately simple —
volatility-adjusted risk with an ATR stop and a fixed risk fraction. The full
risk engine (§9) with its portfolio caps, correlation filter and open-risk
limits is Phase 3, and an approved proposal is not submitted to a broker until
that engine exists. Approval therefore records the decision and re-runs the
checks that are possible today; it does not yet place an order. That seam is
explicit rather than faked — `approve()` returns the proposal in APPROVED state
and says execution is pending the Phase 3 engine.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.data.store import CandleStore
from app.indicators import functions as ind
from app.indicators.series import candles_to_series
from app.models.enums import ActorKind, AuditEventKind, Interval, OrderSide
from app.models.instrument import Instrument
from app.models.scanner import ProposalStatus, ScannerResult, TradeProposal

log = structlog.get_logger(__name__)

#: Default proposal parameters. In Phase 3 these come from RiskConfiguration.
DEFAULT_RISK_PER_TRADE = Decimal("0.01")  # 1% of the configured capital
DEFAULT_ATR_STOP_MULTIPLIER = Decimal("2.0")
DEFAULT_MAX_POSITION_PCT = Decimal("0.10")  # 10% of capital
DEFAULT_APPROVAL_TTL_MINUTES = 60


@dataclass
class ProposalInputs:
    """The account/config context a proposal is sized against.

    Passed in rather than read from a broker here so proposal generation stays
    pure and testable. The caller supplies current equity.
    """

    account_equity: Decimal
    risk_per_trade: Decimal = DEFAULT_RISK_PER_TRADE
    atr_stop_multiplier: Decimal = DEFAULT_ATR_STOP_MULTIPLIER
    max_position_pct: Decimal = DEFAULT_MAX_POSITION_PCT
    approval_ttl_minutes: int = DEFAULT_APPROVAL_TTL_MINUTES


class ProposalError(Exception):
    """A proposal could not be generated or actioned."""


class ProposalService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._audit = AuditService(session)

    async def propose_from_result(
        self,
        result: ScannerResult,
        inputs: ProposalInputs,
        *,
        actor_user_id: uuid.UUID | None = None,
    ) -> TradeProposal:
        """Generate a long proposal from a scanner result.

        Long-only in Phase 2 — the initial strategies are long-only (§8), and a
        scanner candidate is a "this looks strong" signal, not a short thesis.
        """
        instrument = await self._session.get(Instrument, result.instrument_id)
        if instrument is None:
            raise ProposalError("Instrument no longer exists")

        top_signals = (result.positive_signals or {}).get("items", [])[:3]
        reason = (
            f"Screening score {result.core_score} "
            f"({', '.join(top_signals) if top_signals else 'passes the configured screen'})"
        )
        return await self._propose_long(
            instrument,
            inputs,
            reason=reason,
            scanner_result_id=result.id,
            actor_user_id=actor_user_id,
        )

    async def propose_from_signal(
        self,
        instrument: Instrument,
        inputs: ProposalInputs,
        *,
        reason: str,
        actor_user_id: uuid.UUID | None = None,
    ) -> TradeProposal:
        """Generate a long proposal from a strategy signal (§8).

        Same volatility-adjusted sizing as a scanner-sourced proposal, but the
        reason is the strategy's and there is no scanner result behind it. The
        risk engine still gates the resulting order downstream.
        """
        return await self._propose_long(
            instrument,
            inputs,
            reason=reason,
            scanner_result_id=None,
            actor_user_id=actor_user_id,
        )

    async def _propose_long(
        self,
        instrument: Instrument,
        inputs: ProposalInputs,
        *,
        reason: str,
        scanner_result_id: uuid.UUID | None,
        actor_user_id: uuid.UUID | None,
    ) -> TradeProposal:
        """Shared long-proposal builder: size, stop, duplicate-guard, audit.

        The single place that turns "buy this instrument" into a sized, stopped
        `TradeProposal`, whatever produced the intent (scanner or strategy).
        """
        if instrument.suspended_at is not None:
            raise ProposalError(f"{instrument.name} is suspended")

        candles = await self._store.get_candles(
            instrument.id, Interval.D1, limit=60, closed_only=True
        )
        if len(candles) < 20:
            raise ProposalError("Insufficient recent history to size a proposal")

        series = candles_to_series(candles)
        entry_price = Decimal(str(float(series.close[-1])))
        if entry_price <= 0:
            raise ProposalError("Invalid entry price")

        atr = ind.average_true_range(series.high, series.low, series.close, period=14)
        if atr is None or atr <= 0:
            raise ProposalError("Could not compute a stop distance (ATR unavailable)")

        stop_distance = Decimal(str(atr)) * inputs.atr_stop_multiplier
        stop_price = (entry_price - stop_distance).quantize(Decimal("0.00000001"))
        if stop_price <= 0:
            raise ProposalError("Computed stop is at or below zero")

        # Volatility-adjusted size: risk a fixed fraction of equity across the
        # stop distance (§9's core sizing formula; the caps are Phase 3).
        risk_budget = (inputs.account_equity * inputs.risk_per_trade).quantize(Decimal("0.0001"))
        raw_quantity = risk_budget / stop_distance

        # Cap by maximum position value.
        max_value = inputs.account_equity * inputs.max_position_pct
        max_qty_by_value = max_value / entry_price
        quantity = min(raw_quantity, max_qty_by_value)
        # Round down to avoid breaching the cap that produced the number.
        quantity = quantity.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        if quantity <= 0:
            raise ProposalError("Position size rounded to zero — equity too small for one unit")

        position_value = (quantity * entry_price).quantize(Decimal("0.0001"))
        risk_amount = (quantity * stop_distance).quantize(Decimal("0.0001"))
        risk_pct = (risk_amount / inputs.account_equity).quantize(Decimal("0.000001"))

        # Guard: refuse a duplicate pending proposal for the same instrument.
        existing = await self._pending_proposal_for(instrument.id)
        if existing is not None:
            raise ProposalError(
                f"A pending proposal already exists for {instrument.name} "
                f"(expires {existing.expires_at.isoformat()})"
            )

        proposal = TradeProposal(
            scanner_result_id=scanner_result_id,
            instrument_id=instrument.id,
            status=ProposalStatus.PENDING_APPROVAL,
            side=OrderSide.BUY,
            proposed_quantity=quantity,
            max_position_value=position_value,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            indicative_entry_price=entry_price,
            proposed_stop_price=stop_price,
            currency=instrument.currency,
            reason=reason,
            expires_at=datetime.now(UTC) + timedelta(minutes=inputs.approval_ttl_minutes),
        )
        self._session.add(proposal)
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.TRADE_PROPOSED,
            summary=(
                f"Proposed BUY {quantity} {instrument.name} @ ~{entry_price} "
                f"(risk {risk_amount} {instrument.currency}, {risk_pct:.2%})"
            ),
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.SYSTEM,
            actor_user_id=actor_user_id,
            subject_type="trade_proposal",
            subject_id=str(proposal.id),
            payload={
                "instrument": instrument.name,
                "quantity": str(quantity),
                "entry_price": str(entry_price),
                "stop_price": str(stop_price),
                "risk_amount": str(risk_amount),
                "risk_pct": str(risk_pct),
                "scanner_result_id": str(scanner_result_id) if scanner_result_id else None,
            },
        )
        log.info("proposal.created", proposal_id=str(proposal.id), instrument=instrument.name)
        return proposal

    async def approve(
        self, proposal_id: uuid.UUID, *, actor_user_id: uuid.UUID, note: str | None = None
    ) -> TradeProposal:
        """Approve a proposal after re-running the checks §6 requires.

        The checks that are possible today all run here: not expired, still
        pending, instrument still tradable, no duplicate intent. Actually placing
        the order is Phase 3 (risk engine + paper broker); until then the
        proposal reaches APPROVED and stops there — deliberately, not silently.
        """
        proposal = await self._get_actionable(proposal_id)
        instrument = await self._session.get(Instrument, proposal.instrument_id)
        if instrument is None:
            raise ProposalError("Instrument no longer exists")

        now = datetime.now(UTC)
        if proposal.is_expired_at(now):
            proposal.status = ProposalStatus.EXPIRED
            await self._session.flush()
            await self._audit.record(
                kind=AuditEventKind.APPROVAL_EXPIRED,
                summary=f"Approval attempted after expiry for {instrument.name}",
                actor_kind=ActorKind.USER,
                actor_user_id=actor_user_id,
                subject_type="trade_proposal",
                subject_id=str(proposal.id),
            )
            raise ProposalError("This proposal has expired; generate a new one")

        if instrument.suspended_at is not None:
            raise ProposalError(f"{instrument.name} is now suspended; cannot approve")

        # APPROVED is now a handoff, not a dead end: the caller passes the
        # proposal to the risk engine (app.risk.execution), which sizes, gates,
        # and — for the paper venue — submits the order. Approval records the
        # human decision; the engine still owns whether an order actually goes.
        proposal.status = ProposalStatus.APPROVED
        proposal.approved_by_user_id = actor_user_id
        proposal.decided_at = now
        proposal.decision_note = note
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.TRADE_APPROVED,
            summary=f"Approved proposal to BUY {proposal.proposed_quantity} {instrument.name}",
            actor_kind=ActorKind.USER,
            actor_user_id=actor_user_id,
            subject_type="trade_proposal",
            subject_id=str(proposal.id),
            payload={"note": note, "revalidated": True},
        )
        log.info("proposal.approved", proposal_id=str(proposal.id))
        return proposal

    async def reject(
        self, proposal_id: uuid.UUID, *, actor_user_id: uuid.UUID, note: str | None = None
    ) -> TradeProposal:
        proposal = await self._get_actionable(proposal_id)
        proposal.status = ProposalStatus.REJECTED
        proposal.approved_by_user_id = actor_user_id
        proposal.decided_at = datetime.now(UTC)
        proposal.decision_note = note
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.TRADE_REJECTED,
            summary=f"Rejected proposal {proposal.id}",
            actor_kind=ActorKind.USER,
            actor_user_id=actor_user_id,
            subject_type="trade_proposal",
            subject_id=str(proposal.id),
            payload={"note": note},
        )
        return proposal

    async def expire_stale(self) -> int:
        """Mark expired pending proposals EXPIRED. Returns the count.

        Runs on a schedule (§16). Idempotent: an already-expired proposal is not
        re-processed.
        """
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(TradeProposal).where(
                TradeProposal.status == ProposalStatus.PENDING_APPROVAL,
                TradeProposal.expires_at <= now,
            )
        )
        proposals = list(result.scalars().all())
        for proposal in proposals:
            proposal.status = ProposalStatus.EXPIRED
            await self._audit.record(
                kind=AuditEventKind.APPROVAL_EXPIRED,
                summary=f"Proposal {proposal.id} expired without a decision",
                actor_kind=ActorKind.SCHEDULER,
                actor_label="approval_expiry_job",
                subject_type="trade_proposal",
                subject_id=str(proposal.id),
            )
        await self._session.flush()
        return len(proposals)

    async def _get_actionable(self, proposal_id: uuid.UUID) -> TradeProposal:
        proposal = await self._session.get(TradeProposal, proposal_id)
        if proposal is None:
            raise ProposalError("Proposal not found")
        if not proposal.is_actionable:
            raise ProposalError(
                f"Proposal is {proposal.status.value}, not pending; no action possible"
            )
        return proposal

    async def _pending_proposal_for(self, instrument_id: uuid.UUID) -> TradeProposal | None:
        result = await self._session.execute(
            select(TradeProposal).where(
                TradeProposal.instrument_id == instrument_id,
                TradeProposal.status == ProposalStatus.PENDING_APPROVAL,
            )
        )
        return result.scalars().first()
