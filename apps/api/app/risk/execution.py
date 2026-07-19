"""Executing an approved proposal against the paper venue (§9, §10).

This is the seam Phase 2 left explicit: `ProposalService.approve` records the
human decision, and this service turns an APPROVED proposal into a sized,
risk-checked, stopped position. It is deliberately paper-only in this pass — the
internal paper broker and Trading 212 demo are the only venues reached; live
execution is gated elsewhere and out of scope here.

The order of operations is the safety story:
  1. Re-validate the proposal (approval may have gone stale).
  2. Size and gate it through the risk engine — which can reject it outright.
  3. Record a `TradeIntent` *before* submitting, so a duplicate submission is
     impossible and an ambiguous outcome is reconcilable (§10).
  4. Submit, fill, place the protective stop broker-side, mark EXECUTED.
An ambiguous broker response is never retried — it escalates to reconciliation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.broker.base import Broker
from app.broker.factory import resolve_broker
from app.broker.types import (
    BrokerAmbiguousResponseError,
    BrokerOrderRejectedError,
    BrokerOrderRequest,
)
from app.data.store import CandleStore
from app.models.enums import (
    ActorKind,
    AuditEventKind,
    BrokerKind,
    Interval,
    OrderSide,
    OrderType,
    TradeIntentStatus,
)
from app.models.instrument import Instrument
from app.models.market_data import Candle
from app.models.risk import TradeIntent
from app.models.scanner import ProposalStatus, TradeProposal
from app.risk.config import load_active_risk_config
from app.risk.engine import RiskEngine

log = structlog.get_logger(__name__)


class ExecutionError(Exception):
    """An approved proposal could not be executed."""


class ExecutionService:
    def __init__(
        self, session: AsyncSession, *, broker_kind: BrokerKind = BrokerKind.INTERNAL_PAPER
    ) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._audit = AuditService(session)
        self._engine = RiskEngine(session)
        if broker_kind.is_live:
            # This pass never executes live. The guard is here, not just at the
            # call site, so the invariant holds however the service is reached.
            raise ExecutionError("Live execution is out of scope for this service.")
        self._broker_kind = broker_kind

    async def execute_approved(
        self, proposal: TradeProposal, *, actor_user_id: uuid.UUID | None = None
    ) -> TradeProposal:
        # Idempotency comes first: a proposal that already has a live intent has
        # been acted on (possibly reaching EXECUTED), so a retry must return what
        # exists rather than tripping the status check or submitting again.
        existing = await self._intent_for_proposal(proposal.id)
        if existing is not None and existing.status not in {
            TradeIntentStatus.FAILED,
            TradeIntentStatus.ABANDONED,
        }:
            return proposal

        if proposal.status is not ProposalStatus.APPROVED:
            raise ExecutionError(
                f"Only an APPROVED proposal can be executed; this is {proposal.status.value}."
            )

        instrument = await self._session.get(Instrument, proposal.instrument_id)
        if instrument is None:
            raise ExecutionError("Instrument no longer exists")
        if instrument.is_suspended:
            proposal.status = ProposalStatus.REJECTED_BY_RISK
            await self._session.flush()
            raise ExecutionError(f"{instrument.name} is suspended; cannot execute")

        broker = resolve_broker(self._broker_kind, session=self._session)
        try:
            return await self._execute(proposal, instrument, broker, actor_user_id)
        finally:
            await broker.close()

    async def _execute(
        self,
        proposal: TradeProposal,
        instrument: Instrument,
        broker: Broker,
        actor_user_id: uuid.UUID | None,
    ) -> TradeProposal:
        config = await load_active_risk_config(self._session)
        account = await broker.get_account()
        positions = await broker.get_positions()
        candles = await self._store.get_candles(
            instrument.id, Interval.D1, limit=250, closed_only=True
        )
        benchmark = (
            await self._benchmark_candles(config.correlation_benchmark_symbol)
            if config
            else None
        )

        decision = await self._engine.evaluate(
            instrument=instrument,
            config=config,
            account=account,
            positions=positions,
            candles=candles,
            benchmark_candles=benchmark,
            broker=self._broker_kind,
        )
        if decision.rejected:
            proposal.status = ProposalStatus.REJECTED_BY_RISK
            await self._session.flush()
            await self._audit.record(
                kind=AuditEventKind.ORDER_REJECTED,
                summary=f"Risk engine rejected {instrument.name}: {decision.reason}",
                actor_kind=ActorKind.RISK_ENGINE,
                subject_type="trade_proposal",
                subject_id=str(proposal.id),
                payload={"reason": decision.reason},
            )
            log.info("execution.rejected_by_risk", proposal_id=str(proposal.id))
            return proposal

        intent = TradeIntent(
            proposal_id=proposal.id,
            instrument_id=instrument.id,
            broker=self._broker_kind,
            client_reference=uuid.uuid4(),
            status=TradeIntentStatus.SUBMITTING,
            side=OrderSide.BUY,
            quantity=decision.approved_quantity,
            stop_price=decision.stop_price,
        )
        self._session.add(intent)
        await self._session.flush()

        request = BrokerOrderRequest(
            broker_ticker=str(instrument.id),
            side=OrderSide.BUY,
            quantity=decision.approved_quantity,
            order_type=OrderType.MARKET,
            client_reference=str(intent.client_reference),
        )
        try:
            order = await broker.place_order(request)
        except BrokerAmbiguousResponseError as exc:
            # The order may or may not exist. Never retry — reconcile.
            intent.status = TradeIntentStatus.RECONCILIATION_REQUIRED
            await self._session.flush()
            await self._audit.record(
                kind=AuditEventKind.RECONCILIATION_REQUIRED,
                summary=f"Ambiguous submission for {instrument.name}; reconciliation required",
                actor_kind=ActorKind.RISK_ENGINE,
                subject_type="trade_intent",
                subject_id=str(intent.id),
                trade_intent_id=str(intent.id),
            )
            raise ExecutionError(
                "Order outcome is ambiguous; reconciliation required before any retry."
            ) from exc
        except BrokerOrderRejectedError as exc:
            intent.status = TradeIntentStatus.FAILED
            proposal.status = ProposalStatus.REJECTED_BY_RISK
            await self._session.flush()
            await self._audit.record(
                kind=AuditEventKind.ORDER_REJECTED,
                summary=f"Broker rejected {instrument.name}: {exc}",
                actor_kind=ActorKind.RISK_ENGINE,
                subject_type="trade_proposal",
                subject_id=str(proposal.id),
                payload={"reason": str(exc)},
            )
            raise ExecutionError(f"Broker rejected the order: {exc}") from exc

        now = datetime.now(UTC)
        intent.status = TradeIntentStatus.RECONCILED  # paper fills synchronously
        intent.broker_order_id = order.broker_order_id
        intent.submitted_at = now
        intent.reconciled_at = now
        intent.filled_quantity = order.filled_quantity
        intent.filled_price = order.average_fill_price

        await self._audit.record(
            kind=AuditEventKind.ORDER_SUBMITTED,
            summary=f"Submitted BUY {order.filled_quantity} {instrument.name}",
            actor_kind=ActorKind.RISK_ENGINE,
            subject_type="trade_intent",
            subject_id=str(intent.id),
            trade_intent_id=str(intent.id),
            payload={"broker_order_id": order.broker_order_id},
        )
        await self._audit.record(
            kind=AuditEventKind.ORDER_FILLED,
            summary=(
                f"Filled BUY {order.filled_quantity} {instrument.name} "
                f"@ {order.average_fill_price}"
            ),
            actor_kind=ActorKind.RISK_ENGINE,
            subject_type="trade_intent",
            subject_id=str(intent.id),
            trade_intent_id=str(intent.id),
            payload={
                "filled_quantity": str(order.filled_quantity),
                "fill_price": str(order.average_fill_price),
            },
        )

        # Protective stop, placed broker-side so it survives our process dying.
        if decision.stop_price is not None:
            stop_order = await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker=str(instrument.id),
                    side=OrderSide.SELL,
                    quantity=order.filled_quantity,
                    order_type=OrderType.STOP,
                    stop_price=decision.stop_price,
                    client_reference=str(intent.client_reference),
                )
            )
            intent.stop_broker_order_id = stop_order.broker_order_id

        proposal.status = ProposalStatus.EXECUTED
        await self._session.flush()
        log.info(
            "execution.filled",
            proposal_id=str(proposal.id),
            quantity=str(order.filled_quantity),
        )
        return proposal

    async def _intent_for_proposal(self, proposal_id: uuid.UUID) -> TradeIntent | None:
        return (
            await self._session.execute(
                select(TradeIntent).where(TradeIntent.proposal_id == proposal_id)
            )
        ).scalars().first()

    async def _benchmark_candles(self, symbol: str) -> list[Candle] | None:
        """Best-effort benchmark candles for the correlation filter.

        Resolved by exchange ticker. Absent benchmark data disables the
        correlation reduction (it cannot fabricate a correlation) but never
        blocks the trade — the reduction only ever *tightens* sizing.
        """
        instrument = (
            await self._session.execute(
                select(Instrument).where(Instrument.exchange_ticker == symbol).limit(1)
            )
        ).scalar_one_or_none()
        if instrument is None:
            return None
        candles = await self._store.get_candles(
            instrument.id, Interval.D1, limit=250, closed_only=True
        )
        return candles or None
