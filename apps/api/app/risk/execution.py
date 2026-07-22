"""Executing an approved proposal (§9, §10, §14).

`ProposalService.approve` records the human decision, and this service turns an
APPROVED proposal into a sized, risk-checked, stopped position on the chosen
venue — the internal paper broker by default, or Trading 212 live when an arming
session authorises it.

The order of operations is the safety story:
  1. Re-validate the proposal (approval may have gone stale).
  2. For live, a preflight re-checks — in this transaction — the server flag, an
     active arming session, no blocking halt, and an available live instrument.
     Any doubt ⇒ REJECTED_BY_RISK, no order (§7 depth-in-defence).
  3. Size and gate it through the risk engine — bounded, for live, by the arming
     session's capital ceiling — which can still reject it outright.
  4. Record a `TradeIntent` *before* submitting, so a duplicate submission is
     impossible and an ambiguous outcome is reconcilable (§10).
  5. Submit against the venue's real ticker, fill, place the protective stop
     broker-side, mark EXECUTED.
An ambiguous broker response is never retried — it escalates to reconciliation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

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
from app.config import get_settings
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
from app.models.instrument import BrokerInstrument, Instrument
from app.models.market_data import Candle
from app.models.risk import TradeIntent
from app.models.scanner import ProposalStatus, TradeProposal
from app.risk.config import load_active_risk_config
from app.risk.engine import RiskEngine
from app.risk.halts import HaltService
from app.services.system_settings import active_broker_kind, live_mode_enabled

log = structlog.get_logger(__name__)


class ExecutionError(Exception):
    """An approved proposal could not be executed."""


class ExecutionService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        broker_kind: BrokerKind | None = None,
        broker: Broker | None = None,
    ) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._audit = AuditService(session)
        self._engine = RiskEngine(session)
        self._halts = HaltService(session)
        # An injected broker lets tests exercise the live path without ever
        # constructing the real live adapter (which needs the server flag +
        # credentials). Its kind wins, so the routing stays honest.
        self._injected_broker = broker
        #: An explicit venue, or None to resolve the active one (paper/live) at
        #: execution time — the product path.
        self._configured_kind: BrokerKind | None = (
            broker.kind if broker is not None else broker_kind
        )
        #: Resolved per call. Defaults to paper so nothing can read "live" by accident.
        self._broker_kind: BrokerKind = self._configured_kind or BrokerKind.TRADING212_DEMO

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

        # The product path follows the paper/live toggle; an explicit venue (tests,
        # or a caller that already knows) wins.
        if self._configured_kind is None:
            self._broker_kind = await active_broker_kind(self._session)

        instrument = await self._session.get(Instrument, proposal.instrument_id)
        if instrument is None:
            raise ExecutionError("Instrument no longer exists")
        if instrument.is_suspended:
            proposal.status = ProposalStatus.REJECTED_BY_RISK
            await self._session.flush()
            raise ExecutionError(f"{instrument.name} is suspended; cannot execute")

        # Live depth-in-defence: re-check the world here, in this transaction,
        # before a real order can be built. Rejecting is recorded, not silent.
        if self._broker_kind.is_live:
            blocked = await self._live_preflight(instrument)
            if blocked is not None:
                proposal.status = ProposalStatus.REJECTED_BY_RISK
                await self._session.flush()
                await self._audit.record(
                    kind=AuditEventKind.ORDER_REJECTED,
                    summary=f"Live preflight refused {instrument.name}: {blocked}",
                    actor_kind=ActorKind.RISK_ENGINE,
                    subject_type="trade_proposal",
                    subject_id=str(proposal.id),
                    payload={"reason": blocked},
                )
                raise ExecutionError(f"Live preflight refused the order: {blocked}")

        broker = self._injected_broker or resolve_broker(
            self._broker_kind, session=self._session
        )
        try:
            return await self._execute(proposal, instrument, broker, actor_user_id)
        finally:
            if self._injected_broker is None:
                await broker.close()

    async def _execute(
        self,
        proposal: TradeProposal,
        instrument: Instrument,
        broker: Broker,
        actor_user_id: uuid.UUID | None,
    ) -> TradeProposal:
        # The venue's real ticker: the instrument id for paper, the broker's own
        # spelling (via BrokerInstrument) for Trading 212. A real broker will not
        # accept our UUID.
        ticker = await self._broker_ticker(instrument)

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

        # For live, the affirmed capital ceiling bounds sizing for this session.
        equity_ceiling = await self._live_equity_ceiling()

        decision = await self._engine.evaluate(
            instrument=instrument,
            config=config,
            account=account,
            positions=positions,
            candles=candles,
            benchmark_candles=benchmark,
            broker=self._broker_kind,
            equity_ceiling=equity_ceiling,
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
            broker_ticker=ticker,
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
                    broker_ticker=ticker,
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

    async def _broker_instrument(self, instrument_id: uuid.UUID) -> BrokerInstrument | None:
        return (
            await self._session.execute(
                select(BrokerInstrument).where(
                    BrokerInstrument.instrument_id == instrument_id,
                    BrokerInstrument.broker == self._broker_kind,
                )
            )
        ).scalars().first()

    async def _broker_ticker(self, instrument: Instrument) -> str:
        """The venue's own ticker for the instrument, or raise if unavailable."""
        if self._broker_kind is BrokerKind.INTERNAL_PAPER:
            return str(instrument.id)
        bi = await self._broker_instrument(instrument.id)
        if bi is None or not bi.is_currently_available:
            raise ExecutionError(
                f"No available {self._broker_kind.value} instrument for {instrument.name}"
            )
        return bi.broker_ticker

    async def _live_preflight(self, instrument: Instrument) -> str | None:
        """Re-check every live precondition here. Returns a reason, or None if clear."""
        settings = get_settings()
        if not settings.live_trading_enabled:
            return "live trading is disabled on the server"
        if not await live_mode_enabled(self._session):
            return "the trading venue is set to paper"
        halt = await self._halts.blocking_halt(instrument.id)
        if halt is not None:
            return f"risk halt active ({halt.kind.value})"
        bi = await self._broker_instrument(instrument.id)
        if bi is None or not bi.is_currently_available:
            return "instrument is not available on the live broker"
        return None

    async def _live_equity_ceiling(self) -> Decimal | None:
        """The configured capital ceiling for live, from the active risk config."""
        if not self._broker_kind.is_live:
            return None
        config = await load_active_risk_config(self._session)
        if config is None or config.max_live_capital is None:
            return None
        return Decimal(config.max_live_capital)

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
