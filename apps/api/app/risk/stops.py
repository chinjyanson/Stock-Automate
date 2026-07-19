"""Stop management after entry (§9).

The initial protective stop is placed at fill time by the execution service. This
service owns everything that happens to a stop *after* that:

  * **Trigger** — fill resting stops a candle has breached (delegated to the
    venue) and close the local intent so its realised P/L is known.
  * **Trail** — ratchet the stop upward as price rises. Never downward: a
    trailing stop that could loosen is not protection, it is a slower loss.
  * **Time stop** — exit a position held past its configured horizon.
  * **Emergency exit** — flatten immediately, cancelling resting stops first.

Every adjustment and close is audited, because a stop that moved without a record
is indistinguishable from one that moved by mistake.
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
from app.broker.types import BrokerOrder, BrokerOrderRequest, BrokerPosition
from app.data.store import CandleStore
from app.indicators import functions as ind
from app.indicators.series import candles_to_series
from app.models.enums import (
    ActorKind,
    AuditEventKind,
    Interval,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeIntentStatus,
)
from app.models.risk import RiskConfiguration, TradeIntent

log = structlog.get_logger(__name__)


class StopService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._audit = AuditService(session)

    async def manage(
        self, broker: Broker, config: RiskConfiguration | None
    ) -> dict[str, int]:
        """One full pass: trigger, record closes, trail, time-stop.

        Safe to run repeatedly (it is scheduled): a closed intent is skipped, a
        stop only ever moves up, and a time exit fires once because it closes the
        intent it acted on.
        """
        triggered = await broker.process_stops()
        closed = await self._record_triggered_closes(broker)
        trailed = 0
        time_exits = 0
        if config is not None:
            if config.trailing_stop_enabled:
                trailed = await self._trail_stops(broker, config)
            if int(config.max_holding_days or 0) > 0:
                time_exits = await self._apply_time_stops(broker, config)
        await self._session.flush()
        return {
            "triggered": triggered,
            "closed": closed,
            "trailed": trailed,
            "time_exits": time_exits,
        }

    async def emergency_exit_all(
        self, broker: Broker, reason: str, *, actor_user_id: uuid.UUID | None = None
    ) -> int:
        """Flatten every open position at market, cancelling stops first."""
        positions = await self._positions_map(broker)
        count = 0
        for intent in await self._open_intents(broker):
            position = positions.get(str(intent.instrument_id))
            if position is None:
                continue
            await self._close_at_market(
                broker, intent, position, reason="emergency", actor_user_id=actor_user_id
            )
            count += 1
        await self._session.flush()
        if count:
            log.warning("stops.emergency_exit", reason=reason, positions=count)
        return count

    # -- Passes -------------------------------------------------------------

    async def _record_triggered_closes(self, broker: Broker) -> int:
        """Close intents whose resting stop the venue has now filled."""
        orders = {o.broker_order_id: o for o in await broker.get_order_history()}
        count = 0
        for intent in await self._open_intents(broker):
            if intent.stop_broker_order_id is None:
                continue
            order = orders.get(intent.stop_broker_order_id)
            if order is not None and order.status is OrderStatus.FILLED:
                self._mark_closed(intent, order.average_fill_price, "stop")
                await self._audit_close(intent, "stop", order.average_fill_price)
                count += 1
        return count

    async def _trail_stops(self, broker: Broker, config: RiskConfiguration) -> int:
        positions = await self._positions_map(broker)
        multiplier = Decimal(str(config.atr_stop_multiplier))
        count = 0
        for intent in await self._open_intents(broker):
            position = positions.get(str(intent.instrument_id))
            if position is None or intent.stop_broker_order_id is None:
                continue
            new_stop = await self._candidate_stop(intent.instrument_id, multiplier)
            if new_stop is None or intent.stop_price is None:
                continue
            # Only ever ratchet up.
            if new_stop <= Decimal(intent.stop_price):
                continue
            await self._replace_stop(broker, intent, position, new_stop)
            count += 1
        return count

    async def _apply_time_stops(self, broker: Broker, config: RiskConfiguration) -> int:
        positions = await self._positions_map(broker)
        max_days = int(config.max_holding_days)
        now = datetime.now(UTC)
        count = 0
        for intent in await self._open_intents(broker):
            position = positions.get(str(intent.instrument_id))
            if position is None:
                continue
            opened = intent.submitted_at or intent.created_at
            if (now - opened).days < max_days:
                continue
            await self._close_at_market(broker, intent, position, reason="time_stop")
            count += 1
        return count

    # -- Helpers ------------------------------------------------------------

    async def _open_intents(self, broker: Broker) -> list[TradeIntent]:
        rows = await self._session.execute(
            select(TradeIntent).where(
                TradeIntent.broker == broker.kind,
                TradeIntent.status == TradeIntentStatus.RECONCILED,
                TradeIntent.closed_at.is_(None),
            )
        )
        return list(rows.scalars().all())

    async def _positions_map(self, broker: Broker) -> dict[str, BrokerPosition]:
        return {p.broker_ticker: p for p in await broker.get_positions() if p.quantity > 0}

    async def _candidate_stop(
        self, instrument_id: uuid.UUID, multiplier: Decimal
    ) -> Decimal | None:
        candles = await self._store.get_candles(
            instrument_id, Interval.D1, limit=60, closed_only=True
        )
        if len(candles) < 20:
            return None
        series = candles_to_series(candles)
        atr = ind.average_true_range(series.high, series.low, series.close, period=14)
        if atr is None or atr <= 0:
            return None
        latest_close = Decimal(str(float(series.close[-1])))
        candidate = latest_close - Decimal(str(atr)) * multiplier
        return candidate if candidate > 0 else None

    async def _replace_stop(
        self,
        broker: Broker,
        intent: TradeIntent,
        position: BrokerPosition,
        new_stop: Decimal,
    ) -> None:
        old_stop = intent.stop_price
        if intent.stop_broker_order_id is not None:
            await broker.cancel_order(intent.stop_broker_order_id)
        order = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(intent.instrument_id),
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.STOP,
                stop_price=new_stop,
                client_reference=str(intent.client_reference),
            )
        )
        intent.stop_price = new_stop
        intent.stop_broker_order_id = order.broker_order_id
        await self._audit.record(
            kind=AuditEventKind.STOP_ADJUSTED,
            summary=(
                f"Trailed stop for {intent.instrument_id} "
                f"{old_stop} -> {new_stop.quantize(Decimal('0.00000001'))}"
            ),
            actor_kind=ActorKind.RISK_ENGINE,
            subject_type="trade_intent",
            subject_id=str(intent.id),
            trade_intent_id=str(intent.id),
            payload={"old_stop": str(old_stop), "new_stop": str(new_stop)},
        )

    async def _close_at_market(
        self,
        broker: Broker,
        intent: TradeIntent,
        position: BrokerPosition,
        *,
        reason: str,
        actor_user_id: uuid.UUID | None = None,
    ) -> BrokerOrder:
        if intent.stop_broker_order_id is not None:
            await broker.cancel_order(intent.stop_broker_order_id)
        order = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(intent.instrument_id),
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
                client_reference=str(intent.client_reference),
            )
        )
        self._mark_closed(intent, order.average_fill_price, reason)
        await self._audit_close(intent, reason, order.average_fill_price, actor_user_id)
        return order

    def _mark_closed(
        self, intent: TradeIntent, exit_price: Decimal | None, reason: str
    ) -> None:
        intent.closed_at = datetime.now(UTC)
        intent.exit_price = exit_price
        intent.exit_reason = reason

    async def _audit_close(
        self,
        intent: TradeIntent,
        reason: str,
        exit_price: Decimal | None,
        actor_user_id: uuid.UUID | None = None,
    ) -> None:
        await self._audit.record(
            kind=AuditEventKind.POSITION_CLOSED,
            summary=f"Closed position for {intent.instrument_id} ({reason}) @ {exit_price}",
            actor_kind=ActorKind.USER if actor_user_id else ActorKind.RISK_ENGINE,
            actor_user_id=actor_user_id,
            subject_type="trade_intent",
            subject_id=str(intent.id),
            trade_intent_id=str(intent.id),
            payload={"reason": reason, "exit_price": str(exit_price)},
        )
