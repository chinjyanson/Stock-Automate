"""Internal paper broker — a DB-backed simulated venue (§3, §7).

A first-class execution venue, not a test double. Unlike `MockBroker` (in-memory
fixtures), this holds its cash, positions and orders in Postgres, so state
survives a restart and reconciliation has a real second side to compare against.

Fills are deterministic and offline: a market order fills at the latest *closed*
daily close from the local candle store — the same store strategies read, never a
live provider (`docs/architecture.md`). Protective stops rest as working orders
and are triggered by `process_stops()` against subsequent candles, so a stop is
broker-side and survives our process dying, exactly as `docs/risk-model.md`
requires.

Ticker convention: `broker_ticker` carries the canonical `Instrument.id` as a
string. The internal venue owns no tickers, so the instrument id *is* its symbol.

Scope: long-only entries with protective sell stops — the Phase 2/3 strategies
are long-only. FX is out of scope; cash is accounted in the candle's stored unit,
which is already provider-unit-normalised (§4).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.broker.base import Broker
from app.broker.types import (
    BrokerAccount,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRejectedError,
    BrokerOrderRequest,
    BrokerPosition,
    BrokerUnavailableError,
    ReconciliationDiscrepancy,
    ReconciliationResult,
)
from app.data.store import CandleStore
from app.models.enums import (
    BrokerKind,
    Interval,
    OrderSide,
    OrderStatus,
    OrderType,
    TradeIntentStatus,
)
from app.models.instrument import Instrument
from app.models.paper import PaperBrokerAccount, PaperBrokerOrder, PaperBrokerPosition
from app.models.risk import TradeIntent

log = structlog.get_logger(__name__)

#: Paper capital granted on first use. Large enough that percentage caps, not
#: "can I afford one share", are what bind position size during paper trading.
DEFAULT_PAPER_CASH = Decimal("100000.00")
DEFAULT_PAPER_CURRENCY = "GBP"


class InternalPaperBroker(Broker):
    """A simulated venue backed by the `paper_broker_*` tables."""

    kind = BrokerKind.INTERNAL_PAPER

    def __init__(
        self,
        session: AsyncSession,
        *,
        starting_cash: Decimal = DEFAULT_PAPER_CASH,
        currency: str = DEFAULT_PAPER_CURRENCY,
    ) -> None:
        self._session = session
        self._store = CandleStore(session)
        self._starting_cash = starting_cash
        self._currency = currency

    # -- Account bootstrap --------------------------------------------------

    async def _ensure_account(self) -> PaperBrokerAccount:
        """Return the single paper account, creating it on first use."""
        account = (
            await self._session.execute(
                select(PaperBrokerAccount).where(
                    PaperBrokerAccount.currency == self._currency
                )
            )
        ).scalar_one_or_none()
        if account is None:
            account = PaperBrokerAccount(
                currency=self._currency,
                cash=self._starting_cash,
                starting_cash=self._starting_cash,
            )
            self._session.add(account)
            await self._session.flush()
        return account

    async def _fill_price(self, broker_ticker: str) -> Decimal:
        """Latest closed daily close for the venue ticker, or fail closed."""
        try:
            instrument_id = uuid.UUID(broker_ticker)
        except ValueError as exc:
            raise BrokerOrderRejectedError(
                f"Not a valid internal-paper ticker: {broker_ticker!r}"
            ) from exc
        candle = await self._store.latest_candle(instrument_id, Interval.D1, closed_only=True)
        if candle is None or candle.close is None or candle.close <= 0:
            # No priced, closed bar means we cannot value a fill — do not trade.
            raise BrokerUnavailableError(
                f"No closed daily candle to price {broker_ticker}; refusing to fill (fail closed)."
            )
        return Decimal(candle.close)

    async def _position(
        self, account_id: uuid.UUID, broker_ticker: str
    ) -> PaperBrokerPosition | None:
        return (
            await self._session.execute(
                select(PaperBrokerPosition).where(
                    PaperBrokerPosition.account_id == account_id,
                    PaperBrokerPosition.broker_ticker == broker_ticker,
                )
            )
        ).scalar_one_or_none()

    # -- Broker interface ---------------------------------------------------

    async def sync_instruments(self) -> list[BrokerInstrument]:
        """The venue trades any instrument with stored candles; id is its ticker."""
        instruments = (
            (await self._session.execute(select(Instrument))).scalars().all()
        )
        return [
            BrokerInstrument(
                broker_ticker=str(i.id),
                name=i.name,
                isin=i.isin,
                currency=i.currency,
                exchange_mic=None,
                kind=i.kind.value,
                is_currently_available=not i.is_suspended,
                min_quantity=Decimal("0.00000001"),
                quantity_step=Decimal("0.00000001"),
                supports_fractional=True,
            )
            for i in instruments
        ]

    async def get_account(self) -> BrokerAccount:
        account = await self._ensure_account()
        positions = await self.get_positions()
        invested = sum(
            (p.quantity * (p.current_price or p.average_price) for p in positions),
            start=Decimal(0),
        )
        return BrokerAccount(
            account_id=str(account.id),
            currency=account.currency,
            cash=Decimal(account.cash),
            total=Decimal(account.cash) + invested,
            free_for_trading=Decimal(account.cash),
            invested=invested,
            result=Decimal(account.cash) + invested - Decimal(account.starting_cash),
            blocked=Decimal(0),
            retrieved_at=datetime.now(UTC),
        )

    async def get_positions(self) -> list[BrokerPosition]:
        account = await self._ensure_account()
        rows = (
            (
                await self._session.execute(
                    select(PaperBrokerPosition).where(
                        PaperBrokerPosition.account_id == account.id
                    )
                )
            )
            .scalars()
            .all()
        )
        positions: list[BrokerPosition] = []
        for row in rows:
            if row.quantity == 0:
                continue
            candle = await self._store.latest_candle(
                uuid.UUID(row.broker_ticker), Interval.D1, closed_only=True
            )
            current = Decimal(candle.close) if candle and candle.close else None
            avg = Decimal(row.average_price)
            positions.append(
                BrokerPosition(
                    broker_ticker=row.broker_ticker,
                    quantity=Decimal(row.quantity),
                    average_price=avg,
                    current_price=current,
                    unrealised_pnl=((current - avg) * Decimal(row.quantity))
                    if current is not None
                    else None,
                )
            )
        return positions

    async def get_pending_orders(self) -> list[BrokerOrder]:
        account = await self._ensure_account()
        rows = (
            (
                await self._session.execute(
                    select(PaperBrokerOrder).where(
                        PaperBrokerOrder.account_id == account.id
                    )
                )
            )
            .scalars()
            .all()
        )
        return [self._to_dto(r) for r in rows if not r.status.is_terminal]

    async def get_order_history(self) -> list[BrokerOrder]:
        account = await self._ensure_account()
        rows = (
            (
                await self._session.execute(
                    select(PaperBrokerOrder)
                    .where(PaperBrokerOrder.account_id == account.id)
                    .order_by(PaperBrokerOrder.placed_at.asc())
                )
            )
            .scalars()
            .all()
        )
        return [self._to_dto(r) for r in rows]

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        request.validate()
        account = await self._ensure_account()
        now = datetime.now(UTC)

        if request.order_type is OrderType.STOP:
            # A protective stop rests until a candle breaches it (process_stops).
            return await self._place_resting_stop(account, request, now)
        if request.order_type is OrderType.MARKET:
            return await self._fill_market(account, request, now)

        raise BrokerOrderRejectedError(
            "The internal paper venue supports MARKET and STOP orders only, "
            f"not {request.order_type}."
        )

    async def _fill_market(
        self, account: PaperBrokerAccount, request: BrokerOrderRequest, now: datetime
    ) -> BrokerOrder:
        fill_price = await self._fill_price(request.broker_ticker)
        cost = fill_price * request.quantity
        position = await self._position(account.id, request.broker_ticker)

        if request.side is OrderSide.BUY:
            if cost > Decimal(account.cash):
                raise BrokerOrderRejectedError(
                    f"Insufficient cash: need {cost}, have {account.cash}"
                )
            account.cash = Decimal(account.cash) - cost
            if position is None:
                self._session.add(
                    PaperBrokerPosition(
                        account_id=account.id,
                        broker_ticker=request.broker_ticker,
                        quantity=request.quantity,
                        average_price=fill_price,
                        opened_at=now,
                    )
                )
            else:
                held_qty = Decimal(position.quantity)
                new_qty = held_qty + request.quantity
                # Weighted average cost; guards the divide when re-opening flat.
                position.average_price = (
                    (held_qty * Decimal(position.average_price)) + cost
                ) / new_qty
                position.quantity = new_qty
        else:  # SELL
            held_qty = Decimal(position.quantity) if position else Decimal(0)
            if request.quantity > held_qty:
                raise BrokerOrderRejectedError(
                    f"Cannot sell {request.quantity}; position is {held_qty}"
                )
            account.cash = Decimal(account.cash) + cost
            assert position is not None
            position.quantity = held_qty - request.quantity

        order = PaperBrokerOrder(
            account_id=account.id,
            broker_order_id=str(uuid.uuid4()),
            broker_ticker=request.broker_ticker,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.FILLED,
            filled_quantity=request.quantity,
            average_fill_price=fill_price,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            client_reference=request.client_reference,
            placed_at=now,
            terminal_at=now,
        )
        self._session.add(order)
        await self._session.flush()
        return self._to_dto(order)

    async def _place_resting_stop(
        self, account: PaperBrokerAccount, request: BrokerOrderRequest, now: datetime
    ) -> BrokerOrder:
        order = PaperBrokerOrder(
            account_id=account.id,
            broker_order_id=str(uuid.uuid4()),
            broker_ticker=request.broker_ticker,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal(0),
            stop_price=request.stop_price,
            client_reference=request.client_reference,
            placed_at=now,
        )
        self._session.add(order)
        await self._session.flush()
        return self._to_dto(order)

    async def cancel_order(self, broker_order_id: str) -> None:
        order = (
            await self._session.execute(
                select(PaperBrokerOrder).where(
                    PaperBrokerOrder.broker_order_id == broker_order_id
                )
            )
        ).scalar_one_or_none()
        if order is not None and not order.status.is_terminal:
            order.status = OrderStatus.CANCELLED
            order.terminal_at = datetime.now(UTC)
            await self._session.flush()

    async def process_stops(self) -> int:
        """Trigger resting stops breached by the latest candle. Returns fills.

        A protective sell stop fills when the day's low reaches the stop price;
        the fill is booked at the stop price (a conservative, deterministic
        approximation of the stop-out). Idempotent: an already-filled stop is
        terminal and is not reconsidered.
        """
        account = await self._ensure_account()
        working = (
            (
                await self._session.execute(
                    select(PaperBrokerOrder).where(
                        PaperBrokerOrder.account_id == account.id,
                        PaperBrokerOrder.order_type == OrderType.STOP,
                        PaperBrokerOrder.status == OrderStatus.SUBMITTED,
                    )
                )
            )
            .scalars()
            .all()
        )
        filled = 0
        now = datetime.now(UTC)
        for order in working:
            candle = await self._store.latest_candle(
                uuid.UUID(order.broker_ticker), Interval.D1, closed_only=True
            )
            if candle is None or order.stop_price is None:
                continue
            breached = order.side is OrderSide.SELL and Decimal(candle.low) <= Decimal(
                order.stop_price
            )
            if not breached:
                continue
            position = await self._position(account.id, order.broker_ticker)
            if position is None or Decimal(position.quantity) <= 0:
                # Nothing to close; retire the orphaned stop.
                order.status = OrderStatus.CANCELLED
                order.terminal_at = now
                continue
            sell_qty = min(Decimal(order.quantity), Decimal(position.quantity))
            stop_price = Decimal(order.stop_price)
            account.cash = Decimal(account.cash) + stop_price * sell_qty
            position.quantity = Decimal(position.quantity) - sell_qty
            order.status = OrderStatus.FILLED
            order.filled_quantity = sell_qty
            order.average_fill_price = stop_price
            order.terminal_at = now
            filled += 1
        await self._session.flush()
        return filled

    async def reconcile(self) -> ReconciliationResult:
        """Compare venue truth against local `TradeIntent` records (§10)."""
        account = await self._ensure_account()
        orders = (
            (
                await self._session.execute(
                    select(PaperBrokerOrder).where(
                        PaperBrokerOrder.account_id == account.id
                    )
                )
            )
            .scalars()
            .all()
        )
        positions = await self.get_positions()
        intents = (
            (
                await self._session.execute(
                    select(TradeIntent).where(TradeIntent.broker == self.kind)
                )
            )
            .scalars()
            .all()
        )

        venue_order_ids = {o.broker_order_id for o in orders}
        discrepancies: list[ReconciliationDiscrepancy] = []
        for intent in intents:
            if (
                intent.status
                in {TradeIntentStatus.SUBMITTED, TradeIntentStatus.RECONCILED}
                and intent.broker_order_id
                and intent.broker_order_id not in venue_order_ids
            ):
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        kind="missing_order",
                        broker_ticker=str(intent.instrument_id),
                        detail=(
                            "Local intent references a broker order the venue does not have."
                        ),
                        local_value=intent.broker_order_id,
                        broker_value=None,
                    )
                )
        return ReconciliationResult(
            broker=self.kind,
            reconciled_at=datetime.now(UTC),
            positions_checked=len(positions),
            orders_checked=len(orders),
            discrepancies=discrepancies,
        )

    async def health_check(self) -> bool:
        try:
            await self._ensure_account()
        except Exception:
            return False
        return True

    def _to_dto(self, order: PaperBrokerOrder) -> BrokerOrder:
        return BrokerOrder(
            broker_order_id=order.broker_order_id,
            broker_ticker=order.broker_ticker,
            side=order.side,
            order_type=order.order_type,
            quantity=Decimal(order.quantity),
            status=order.status,
            filled_quantity=Decimal(order.filled_quantity),
            average_fill_price=Decimal(order.average_fill_price)
            if order.average_fill_price is not None
            else None,
            limit_price=Decimal(order.limit_price)
            if order.limit_price is not None
            else None,
            stop_price=Decimal(order.stop_price) if order.stop_price is not None else None,
            created_at=order.placed_at,
            updated_at=order.terminal_at or order.placed_at,
            client_reference=order.client_reference,
        )
