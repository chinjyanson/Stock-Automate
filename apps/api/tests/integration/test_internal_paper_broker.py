"""Internal paper broker against real PostgreSQL.

The venue is DB-backed, so these prove the properties an in-memory stand-in
could not: cash and positions persist across broker instances, fills are priced
from the local candle store, and protective stops rest until a candle triggers
them.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.broker.internal_paper import InternalPaperBroker
from app.broker.types import (
    BrokerOrderRejectedError,
    BrokerOrderRequest,
)
from app.data.store import CandleStore
from app.models.enums import Interval, OrderSide, OrderStatus, OrderType
from app.models.instrument import Instrument

pytestmark = pytest.mark.asyncio


async def _last_close(db: object, instrument: Instrument) -> Decimal:
    candle = await CandleStore(db).latest_candle(  # type: ignore[arg-type]
        instrument.id, Interval.D1, closed_only=True
    )
    assert candle is not None
    return Decimal(candle.close)


class TestMarketFills:
    async def test_market_buy_fills_at_last_close_and_debits_cash(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        price = await _last_close(db, candled_instrument)

        order = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(candled_instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("2"),
                order_type=OrderType.MARKET,
            )
        )
        await db.commit()  # type: ignore[attr-defined]

        assert order.status is OrderStatus.FILLED
        assert order.average_fill_price == price
        account = await broker.get_account()
        # Cash is a Money column (4 dp), so compare at that stored precision.
        expected_cash = (Decimal("100000") - price * Decimal("2")).quantize(Decimal("0.0001"))
        assert account.cash == expected_cash

        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == Decimal("2")
        assert positions[0].average_price == price

    async def test_insufficient_cash_is_rejected(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("1"))  # type: ignore[arg-type]
        with pytest.raises(BrokerOrderRejectedError, match="Insufficient cash"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker=str(candled_instrument.id),
                    side=OrderSide.BUY,
                    quantity=Decimal("10"),
                    order_type=OrderType.MARKET,
                )
            )

    async def test_cannot_sell_more_than_held(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        with pytest.raises(BrokerOrderRejectedError, match="Cannot sell"):
            await broker.place_order(
                BrokerOrderRequest(
                    broker_ticker=str(candled_instrument.id),
                    side=OrderSide.SELL,
                    quantity=Decimal("1"),
                    order_type=OrderType.MARKET,
                )
            )


class TestPersistence:
    async def test_state_survives_a_new_broker_instance(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        first = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        await first.place_order(
            BrokerOrderRequest(
                broker_ticker=str(candled_instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("3"),
                order_type=OrderType.MARKET,
            )
        )
        await db.commit()  # type: ignore[attr-defined]

        # A brand-new broker reading the same database sees the position — proof
        # the state is in Postgres, not process memory.
        second = InternalPaperBroker(db)  # type: ignore[arg-type]
        positions = await second.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == Decimal("3")


class TestStops:
    async def test_resting_stop_persists_then_triggers_on_breach(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        price = await _last_close(db, candled_instrument)

        await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(candled_instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("2"),
                order_type=OrderType.MARKET,
            )
        )
        # A stop at the last close is trivially breached (low <= close), which
        # lets the test trigger it deterministically without new candles.
        stop = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(candled_instrument.id),
                side=OrderSide.SELL,
                quantity=Decimal("2"),
                order_type=OrderType.STOP,
                stop_price=price,
            )
        )
        await db.commit()  # type: ignore[attr-defined]

        # It rests, not fills, on placement.
        assert stop.status is OrderStatus.SUBMITTED
        pending = await broker.get_pending_orders()
        assert any(o.broker_order_id == stop.broker_order_id for o in pending)

        filled = await broker.process_stops()
        await db.commit()  # type: ignore[attr-defined]
        assert filled == 1

        # Position closed, and the stop is no longer working.
        positions = await broker.get_positions()
        assert positions == []
        pending_after = await broker.get_pending_orders()
        assert pending_after == []


class TestReconciliation:
    async def test_reconcile_is_clean_with_no_intents(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(candled_instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                order_type=OrderType.MARKET,
            )
        )
        await db.commit()  # type: ignore[attr-defined]

        result = await broker.reconcile()
        assert result.is_clean
        assert result.positions_checked == 1
