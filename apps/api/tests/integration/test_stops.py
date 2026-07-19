"""Stop management against real PostgreSQL.

Trailing only ever raises the stop, time stops fire once, triggered stops close
the local intent, and an emergency exit flattens everything — the behaviours a
protective stop is worthless without.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.broker.internal_paper import InternalPaperBroker
from app.broker.types import BrokerOrderRequest
from app.data.store import CandleStore
from app.models.enums import Interval, OrderSide, OrderType, TradeIntentStatus
from app.models.instrument import Instrument
from app.models.risk import RiskConfiguration, TradeIntent
from app.risk.stops import StopService

pytestmark = pytest.mark.asyncio


async def _config(db: object, **overrides: object) -> RiskConfiguration:
    """A persisted config so its column defaults (ATR multiplier, etc.) apply."""
    config = RiskConfiguration(name="c", is_active=True, **overrides)
    db.add(config)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return config


async def _last_close(db: object, instrument: Instrument) -> Decimal:
    candle = await CandleStore(db).latest_candle(  # type: ignore[arg-type]
        instrument.id, Interval.D1, closed_only=True
    )
    assert candle is not None
    return Decimal(candle.close)


async def _open_position(
    db: object, instrument: Instrument, *, stop_price: Decimal, quantity: Decimal = Decimal("2")
) -> tuple[TradeIntent, InternalPaperBroker]:
    """Buy, rest a stop, and record the RECONCILED intent that links them."""
    broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
    buy = await broker.place_order(
        BrokerOrderRequest(
            broker_ticker=str(instrument.id),
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
    )
    stop = await broker.place_order(
        BrokerOrderRequest(
            broker_ticker=str(instrument.id),
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
        )
    )
    now = datetime.now(UTC)
    intent = TradeIntent(
        instrument_id=instrument.id,
        broker=broker.kind,
        client_reference=uuid.uuid4(),
        status=TradeIntentStatus.RECONCILED,
        side=OrderSide.BUY,
        quantity=quantity,
        broker_order_id=buy.broker_order_id,
        submitted_at=now,
        reconciled_at=now,
        filled_quantity=quantity,
        filled_price=buy.average_fill_price,
        stop_price=stop_price,
        stop_broker_order_id=stop.broker_order_id,
    )
    db.add(intent)  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]
    return intent, broker


class TestTrailing:
    async def test_trailing_raises_a_stop_that_is_below_the_candidate(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        # An artificially low starting stop must be ratcheted up to the ATR band.
        intent, broker = await _open_position(db, candled_instrument, stop_price=Decimal("1"))
        old_order_id = intent.stop_broker_order_id
        config = await _config(db, trailing_stop_enabled=True)

        result = await StopService(db).manage(broker, config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result["trailed"] == 1
        assert Decimal(intent.stop_price) > Decimal("1")
        # A fresh resting stop replaced the old one.
        assert intent.stop_broker_order_id != old_order_id

    async def test_trailing_never_lowers_a_stop(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        price = await _last_close(db, candled_instrument)
        # A stop already at the last close is above the ATR candidate below it.
        intent, broker = await _open_position(db, candled_instrument, stop_price=price)
        config = await _config(db, trailing_stop_enabled=True)

        result = await StopService(db).manage(broker, config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result["trailed"] == 0
        assert Decimal(intent.stop_price) == price


class TestTriggered:
    async def test_a_breached_stop_closes_the_intent(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        price = await _last_close(db, candled_instrument)
        # A stop at the last close is trivially breached (low <= close).
        intent, broker = await _open_position(db, candled_instrument, stop_price=price)
        config = await _config(db, trailing_stop_enabled=False)

        result = await StopService(db).manage(broker, config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result["triggered"] == 1
        assert result["closed"] == 1
        assert intent.closed_at is not None
        assert intent.exit_reason == "stop"
        assert await broker.get_positions() == []


class TestTimeStop:
    async def test_a_position_past_its_horizon_is_exited(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        intent, broker = await _open_position(db, candled_instrument, stop_price=Decimal("1"))
        # Backdate the entry so the time stop applies.
        intent.submitted_at = datetime.now(UTC) - timedelta(days=10)
        await db.flush()  # type: ignore[attr-defined]
        config = RiskConfiguration(
            name="c", is_active=True, trailing_stop_enabled=False, max_holding_days=5
        )

        result = await StopService(db).manage(broker, config)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result["time_exits"] == 1
        assert intent.closed_at is not None
        assert intent.exit_reason == "time_stop"
        assert await broker.get_positions() == []


class TestEmergencyExit:
    async def test_flatten_closes_every_position(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        intent, broker = await _open_position(db, candled_instrument, stop_price=Decimal("1"))

        closed = await StopService(db).emergency_exit_all(  # type: ignore[arg-type]
            broker, "panic", actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]

        assert closed == 1
        assert intent.closed_at is not None
        assert intent.exit_reason == "emergency"
        assert await broker.get_positions() == []
