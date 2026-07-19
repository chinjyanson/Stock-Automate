"""Scheduled reconciliation against real PostgreSQL.

A clean reconciliation resolves pending intents and clears a reconciliation halt
the engine raised; a divergence raises a global halt and fails closed. A human's
kill switch is never touched by the auto-clear.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.broker.internal_paper import InternalPaperBroker
from app.broker.types import BrokerOrderRequest
from app.models.enums import (
    HaltKind,
    HaltScope,
    OrderSide,
    OrderType,
    TradeIntentStatus,
)
from app.models.instrument import Instrument
from app.models.risk import TradeIntent
from app.risk.halts import HaltService
from app.services.reconciliation import ReconciliationService

pytestmark = pytest.mark.asyncio


class TestDivergence:
    async def test_a_missing_order_raises_a_reconciliation_halt(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        # An intent that claims a broker order the venue has never seen.
        db.add(  # type: ignore[attr-defined]
            TradeIntent(
                instrument_id=candled_instrument.id,
                broker=broker.kind,
                client_reference=uuid.uuid4(),
                status=TradeIntentStatus.SUBMITTED,
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                broker_order_id="ghost-order",
            )
        )
        await db.flush()  # type: ignore[attr-defined]

        result = await ReconciliationService(db).run(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert not result.is_clean
        halts = await HaltService(db).active_halts()  # type: ignore[arg-type]
        assert any(h.kind is HaltKind.RECONCILIATION for h in halts)


class TestClean:
    async def test_clean_reconciliation_clears_a_system_halt(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        # A pre-existing engine-raised reconciliation halt.
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.RECONCILIATION, "earlier divergence", scope=HaltScope.GLOBAL
        )
        await db.flush()  # type: ignore[attr-defined]

        result = await ReconciliationService(db).run(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result.is_clean
        active = await HaltService(db).active_halts()  # type: ignore[arg-type]
        assert not any(h.kind is HaltKind.RECONCILIATION for h in active)

    async def test_clean_reconciliation_leaves_a_user_kill_switch_alone(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        await HaltService(db).kill_switch("human stop", actor_user_id=approver)  # type: ignore[arg-type]
        await db.flush()  # type: ignore[attr-defined]

        await ReconciliationService(db).run(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        active = await HaltService(db).active_halts()  # type: ignore[arg-type]
        assert any(h.kind is HaltKind.KILL_SWITCH for h in active)

    async def test_clean_reconciliation_resolves_a_pending_intent(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        broker = InternalPaperBroker(db, starting_cash=Decimal("100000"))  # type: ignore[arg-type]
        order = await broker.place_order(
            BrokerOrderRequest(
                broker_ticker=str(candled_instrument.id),
                side=OrderSide.BUY,
                quantity=Decimal("1"),
                order_type=OrderType.MARKET,
            )
        )
        # An intent left in the ambiguous state, but whose order the venue holds.
        intent = TradeIntent(
            instrument_id=candled_instrument.id,
            broker=broker.kind,
            client_reference=uuid.uuid4(),
            status=TradeIntentStatus.RECONCILIATION_REQUIRED,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            broker_order_id=order.broker_order_id,
        )
        db.add(intent)  # type: ignore[attr-defined]
        await db.flush()  # type: ignore[attr-defined]

        result = await ReconciliationService(db).run(broker)  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]

        assert result.is_clean
        refreshed = (
            await db.execute(  # type: ignore[attr-defined]
                select(TradeIntent).where(TradeIntent.id == intent.id)
            )
        ).scalar_one()
        assert refreshed.status is TradeIntentStatus.RECONCILED
        assert refreshed.reconciled_at is not None
