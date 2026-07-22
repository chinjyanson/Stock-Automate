"""Approve → risk → paper fill, end to end against real PostgreSQL.

The Phase 2 seam (approval was a dead end) is now closed: an approved proposal is
sized by the risk engine, filled on the internal paper venue, and protected by a
broker-side stop — or, when the engine refuses, parked in REJECTED_BY_RISK so the
"user said yes, system said no" outcome is visible rather than silent.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.broker.internal_paper import InternalPaperBroker
from app.models.enums import HaltKind, HaltScope, OrderType, TradeIntentStatus
from app.models.instrument import Instrument
from app.models.risk import RiskConfiguration, TradeIntent
from app.models.scanner import ProposalStatus, ScannerResult
from app.risk.execution import ExecutionError, ExecutionService
from app.risk.halts import HaltService
from app.scanner.engine import ScannerEngine
from app.scanner.proposals import ProposalInputs, ProposalService

pytestmark = pytest.mark.asyncio


async def _seed_config(db: object, **overrides: object) -> None:
    db.add(RiskConfiguration(name="default", is_active=True, **overrides))  # type: ignore[attr-defined]
    await db.flush()  # type: ignore[attr-defined]


async def _proposal_id(db: object, instrument: Instrument, approver: uuid.UUID):  # type: ignore[no-untyped-def]
    await ScannerEngine(db).run([instrument])  # type: ignore[arg-type]
    await db.commit()  # type: ignore[attr-defined]
    result = (
        await db.execute(  # type: ignore[attr-defined]
            select(ScannerResult).where(ScannerResult.instrument_id == instrument.id)
        )
    ).scalar_one()
    proposal = await ProposalService(db).propose_from_result(  # type: ignore[arg-type]
        result, ProposalInputs(account_equity=Decimal("100000"))
    )
    await db.commit()  # type: ignore[attr-defined]
    approved = await ProposalService(db).approve(proposal.id, actor_user_id=approver)  # type: ignore[arg-type]
    await db.commit()  # type: ignore[attr-defined]
    return approved


class TestHappyPath:
    async def test_approved_proposal_fills_and_gets_a_stop(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await _seed_config(db)
        proposal = await _proposal_id(db, candled_instrument, approver)

        executed = await ExecutionService(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).execute_approved(
            proposal, actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]
        assert executed.status is ProposalStatus.EXECUTED

        # The intent is the durable record, reconciled after a synchronous fill.
        intent = (
            await db.execute(  # type: ignore[attr-defined]
                select(TradeIntent).where(TradeIntent.proposal_id == proposal.id)
            )
        ).scalar_one()
        assert intent.status is TradeIntentStatus.RECONCILED
        assert intent.broker_order_id is not None
        assert intent.filled_quantity is not None
        assert intent.stop_broker_order_id is not None

        # The venue holds the position and a resting protective stop.
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        positions = await broker.get_positions()
        assert len(positions) == 1
        pending = await broker.get_pending_orders()
        assert any(o.order_type is OrderType.STOP for o in pending)

    async def test_execution_is_idempotent(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await _seed_config(db)
        proposal = await _proposal_id(db, candled_instrument, approver)
        service = ExecutionService(db, broker=InternalPaperBroker(db))  # type: ignore[arg-type]

        await service.execute_approved(proposal, actor_user_id=approver)
        await db.commit()  # type: ignore[attr-defined]
        # A second call must not submit a second order.
        await service.execute_approved(proposal, actor_user_id=approver)
        await db.commit()  # type: ignore[attr-defined]

        intents = (
            (
                await db.execute(  # type: ignore[attr-defined]
                    select(TradeIntent).where(TradeIntent.proposal_id == proposal.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(intents) == 1


class TestRiskRefusal:
    async def test_a_halt_parks_the_proposal_in_rejected_by_risk(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await _seed_config(db)
        proposal = await _proposal_id(db, candled_instrument, approver)
        await HaltService(db).activate(  # type: ignore[arg-type]
            HaltKind.KILL_SWITCH, "stop everything", scope=HaltScope.GLOBAL, actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]

        executed = await ExecutionService(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).execute_approved(
            proposal, actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]
        assert executed.status is ProposalStatus.REJECTED_BY_RISK

        # Nothing reached the venue.
        broker = InternalPaperBroker(db)  # type: ignore[arg-type]
        assert await broker.get_positions() == []

    async def test_missing_configuration_rejects(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        # No RiskConfiguration seeded — the engine fails closed.
        proposal = await _proposal_id(db, candled_instrument, approver)
        executed = await ExecutionService(
            db,  # type: ignore[arg-type]
            broker=InternalPaperBroker(db),  # type: ignore[arg-type]
        ).execute_approved(
            proposal, actor_user_id=approver
        )
        await db.commit()  # type: ignore[attr-defined]
        assert executed.status is ProposalStatus.REJECTED_BY_RISK

    async def test_only_approved_proposals_execute(
        self, db: object, candled_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await _seed_config(db)
        await ScannerEngine(db).run([candled_instrument])  # type: ignore[arg-type]
        await db.commit()  # type: ignore[attr-defined]
        result = (
            await db.execute(  # type: ignore[attr-defined]
                select(ScannerResult).where(ScannerResult.instrument_id == candled_instrument.id)
            )
        ).scalar_one()
        proposal = await ProposalService(db).propose_from_result(  # type: ignore[arg-type]
            result, ProposalInputs(account_equity=Decimal("100000"))
        )
        await db.commit()  # type: ignore[attr-defined]

        # Pending, not approved — execution must refuse.
        with pytest.raises(ExecutionError, match="APPROVED"):
            await ExecutionService(
                db,  # type: ignore[arg-type]
                broker=InternalPaperBroker(db),  # type: ignore[arg-type]
            ).execute_approved(proposal, actor_user_id=approver)
