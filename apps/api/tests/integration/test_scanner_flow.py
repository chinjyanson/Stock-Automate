"""Scanner → proposal → approval flow against real PostgreSQL.

Uses the deterministic mock provider for candles so the scan is reproducible,
and drives the full workflow the way the API does: ingest, scan, propose,
approve/reject/expire. The money-adjacent invariants (no duplicate pending
proposal, approval revalidation, expiry) are the focus.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.mock_provider import MockMarketDataProvider
from app.models.enums import InstrumentKind, LifecycleState, PriceUnit, ProviderKind
from app.models.instrument import Exchange, Instrument, MarketDataMapping
from app.models.scanner import ProposalStatus
from app.scanner.engine import ScannerEngine
from app.scanner.proposals import ProposalError, ProposalInputs, ProposalService
from app.services.ingestion import IngestionService


@pytest.fixture
async def approver(db: AsyncSession) -> uuid.UUID:
    """A real user row — approvals FK to it, so a random UUID is not enough."""
    from app.models.user import User

    user = User(
        id=uuid.uuid4(),
        email="approver@example.com",
        password_hash="x",
        is_admin=True,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    return user.id


@pytest.fixture
async def scannable_instrument(db: AsyncSession) -> Instrument:
    """An instrument mapped to the mock provider with a year of candles."""
    exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
    db.add(exchange)
    await db.flush()

    instrument = Instrument(
        id=uuid.uuid4(),
        isin="US0378331005",
        exchange_id=exchange.id,
        exchange_ticker="AAPL",
        name="Apple Inc.",
        kind=InstrumentKind.STOCK,
        currency="USD",
        price_unit=PriceUnit.USD,
        lifecycle_state=LifecycleState.DISCOVERED,
        is_scanner_eligible=True,
    )
    db.add(instrument)
    await db.flush()
    db.add(
        MarketDataMapping(
            instrument_id=instrument.id,
            provider=ProviderKind.MOCK,
            provider_symbol="AAPL",
            is_signal_source=True,
            confirmed_by_user=True,
        )
    )
    await db.flush()

    await IngestionService(db).ingest_daily(instrument, MockMarketDataProvider(), backfill_days=400)
    await db.commit()
    return instrument


class TestScanning:
    async def test_scan_produces_a_result_with_provenance(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        summary = await ScannerEngine(db).run([scannable_instrument])
        await db.commit()

        assert summary.scored == 1
        from app.models.scanner import ScannerResult

        result = (
            await db.execute(
                select(ScannerResult).where(ScannerResult.instrument_id == scannable_instrument.id)
            )
        ).scalar_one()

        assert 0 <= float(result.core_score) <= 100
        assert result.candles_used > 200
        # Provenance is populated, not null.
        assert result.confidence is not None
        assert result.data_completeness is not None
        assert result.positive_signals is not None

    async def test_scan_updates_last_scanned_at(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        assert scannable_instrument.last_scanned_at is None
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        await db.refresh(scannable_instrument)
        assert scannable_instrument.last_scanned_at is not None

    async def test_instrument_with_no_candles_is_skipped_not_failed(self, db: AsyncSession) -> None:
        bare = Instrument(
            id=uuid.uuid4(),
            isin="US_BARE",
            exchange_ticker="BARE",
            name="No Data Co.",
            kind=InstrumentKind.STOCK,
            currency="USD",
            price_unit=PriceUnit.USD,
        )
        db.add(bare)
        await db.flush()

        summary = await ScannerEngine(db).run([bare])
        await db.commit()
        # Skipped, not scored — and not recorded as a low score (§6).
        assert summary.scored == 0
        assert summary.skipped == 1


class TestProposalWorkflow:
    async def _latest_result(self, db: AsyncSession, instrument_id: uuid.UUID):  # type: ignore[no-untyped-def]
        from app.models.scanner import ScannerResult

        return (
            await db.execute(
                select(ScannerResult).where(ScannerResult.instrument_id == instrument_id)
            )
        ).scalar_one()

    async def test_proposal_is_volatility_sized(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        result = await self._latest_result(db, scannable_instrument.id)

        proposal = await ProposalService(db).propose_from_result(
            result, ProposalInputs(account_equity=Decimal("10000"))
        )
        await db.commit()

        # Risk is ~1% of equity by construction; a stop sits below entry.
        assert proposal.risk_pct <= Decimal("0.02")
        assert proposal.proposed_stop_price < proposal.indicative_entry_price
        assert proposal.proposed_quantity > 0
        assert proposal.status is ProposalStatus.PENDING_APPROVAL

    async def test_duplicate_pending_proposal_is_refused(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        result = await self._latest_result(db, scannable_instrument.id)
        service = ProposalService(db)

        await service.propose_from_result(result, ProposalInputs(account_equity=Decimal("10000")))
        await db.commit()

        with pytest.raises(ProposalError, match="already exists"):
            await service.propose_from_result(
                result, ProposalInputs(account_equity=Decimal("10000"))
            )

    async def test_approval_requires_a_pending_proposal(
        self, db: AsyncSession, scannable_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        result = await self._latest_result(db, scannable_instrument.id)
        service = ProposalService(db)

        proposal = await service.propose_from_result(
            result, ProposalInputs(account_equity=Decimal("10000"))
        )
        await db.commit()

        approved = await service.approve(proposal.id, actor_user_id=approver)
        await db.commit()
        assert approved.status is ProposalStatus.APPROVED
        assert approved.decided_at is not None

        # A second approval is refused — it is no longer pending.
        with pytest.raises(ProposalError, match="not pending"):
            await service.approve(proposal.id, actor_user_id=approver)

    async def test_expired_proposal_cannot_be_approved(
        self, db: AsyncSession, scannable_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        result = await self._latest_result(db, scannable_instrument.id)
        service = ProposalService(db)

        proposal = await service.propose_from_result(
            result, ProposalInputs(account_equity=Decimal("10000"), approval_ttl_minutes=60)
        )
        # Force expiry.
        proposal.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        await db.commit()

        with pytest.raises(ProposalError, match="expired"):
            await service.approve(proposal.id, actor_user_id=approver)
        await db.commit()

        await db.refresh(proposal)
        assert proposal.status is ProposalStatus.EXPIRED

    async def test_expire_stale_transitions_pending_proposals(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        result = await self._latest_result(db, scannable_instrument.id)
        service = ProposalService(db)

        proposal = await service.propose_from_result(
            result, ProposalInputs(account_equity=Decimal("10000"))
        )
        proposal.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()

        count = await service.expire_stale()
        await db.commit()
        assert count == 1
        await db.refresh(proposal)
        assert proposal.status is ProposalStatus.EXPIRED

    async def test_rejection_records_the_decision(
        self, db: AsyncSession, scannable_instrument: Instrument, approver: uuid.UUID
    ) -> None:
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()
        result = await self._latest_result(db, scannable_instrument.id)
        service = ProposalService(db)

        proposal = await service.propose_from_result(
            result, ProposalInputs(account_equity=Decimal("10000"))
        )
        await db.commit()

        rejected = await service.reject(proposal.id, actor_user_id=approver, note="not now")
        await db.commit()
        assert rejected.status is ProposalStatus.REJECTED
        assert rejected.decision_note == "not now"


class TestAuditTrail:
    async def test_scan_and_proposal_are_audited(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        from app.audit.service import AuditService
        from app.models.enums import AuditEventKind

        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()

        events = await AuditService(db).recent(kind=AuditEventKind.SCANNER_RUN_COMPLETED)
        assert len(events) >= 1
