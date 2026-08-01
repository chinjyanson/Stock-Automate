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
from app.models.scanner import Classification, ProposalStatus
from app.scanner import watchlist
from app.scanner.engine import ScannerEngine
from app.scanner.proposals import ProposalError, ProposalInputs, ProposalService
from app.scanner.rotation import (
    _ROTATION_POOL_LIMIT,
    _top_ranked_ids,
    select_instruments,
)
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


class TestRotationSelection:
    """The rotating sweep must not be crowded out by unscannable instruments."""

    def _bare_instruments(self, count: int) -> list[Instrument]:
        """Scanner-eligible instruments with no candles and no prior scan."""
        return [
            Instrument(
                id=uuid.uuid4(),
                isin=f"US_BARE_{i}",
                exchange_ticker=f"BARE{i}",
                name=f"BARE{i} Co.",
                kind=InstrumentKind.STOCK,
                currency="USD",
                price_unit=PriceUnit.USD,
                is_scanner_eligible=True,
            )
            for i in range(count)
        ]

    async def test_scannable_instrument_selected_even_when_history_less_dominate(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        """Regression: the sweep is restricted to instruments that can be scored.

        The scannable instrument has history but a recent scan time, so under
        `last_scanned_at ASC NULLS FIRST` it sorts *after* every never-scanned,
        history-less instrument. With more history-less instruments than the
        sweep's pool cap, the pre-fix query filled its whole window with
        unscannable rows and dropped the only scorable one entirely.
        """
        scannable_instrument.last_scanned_at = datetime.now(UTC)
        # One more than the pool cap, so an unfiltered sweep is 100% saturated
        # by never-scanned, history-less instruments (which sort first).
        db.add_all(self._bare_instruments(_ROTATION_POOL_LIMIT + 1))
        await db.commit()

        chosen, _ = await select_instruments(db, configuration=None, limit=200)

        chosen_ids = {c.id for c in chosen}
        # The scorable instrument survives; none of the history-less ones appear.
        assert chosen_ids == {scannable_instrument.id}


class TestExploreExploit:
    """The rotation guarantees the best band is re-scored, then explores."""

    async def _scored(
        self, db: AsyncSession, ticker: str, primary: float, core: float, days_ago: int = 0
    ) -> Instrument:
        """An instrument with candles and one scanner result at a given age."""
        from app.models.scanner import ScannerResult, ScannerRun, ScannerRunStatus

        exchange = (
            await db.execute(select(Exchange).where(Exchange.mic == "XNAS"))
        ).scalar_one_or_none()
        if exchange is None:
            exchange = Exchange(
                mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York"
            )
            db.add(exchange)
            await db.flush()
        inst = Instrument(
            id=uuid.uuid4(),
            exchange_id=exchange.id,
            exchange_ticker=ticker,
            name=f"{ticker} Co.",
            kind=InstrumentKind.STOCK,
            currency="USD",
            price_unit=PriceUnit.USD,
            is_scanner_eligible=True,
        )
        db.add(inst)
        await db.flush()
        db.add(
            MarketDataMapping(
                instrument_id=inst.id,
                provider=ProviderKind.MOCK,
                provider_symbol=ticker,
                is_signal_source=True,
                confirmed_by_user=True,
            )
        )
        await db.flush()
        await IngestionService(db).ingest_daily(inst, MockMarketDataProvider(), backfill_days=120)
        run = ScannerRun(
            status=ScannerRunStatus.COMPLETED,
            started_at=datetime.now(UTC) - timedelta(days=days_ago),
        )
        db.add(run)
        await db.flush()
        result = ScannerResult(
            run_id=run.id,
            instrument_id=inst.id,
            primary_score=Decimal(str(primary)),
            core_score=Decimal(str(core)),
            trend_score=Decimal("50"),
            momentum_score=Decimal("50"),
            risk_score=Decimal("50"),
            liquidity_score=Decimal("50"),
            positioning_score=Decimal("50"),
            classification=Classification.DOES_NOT_PASS,
            data_completeness=Decimal("1"),
            confidence=Decimal("1"),
            candles_used=120,
        )
        db.add(result)
        # created_at drives recency; set it explicitly for the stale-score test.
        result.created_at = datetime.now(UTC) - timedelta(days=days_ago)
        await db.flush()
        return inst

    async def test_ranks_by_primary_score_not_the_momentum_core(self, db: AsyncSession) -> None:
        """The tier re-verifying "the best" must use the score that defines best.

        Ordering by `core_score` — as this did — sorted the re-check tier by a
        different number than the one deciding the ranking and the strategy's
        universe.
        """
        good = await self._scored(db, "GOODPRI", primary=90, core=10)
        other = await self._scored(db, "HIGHCORE", primary=20, core=99)
        await db.commit()

        top = await _top_ranked_ids(db, 1)
        assert top == [good.id]
        assert other.id not in top

    async def _add_result(
        self, db: AsyncSession, inst: Instrument, primary: float, days_ago: int
    ) -> None:
        """A second scanner result for an instrument already scored."""
        from app.models.scanner import ScannerResult, ScannerRun, ScannerRunStatus

        run = ScannerRun(
            status=ScannerRunStatus.COMPLETED,
            started_at=datetime.now(UTC) - timedelta(days=days_ago),
        )
        db.add(run)
        await db.flush()
        result = ScannerResult(
            run_id=run.id,
            instrument_id=inst.id,
            primary_score=Decimal(str(primary)),
            core_score=Decimal("50"),
            trend_score=Decimal("50"),
            momentum_score=Decimal("50"),
            risk_score=Decimal("50"),
            liquidity_score=Decimal("50"),
            positioning_score=Decimal("50"),
            classification=Classification.DOES_NOT_PASS,
            data_completeness=Decimal("1"),
            confidence=Decimal("1"),
            candles_used=120,
        )
        db.add(result)
        result.created_at = datetime.now(UTC) - timedelta(days=days_ago)
        await db.flush()

    async def test_a_demoted_stock_stops_being_treated_as_top_ranked(
        self, db: AsyncSession
    ) -> None:
        """One instrument, two scores: a stale 95 and a fresh 10.

        De-duplicating a score-ordered history keeps the highest score an
        instrument ever had, so re-scanning it could never demote it — it would
        occupy an exploit slot forever on the strength of one good night.
        """
        faded = await self._scored(db, "FADED", primary=95, core=50, days_ago=21)
        await self._add_result(db, faded, primary=10, days_ago=0)
        steady = await self._scored(db, "STEADY", primary=60, core=50, days_ago=0)
        await db.commit()

        top = await _top_ranked_ids(db, 5)

        # Ranked on its latest score (10), so it sits below the steady 60.
        assert top.index(steady.id) < top.index(faded.id)

    async def test_the_selection_reason_reports_the_split(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        """Operators should be able to see how the budget was spent."""
        await db.commit()
        _, reason = await select_instruments(db, configuration=None, limit=50)
        assert "top-ranked re-scored" in reason
        assert "explored" in reason


class TestWatchlistSelection:
    """A pinned instrument is selected next run — that is the whole promise."""

    async def test_pinned_instrument_leads_the_selection(
        self, db: AsyncSession, approver: uuid.UUID, scannable_instrument: Instrument
    ) -> None:
        """Freshly scanned, so the sweep would rank it last — the pin overrides that.

        Without the pin, `last_scanned_at ASC` puts an instrument scanned seconds
        ago behind everything else; under a tight cap it would not be scanned
        again for days.
        """
        scannable_instrument.last_scanned_at = datetime.now(UTC)
        await watchlist.add(db, approver, scannable_instrument.id)
        await db.commit()

        chosen, reason = await select_instruments(db, configuration=None, limit=200)

        assert chosen[0].id == scannable_instrument.id
        assert "1 watchlisted" in reason

    async def test_pin_survives_a_cap_that_would_otherwise_exclude_it(
        self, db: AsyncSession, approver: uuid.UUID, scannable_instrument: Instrument
    ) -> None:
        """A one-instrument scan still spends its single slot on the pinned one."""
        scannable_instrument.last_scanned_at = datetime.now(UTC)
        db.add_all(self._other_scannables(3))
        await watchlist.add(db, approver, scannable_instrument.id)
        await db.commit()

        chosen, _ = await select_instruments(db, configuration=None, limit=1)

        assert [c.id for c in chosen] == [scannable_instrument.id]

    async def test_pinned_without_history_is_excluded_and_reported(
        self, db: AsyncSession, approver: uuid.UUID, scannable_instrument: Instrument
    ) -> None:
        """The one case where "definitely rescanned" cannot hold, said out loud.

        An instrument with no stored candles cannot be scored, so pinning it
        does not conjure data. Silently dropping it is what makes this look like
        a bug; the selection reason names it instead.
        """
        bare = Instrument(
            id=uuid.uuid4(),
            exchange_ticker="NOHIST",
            name="No History Co.",
            kind=InstrumentKind.STOCK,
            currency="USD",
            price_unit=PriceUnit.USD,
            is_scanner_eligible=True,
        )
        db.add(bare)
        await db.flush()
        await watchlist.add(db, approver, bare.id)
        await db.commit()

        chosen, reason = await select_instruments(db, configuration=None, limit=200)

        assert bare.id not in {c.id for c in chosen}
        assert "1 watchlisted excluded: too little stored history to score" in reason

    async def test_listing_loads_the_venue_without_lazy_io(
        self, db: AsyncSession, approver: uuid.UUID, scannable_instrument: Instrument
    ) -> None:
        """Regression: the read path must not touch a lazy relationship.

        `list_entries` eager-loads the instrument; the route then reads
        `instrument.exchange.mic`. When that second hop was left lazy, asyncio
        SQLAlchemy raised `MissingGreenlet` instead of querying, so every GET
        500'd — and the page, which swallowed the failure, showed an empty
        watchlist. The pin was in the database the whole time.

        `expunge_all` matters: with the objects still in the identity map the
        attribute resolves from memory and the missing eager load stays hidden.
        """
        await watchlist.add(db, approver, scannable_instrument.id)
        await db.commit()
        db.expunge_all()

        entries = await watchlist.list_entries(db, approver)

        assert len(entries) == 1
        _, instrument = entries[0]
        assert instrument is not None
        # Exactly what the route handler does next.
        assert instrument.exchange is not None
        assert instrument.exchange.mic == "XNAS"

    async def test_listing_an_absent_watchlist_creates_nothing(
        self, db: AsyncSession, approver: uuid.UUID
    ) -> None:
        """A read must not write. The GET handler never commits, so a row created
        here would be rolled back on every request — churn with nothing to show."""
        assert await watchlist.list_entries(db, approver) == []
        assert await watchlist.find_default(db, approver) is None

    async def test_pinning_is_idempotent(
        self, db: AsyncSession, approver: uuid.UUID, scannable_instrument: Instrument
    ) -> None:
        _, first = await watchlist.add(db, approver, scannable_instrument.id)
        _, second = await watchlist.add(db, approver, scannable_instrument.id)
        await db.commit()

        assert first is True
        assert second is False
        assert await watchlist.watchlisted_instrument_ids(db) == [scannable_instrument.id]

    async def test_unpinning_returns_it_to_the_ordinary_rotation(
        self, db: AsyncSession, approver: uuid.UUID, scannable_instrument: Instrument
    ) -> None:
        await watchlist.add(db, approver, scannable_instrument.id)
        await db.commit()
        removed = await watchlist.remove(db, approver, scannable_instrument.id)
        await db.commit()

        assert removed is True
        assert await watchlist.watchlisted_instrument_ids(db) == []
        _, reason = await select_instruments(db, configuration=None, limit=200)
        assert "watchlisted" not in reason

    def _other_scannables(self, count: int) -> list[Instrument]:
        """Never-scanned instruments that would otherwise win the sweep's slots."""
        return [
            Instrument(
                id=uuid.uuid4(),
                exchange_ticker=f"OTHER{i}",
                name=f"OTHER{i} Co.",
                kind=InstrumentKind.STOCK,
                currency="USD",
                price_unit=PriceUnit.USD,
                is_scanner_eligible=True,
            )
            for i in range(count)
        ]


class TestSectorContext:
    """Sector-ETF proxy feeds the sector score + relative-strength signal."""

    async def _sector_etf(self, db: AsyncSession, symbol: str) -> Instrument:
        """A sector-ETF instrument (mock-candled) the scorer can load by symbol."""
        exchange = Exchange(mic="ARCX", name="NYSE Arca", country="US", timezone="America/New_York")
        db.add(exchange)
        await db.flush()
        etf = Instrument(
            id=uuid.uuid4(),
            exchange_id=exchange.id,
            exchange_ticker=symbol,
            name=f"{symbol} Sector ETF",
            kind=InstrumentKind.ETF,
            currency="USD",
            price_unit=PriceUnit.USD,
            is_scanner_eligible=True,
        )
        db.add(etf)
        await db.flush()
        db.add(
            MarketDataMapping(
                instrument_id=etf.id,
                provider=ProviderKind.MOCK,
                provider_symbol=symbol,  # loaded by _load_benchmark(symbol)
                is_signal_source=True,
                confirmed_by_user=True,
            )
        )
        await db.flush()
        await IngestionService(db).ingest_daily(etf, MockMarketDataProvider(), backfill_days=400)
        await db.commit()
        return etf

    async def test_tagged_instrument_gets_a_real_sector_score(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        await self._sector_etf(db, "XLK")
        scannable_instrument.sector = "Technology"  # → XLK proxy
        await db.commit()

        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()

        from app.models.scanner import ScannerResult

        result = (
            await db.execute(
                select(ScannerResult).where(ScannerResult.instrument_id == scannable_instrument.id)
            )
        ).scalar_one()
        # The sector category was computed from the proxy (not the neutral midpoint),
        # and the relative-strength-vs-sector signal was recorded.
        assert result.sector_score is not None
        assert "relative_momentum_vs_sector_12m" in (result.metrics or {})

    async def test_untagged_instrument_sector_is_neutral(
        self, db: AsyncSession, scannable_instrument: Instrument
    ) -> None:
        # No sector tag and no proxy → the sector category is the neutral midpoint
        # (half of its 20-point weight), never a penalty.
        await ScannerEngine(db).run([scannable_instrument])
        await db.commit()

        from app.models.scanner import ScannerResult

        result = (
            await db.execute(
                select(ScannerResult).where(ScannerResult.instrument_id == scannable_instrument.id)
            )
        ).scalar_one()
        assert float(result.sector_score) == 10.0


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
