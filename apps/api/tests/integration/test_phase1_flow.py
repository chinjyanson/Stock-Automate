"""End-to-end Phase 1 flow against real PostgreSQL.

Walks the acceptance criteria this phase claims: connect a broker, synchronise
instruments, map one to a market-data symbol, backfill candles, refresh
incrementally, and see every step in an immutable audit log.

Uses the mock broker and mock provider throughout, which is the point of §23 —
this must all work on a clone with no credentials.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.auth.service import AuthService
from app.broker.mock import MockBroker
from app.data.mock_provider import MockMarketDataProvider
from app.data.store import CandleStore
from app.data.types import ProviderUnavailableError
from app.models.enums import (
    AuditEventKind,
    Interval,
    LifecycleState,
    PriceUnit,
)
from app.models.instrument import Exchange, Instrument
from app.services.ingestion import IngestionService
from app.services.instrument_sync import InstrumentSyncService
from app.services.mapping import MappingService


@pytest.fixture
async def exchanges(db: AsyncSession) -> dict[str, Exchange]:
    """The venues instrument sync attaches instruments to."""
    rows = {
        "XLON": Exchange(
            mic="XLON", name="London Stock Exchange", country="GB", timezone="Europe/London"
        ),
        "XNAS": Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York"),
    }
    for exchange in rows.values():
        db.add(exchange)
    await db.commit()
    return rows


class TestInstrumentSync:
    async def test_sync_creates_instruments_and_broker_instruments(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        result = await InstrumentSyncService(db).sync(MockBroker())
        await db.commit()

        assert result.total_from_broker == 7
        assert result.broker_instruments_created == 7
        assert result.instruments_created == 7
        assert not result.errors

    async def test_sync_is_idempotent(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        """Re-running must not duplicate. Jobs get retried (§16)."""
        service = InstrumentSyncService(db)
        await service.sync(MockBroker())
        await db.commit()

        second = await service.sync(MockBroker())
        await db.commit()

        assert second.broker_instruments_created == 0
        assert second.broker_instruments_updated == 7
        assert second.instruments_created == 0

        total = (await db.execute(select(Instrument))).scalars().all()
        assert len(total) == 7

    async def test_sync_does_not_grant_bot_universe_membership(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        """§7: appearing in the catalogue must not make an instrument tradable."""
        await InstrumentSyncService(db).sync(MockBroker())
        await db.commit()

        instruments = (await db.execute(select(Instrument))).scalars().all()
        assert all(not i.is_bot_universe for i in instruments)
        assert all(i.lifecycle_state is LifecycleState.DISCOVERED for i in instruments)

    async def test_gbx_instrument_settles_in_gbp(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        """A pence-quoted line is denominated in pounds.

        Storing currency="GBX" would make every cash figure 100x wrong.
        """
        await InstrumentSyncService(db).sync(MockBroker())
        await db.commit()

        sgln = (
            await db.execute(select(Instrument).where(Instrument.isin == "IE00B4ND3602"))
        ).scalar_one()

        assert sgln.price_unit is PriceUnit.GBX
        assert sgln.currency == "GBP"

    async def test_ticker_suffix_is_stripped_to_exchange_ticker(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        await InstrumentSyncService(db).sync(MockBroker())
        await db.commit()

        aapl = (
            await db.execute(select(Instrument).where(Instrument.isin == "US0378331005"))
        ).scalar_one()
        assert aapl.exchange_ticker == "AAPL"
        assert aapl.exchange is not None
        assert aapl.exchange.mic == "XNAS"

    async def test_delisted_instruments_are_marked_not_deleted(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        """History references them; marking is reversible, deleting is not."""
        service = InstrumentSyncService(db)
        await service.sync(MockBroker())
        await db.commit()

        class ShrunkBroker(MockBroker):
            async def sync_instruments(self):  # type: ignore[no-untyped-def]
                full = await super().sync_instruments()
                return [i for i in full if i.broker_ticker != "AAPL_US_EQ"]

        result = await service.sync(ShrunkBroker())
        await db.commit()

        assert result.delisted == 1
        # The canonical instrument still exists.
        assert (
            await db.execute(select(Instrument).where(Instrument.isin == "US0378331005"))
        ).scalar_one_or_none() is not None

    async def test_sync_writes_an_audit_event(
        self, db: AsyncSession, exchanges: dict[str, Exchange]
    ) -> None:
        await InstrumentSyncService(db).sync(MockBroker())
        await db.commit()

        events = await AuditService(db).recent(kind=AuditEventKind.INSTRUMENT_SYNCED)
        assert len(events) == 1
        assert "7 instruments" in events[0].summary


class TestMappingAndIngestion:
    @pytest.fixture
    async def synced(self, db: AsyncSession, exchanges: dict[str, Exchange]) -> Instrument:
        await InstrumentSyncService(db).sync(MockBroker())
        await db.commit()
        return (
            await db.execute(select(Instrument).where(Instrument.isin == "US78462F1030"))
        ).scalar_one()

    async def test_mapping_resolves_a_provider_symbol(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        result = await MappingService(db).resolve(
            synced, MockMarketDataProvider(), is_signal_source=True
        )
        await db.commit()

        assert result.resolved
        assert result.mapping is not None
        assert result.mapping.provider_symbol == "SPY"
        assert result.mapping.is_signal_source

    async def test_only_one_mapping_can_be_the_signal_source(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        """Two signal sources would make "the" signal ambiguous."""
        service = MappingService(db)
        await service.resolve(synced, MockMarketDataProvider(), is_signal_source=True)
        await db.commit()

        mappings = await service.list_for_instrument(synced.id)
        signal_sources = [m for m in mappings if m.is_signal_source]
        assert len(signal_sources) == 1

    async def test_ingestion_without_a_mapping_is_skipped_not_errored(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        """Unmapped is a state to surface, not a failure to alert on (§7)."""
        result = await IngestionService(db).ingest_daily(synced, MockMarketDataProvider())
        assert result.skipped_reason is not None
        assert "mapping" in result.skipped_reason.lower()
        assert result.candles_written == 0

    async def test_backfill_then_incremental_refresh(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        """Acceptance criterion 5: cached locally, updated incrementally."""
        await MappingService(db).resolve(synced, MockMarketDataProvider(), is_signal_source=True)
        await db.commit()

        service = IngestionService(db)
        provider = MockMarketDataProvider()

        first = await service.ingest_daily(synced, provider, backfill_days=400)
        await db.commit()

        assert first.was_backfill
        assert first.candles_written > 200

        # The second pass must request only a short tail, not 400 days again.
        second = await service.ingest_daily(synced, provider, backfill_days=400)
        await db.commit()

        assert not second.was_backfill
        assert second.window_start is not None
        window_days = (datetime.now(UTC) - second.window_start).days
        assert window_days < 30, "incremental refresh should request a small overlap"

    async def test_refetching_upserts_rather_than_duplicating(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        await MappingService(db).resolve(synced, MockMarketDataProvider(), is_signal_source=True)
        await db.commit()

        service = IngestionService(db)
        provider = MockMarketDataProvider()
        store = CandleStore(db)

        await service.ingest_daily(synced, provider, backfill_days=100)
        await db.commit()
        count_after_first = await store.count_candles(synced.id, Interval.D1)

        await service.ingest_daily(synced, provider, force_full_backfill=True, backfill_days=100)
        await db.commit()
        count_after_second = await store.count_candles(synced.id, Interval.D1)

        assert count_after_first == count_after_second

    async def test_stored_candles_are_readable_and_ordered(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        await MappingService(db).resolve(synced, MockMarketDataProvider(), is_signal_source=True)
        await db.commit()
        await IngestionService(db).ingest_daily(synced, MockMarketDataProvider(), backfill_days=100)
        await db.commit()

        candles = await CandleStore(db).get_candles(synced.id, Interval.D1, limit=10)
        assert len(candles) == 10
        timestamps = [c.timestamp for c in candles]
        assert timestamps == sorted(timestamps), "candles must come back ascending"
        assert all(isinstance(c.close, Decimal) for c in candles)

    async def test_limit_returns_the_newest_bars(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        """`limit` means "the last N", not "the first N" — an off-by-everything
        bug that would feed a strategy year-old prices."""
        await MappingService(db).resolve(synced, MockMarketDataProvider(), is_signal_source=True)
        await db.commit()
        await IngestionService(db).ingest_daily(synced, MockMarketDataProvider(), backfill_days=200)
        await db.commit()

        store = CandleStore(db)
        everything = await store.get_candles(synced.id, Interval.D1)
        last_five = await store.get_candles(synced.id, Interval.D1, limit=5)

        assert [c.timestamp for c in last_five] == [c.timestamp for c in everything[-5:]]

    async def test_provider_failure_writes_a_data_quality_event(
        self, db: AsyncSession, synced: Instrument
    ) -> None:
        """§17: an outage must fail closed and be visible, not pass silently."""
        await MappingService(db).resolve(synced, MockMarketDataProvider(), is_signal_source=True)
        await db.commit()

        failing = MockMarketDataProvider()
        failing.fail_with = ProviderUnavailableError("simulated provider outage")

        result = await IngestionService(db).ingest_daily(synced, failing)
        await db.commit()

        assert result.errors
        assert result.candles_written == 0

        from app.models.market_data import DataQualityEvent

        events = (await db.execute(select(DataQualityEvent))).scalars().all()
        assert len(events) == 1
        assert events[0].severity == "error"


class TestAuditChain:
    async def test_chain_is_intact_across_many_appends(self, db: AsyncSession) -> None:
        service = AuditService(db)
        for index in range(10):
            await service.record(kind=AuditEventKind.SETTING_CHANGED, summary=f"event {index}")
        await db.commit()

        is_intact, problems = await service.verify_chain()
        assert is_intact, problems

    async def test_each_event_commits_to_its_predecessor(self, db: AsyncSession) -> None:
        service = AuditService(db)
        first = await service.record(kind=AuditEventKind.SETTING_CHANGED, summary="first")
        second = await service.record(kind=AuditEventKind.SETTING_CHANGED, summary="second")
        await db.commit()

        assert second.previous_hash == first.event_hash
        assert first.sequence < second.sequence

    async def test_secrets_are_redacted_from_payloads(self, db: AsyncSession) -> None:
        """Defence in depth: an audit row is permanent, so a leaked credential
        in one cannot be deleted afterwards."""
        service = AuditService(db)
        event = await service.record(
            kind=AuditEventKind.CREDENTIAL_CHANGED,
            summary="credential rotated",
            payload={
                "broker": "trading212_live",
                "api_key": "super-secret-live-key",
                "nested": {"password": "hunter2", "safe": "kept"},
            },
        )
        await db.commit()

        assert event.payload is not None
        assert event.payload["api_key"] == "[redacted]"
        assert event.payload["nested"]["password"] == "[redacted]"
        # Non-secret context must survive, or the log stops being useful.
        assert event.payload["broker"] == "trading212_live"
        assert event.payload["nested"]["safe"] == "kept"

    async def test_database_rejects_an_update(self, db: AsyncSession) -> None:
        """The immutability guarantee, enforced below the application."""
        service = AuditService(db)
        event = await service.record(kind=AuditEventKind.SETTING_CHANGED, summary="original")
        await db.commit()

        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError, match="append-only"):
            await db.execute(
                text("UPDATE audit_events SET summary = 'tampered' WHERE id = :id"),
                {"id": event.id},
            )
        await db.rollback()

    async def test_database_rejects_a_delete(self, db: AsyncSession) -> None:
        service = AuditService(db)
        event = await service.record(kind=AuditEventKind.SETTING_CHANGED, summary="keep me")
        await db.commit()

        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError, match="append-only"):
            await db.execute(text("DELETE FROM audit_events WHERE id = :id"), {"id": event.id})
        await db.rollback()


class TestAuth:
    async def test_login_succeeds_and_opens_a_session(self, db: AsyncSession) -> None:
        service = AuthService(db)
        await service.create_user(email="a@example.com", password="a-long-enough-password")
        await db.commit()

        result = await service.authenticate(
            email="a@example.com", password="a-long-enough-password"
        )
        await db.commit()

        assert result.token
        resolved = await service.resolve_session(result.token)
        assert resolved is not None

    async def test_wrong_password_is_rejected(self, db: AsyncSession) -> None:
        from app.auth.service import AuthError

        service = AuthService(db)
        await service.create_user(email="b@example.com", password="a-long-enough-password")
        await db.commit()

        with pytest.raises(AuthError):
            await service.authenticate(email="b@example.com", password="wrong-password")
        await db.commit()

    async def test_unknown_account_and_wrong_password_are_indistinguishable(
        self, db: AsyncSession
    ) -> None:
        """No account-enumeration oracle."""
        from app.auth.service import AuthError

        service = AuthService(db)
        await service.create_user(email="c@example.com", password="a-long-enough-password")
        await db.commit()

        with pytest.raises(AuthError) as unknown:
            await service.authenticate(email="nobody@example.com", password="whatever")
        await db.commit()

        with pytest.raises(AuthError) as wrong:
            await service.authenticate(email="c@example.com", password="wrong-password")
        await db.commit()

        assert str(unknown.value) == str(wrong.value)

    async def test_revoked_session_stops_resolving_immediately(self, db: AsyncSession) -> None:
        """Why sessions are server-side: the kill switch needs this (§17)."""
        service = AuthService(db)
        await service.create_user(email="d@example.com", password="a-long-enough-password")
        await db.commit()

        result = await service.authenticate(
            email="d@example.com", password="a-long-enough-password"
        )
        await db.commit()

        await service.revoke_session(result.session)
        await db.commit()

        assert await service.resolve_session(result.token) is None

    async def test_session_token_is_not_stored_in_plaintext(self, db: AsyncSession) -> None:
        service = AuthService(db)
        await service.create_user(email="e@example.com", password="a-long-enough-password")
        await db.commit()

        result = await service.authenticate(
            email="e@example.com", password="a-long-enough-password"
        )
        await db.commit()

        assert result.session.token_hash != result.token
        assert len(result.session.token_hash) == 64

    async def test_reauth_window_expires(self, db: AsyncSession) -> None:
        service = AuthService(db)
        await service.create_user(email="f@example.com", password="a-long-enough-password")
        await db.commit()
        result = await service.authenticate(
            email="f@example.com", password="a-long-enough-password"
        )
        await db.commit()

        assert AuthService.is_recently_reauthenticated(result.session)
        # Six minutes later, a five-minute window has closed.
        later = datetime.now(UTC) + timedelta(minutes=6)
        assert not AuthService.is_recently_reauthenticated(result.session, now=later)
