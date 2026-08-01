"""The batched daily refresh, against real PostgreSQL.

The nightly sweep used to fetch one instrument per provider request, which is
why it was capped at ~100 names a night and took roughly four months to work
through a 12,800-instrument catalogue. Since the scanner ranks on *stored*
candles, that lap time was the age of the prices it ranked on — measured at 11
days average, 28 days worst.

These tests pin the two properties that make the batched path worth having:
it issues one request per chunk rather than per instrument, and it groups
instruments by how much history each needs so one never-ingested name cannot
drag its whole chunk into a two-year fetch.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.base import MarketDataProvider
from app.data.store import CandleStore
from app.data.types import Candle as CandleDTO
from app.data.types import Fundamentals, ProviderMapping, ProviderQuotaExceededError, Quote
from app.models.enums import (
    InstrumentKind,
    Interval,
    LifecycleState,
    PriceUnit,
    ProviderKind,
)
from app.models.instrument import Exchange, Instrument, MarketDataMapping
from app.services.ingestion import EMPTY_FETCHES_BEFORE_BACKOFF, IngestionService

pytestmark = pytest.mark.asyncio


class RecordingProvider(MarketDataProvider):
    """A provider that records how it was called and answers from memory."""

    kind = ProviderKind.MOCK

    def __init__(self, *, bars: int = 400, quota_after: int | None = None) -> None:
        self._bars = bars
        self._quota_after = quota_after
        #: One entry per batched call: (symbols, window_days).
        self.batches: list[tuple[list[str], int]] = []
        self.single_calls: list[str] = []

    async def get_batch_daily_candles(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        unit_by_symbol: dict[str, PriceUnit] | None = None,
    ) -> dict[str, list[CandleDTO]]:
        if self._quota_after is not None and len(self.batches) >= self._quota_after:
            raise ProviderQuotaExceededError("budget exhausted")
        self.batches.append((sorted(symbols), (end - start).days))
        return {symbol: self._series(symbol, start, end) for symbol in symbols}

    async def get_daily_candles(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CandleDTO]:
        self.single_calls.append(symbol)
        return self._series(symbol, start, end)

    def _series(self, symbol: str, start: datetime, end: datetime) -> list[CandleDTO]:
        # One bar per day across the requested window, capped so a two-year
        # window does not build an enormous fixture.
        days = min((end - start).days, self._bars)
        return [
            CandleDTO(
                symbol=symbol,
                interval=Interval.D1,
                timestamp=(end - timedelta(days=days - 1 - i)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                currency="USD",
                price_unit=PriceUnit.USD,
                provider=self.kind,
                is_closed=True,
            )
            for i in range(days)
        ]

    # -- Unused interface surface ------------------------------------------
    async def resolve_instrument(self, instrument: Instrument) -> ProviderMapping | None:
        return None

    async def get_intraday_candles(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> list[CandleDTO]:
        return []

    async def get_quote(self, symbol: str) -> Quote | None:
        return None

    async def get_basic_fundamentals(self, symbol: str) -> Fundamentals | None:
        return None


class SequentialOnlyProvider(RecordingProvider):
    """A provider with no batched fetch, exercising the base-class fallback."""

    # Deliberately not overriding `get_batch_daily_candles`, so the ABC's
    # looping default is what runs.
    get_batch_daily_candles = MarketDataProvider.get_batch_daily_candles  # type: ignore[assignment]


async def _exchange(db: AsyncSession) -> Exchange:
    existing = (
        await db.execute(select(Exchange).where(Exchange.mic == "XNAS"))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
    db.add(exchange)
    await db.flush()
    return exchange


async def _mapped(
    db: AsyncSession, ticker: str, *, currency: str = "USD", mapped: bool = True
) -> Instrument:
    exchange = await _exchange(db)
    instrument = Instrument(
        id=uuid.uuid4(),
        exchange_id=exchange.id,
        exchange_ticker=ticker,
        name=f"{ticker} Inc.",
        kind=InstrumentKind.STOCK,
        currency=currency,
        price_unit=PriceUnit.USD,
        lifecycle_state=LifecycleState.DISCOVERED,
        is_scanner_eligible=True,
    )
    db.add(instrument)
    await db.flush()
    if mapped:
        db.add(
            MarketDataMapping(
                instrument_id=instrument.id,
                provider=ProviderKind.MOCK,
                provider_symbol=ticker,
                is_signal_source=True,
                confirmed_by_user=True,
            )
        )
        await db.flush()
    return instrument


async def _seed_recent_bars(db: AsyncSession, instrument: Instrument, days: int) -> None:
    """Give an instrument bars up to yesterday, so it needs only a tail refresh."""
    now = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    await CandleStore(db).upsert_candles(
        instrument.id,
        [
            CandleDTO(
                symbol=instrument.exchange_ticker or "X",
                interval=Interval.D1,
                timestamp=now - timedelta(days=days - i),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("1000"),
                currency="USD",
                price_unit=PriceUnit.USD,
                provider=ProviderKind.MOCK,
                is_closed=True,
            )
            for i in range(days)
        ],
    )
    await db.flush()


class TestBatching:
    async def test_a_chunk_is_one_request_not_one_per_instrument(self, db: AsyncSession) -> None:
        """The entire point: request count scales with chunks, not instruments."""
        instruments = [await _mapped(db, f"SYM{i}") for i in range(12)]
        for instrument in instruments:
            await _seed_recent_bars(db, instrument, 30)
        await db.commit()

        provider = RecordingProvider()
        results = await IngestionService(db).refresh_many_batched(
            instruments, provider, chunk_size=50
        )
        await db.commit()

        assert len(provider.batches) == 1, "12 instruments in one chunk must be one request"
        assert provider.single_calls == [], "the per-instrument path must not be used"
        assert len(results) == 12
        assert all(r.candles_written > 0 for r in results)

    async def test_chunk_size_bounds_the_request(self, db: AsyncSession) -> None:
        instruments = [await _mapped(db, f"CH{i}") for i in range(10)]
        for instrument in instruments:
            await _seed_recent_bars(db, instrument, 30)
        await db.commit()

        provider = RecordingProvider()
        await IngestionService(db).refresh_many_batched(instruments, provider, chunk_size=4)
        await db.commit()

        assert [len(symbols) for symbols, _ in provider.batches] == [4, 4, 2]


class TestWindowGrouping:
    async def test_one_stale_instrument_does_not_widen_a_fresh_chunk(
        self, db: AsyncSession
    ) -> None:
        """The measured regression this grouping exists to prevent.

        Ungrouped, a single never-ingested name pulled every symbol sharing its
        chunk into a two-year fetch — 38,000 candles to refresh 100 instruments
        that between them needed a few hundred, repeating nightly because a
        delisted ticker never acquires candles.
        """
        fresh = [await _mapped(db, f"FRESH{i}") for i in range(4)]
        for instrument in fresh:
            await _seed_recent_bars(db, instrument, 30)
        # No bars at all: needs the full backfill window.
        never = await _mapped(db, "NEVER")
        await db.commit()

        # Deep name deliberately *first*. Without grouping it shares a chunk
        # with three fresh ones and widens their window to two years — put it
        # last and it would land in its own chunk regardless, and this test
        # would pass whether or not the grouping exists.
        provider = RecordingProvider()
        await IngestionService(db).refresh_many_batched(
            [never, *fresh], provider, chunk_size=4, backfill_days=730
        )
        await db.commit()

        batches = {tuple(symbols): days for symbols, days in provider.batches}
        deep = [(symbols, days) for symbols, days in provider.batches if days > 60]

        assert deep, f"the never-ingested name must still get its history, got {batches}"
        # The assertion that matters: nothing *except* the deep name may appear
        # in a deep fetch. Merely finding some shallow chunk elsewhere would
        # pass even when three of the four fresh names were dragged along.
        for symbols, days in deep:
            assert symbols == ["NEVER"], (
                f"a {days}-day fetch pulled in {symbols} — fresh instruments were "
                f"dragged into a deep chunk"
            )
        assert deep[0][1] > 700, "the never-ingested name needs its full backfill window"

    async def test_the_cheap_chunk_runs_first(self, db: AsyncSession) -> None:
        """Priority survives a run cut short: tail refreshes complete first."""
        never = await _mapped(db, "DEEP")
        fresh = [await _mapped(db, f"TAIL{i}") for i in range(4)]
        for instrument in fresh:
            await _seed_recent_bars(db, instrument, 30)
        await db.commit()

        provider = RecordingProvider()
        await IngestionService(db).refresh_many_batched(
            [never, *fresh], provider, chunk_size=4, backfill_days=730
        )
        await db.commit()

        assert "DEEP" not in provider.batches[0][0], "the deep name must not be in the first chunk"


class TestContract:
    async def test_an_unmapped_instrument_is_skipped_not_failed(self, db: AsyncSession) -> None:
        mapped = await _mapped(db, "OK")
        await _seed_recent_bars(db, mapped, 30)
        unmapped = await _mapped(db, "NOMAP", mapped=False)
        await db.commit()

        results = await IngestionService(db).refresh_many_batched(
            [mapped, unmapped], RecordingProvider()
        )
        await db.commit()

        by_id = {r.instrument_id: r for r in results}
        assert by_id[unmapped.id].skipped_reason is not None
        assert by_id[unmapped.id].errors == []
        assert by_id[mapped.id].candles_written > 0

    async def test_validation_still_rejects_a_currency_mismatch(self, db: AsyncSession) -> None:
        """Batching must not be a way around the quality gate.

        A provider returning a different currency means the mapping points at a
        different listing — storing those bars corrupts the series silently.
        """
        instrument = await _mapped(db, "GBPSTOCK", currency="GBP")
        await db.commit()

        results = await IngestionService(db).refresh_many_batched([instrument], RecordingProvider())
        await db.commit()

        assert results[0].candles_written == 0
        assert any("Currency mismatch" in e for e in results[0].errors)
        assert (await CandleStore(db).count_candles(instrument.id, Interval.D1)) == 0, (
            "mismatched bars must not be stored"
        )

    async def test_quota_exhaustion_leaves_the_rest_unattempted(self, db: AsyncSession) -> None:
        """Omitted, not marked failed — the caller stamps its cursor from results.

        An instrument the provider never reached must keep its place at the
        front of the queue, or a run that died early would skip it for a whole
        cycle on the strength of a call that never happened.
        """
        instruments = [await _mapped(db, f"Q{i}") for i in range(9)]
        for instrument in instruments:
            await _seed_recent_bars(db, instrument, 30)
        await db.commit()

        # Allows the first chunk, refuses the second.
        provider = RecordingProvider(quota_after=1)
        results = await IngestionService(db).refresh_many_batched(
            instruments, provider, chunk_size=3
        )
        await db.commit()

        assert len(results) == 3, "only the completed chunk is reported"
        assert len(provider.batches) == 1

    async def test_a_provider_without_batching_still_works(self, db: AsyncSession) -> None:
        """The ABC's fallback loops, so no caller has to ask what it is holding."""
        instruments = [await _mapped(db, f"SEQ{i}") for i in range(3)]
        for instrument in instruments:
            await _seed_recent_bars(db, instrument, 30)
        await db.commit()

        provider = SequentialOnlyProvider()
        results = await IngestionService(db).refresh_many_batched(instruments, provider)
        await db.commit()

        assert provider.batches == [], "this provider has no batched fetch"
        assert sorted(provider.single_calls) == ["SEQ0", "SEQ1", "SEQ2"]
        assert all(r.candles_written > 0 for r in results)


class EmptyProvider(RecordingProvider):
    """A provider that answers every symbol with no data.

    Distinct from a failing one: this is what a delisted ticker looks like. The
    request succeeded; the security is simply not there.
    """

    async def get_batch_daily_candles(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        unit_by_symbol: dict[str, PriceUnit] | None = None,
    ) -> dict[str, list[CandleDTO]]:
        self.batches.append((sorted(symbols), (end - start).days))
        return {symbol: [] for symbol in symbols}


class TestEmptyFetchBackoff:
    """Dead tickers must stop consuming a slot in every rotation.

    Roughly a fifth of this catalogue's mapped instruments return nothing and
    always will. Asked nightly, they spend that share of the sweep's budget
    re-learning it — which, at a ~2.6-night lap, is most of a night per lap.
    """

    async def _mapping(self, db: AsyncSession, instrument: Instrument) -> MarketDataMapping:
        return (
            await db.execute(
                select(MarketDataMapping).where(MarketDataMapping.instrument_id == instrument.id)
            )
        ).scalar_one()

    async def test_repeated_empty_responses_eventually_rest_the_symbol(
        self, db: AsyncSession
    ) -> None:
        instrument = await _mapped(db, "DEAD")
        await db.commit()

        service = IngestionService(db)
        for attempt in range(1, EMPTY_FETCHES_BEFORE_BACKOFF + 1):
            await service.refresh_many_batched([instrument], EmptyProvider())
            await db.commit()
            mapping = await self._mapping(db, instrument)
            await db.refresh(mapping)
            assert mapping.consecutive_empty_fetches == attempt
            if attempt < EMPTY_FETCHES_BEFORE_BACKOFF:
                assert mapping.retry_after is None, "must not rest on the first hiccup"

        mapping = await self._mapping(db, instrument)
        assert mapping.retry_after is not None
        assert mapping.retry_after > datetime.now(UTC) + timedelta(days=20)

    async def test_one_good_fetch_revives_a_rested_symbol(self, db: AsyncSession) -> None:
        """A relisting must not be written off permanently.

        This is why the backoff is a timestamp rather than an `is_dead` flag:
        the symbol is still asked, just rarely, and one candle clears the state.
        """
        instrument = await _mapped(db, "BACKALIVE")
        await db.commit()

        service = IngestionService(db)
        for _ in range(EMPTY_FETCHES_BEFORE_BACKOFF):
            await service.refresh_many_batched([instrument], EmptyProvider())
        await db.commit()
        mapping = await self._mapping(db, instrument)
        assert mapping.retry_after is not None

        await service.refresh_many_batched([instrument], RecordingProvider())
        await db.commit()
        await db.refresh(mapping)

        assert mapping.consecutive_empty_fetches == 0
        assert mapping.retry_after is None
        assert mapping.last_error is None

    async def test_an_error_is_not_counted_as_empty(self, db: AsyncSession) -> None:
        """A failed request says something about the network, not the security.

        Counting transport failures towards the backoff would rest live symbols
        during a provider outage — exactly when the data matters.
        """
        instrument = await _mapped(db, "FLAKY")
        await _seed_recent_bars(db, instrument, 30)
        await db.commit()

        class FailingProvider(RecordingProvider):
            async def get_batch_daily_candles(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                raise ProviderQuotaExceededError("budget exhausted")

        await IngestionService(db).refresh_many_batched([instrument], FailingProvider())
        await db.commit()

        mapping = await self._mapping(db, instrument)
        await db.refresh(mapping)
        assert mapping.consecutive_empty_fetches == 0
        assert mapping.retry_after is None
