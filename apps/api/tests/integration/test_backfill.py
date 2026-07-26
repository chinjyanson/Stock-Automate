"""Catalogue backfill: selection scoping, map+ingest, and idempotent resume.

Uses the deterministic mock provider (extended with a batch method) so no network
is touched. The focus is the pipeline the scanner depends on: tradable, unmapped
instruments become mapped-and-candled, and re-running does not re-work them.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.mock_provider import MockMarketDataProvider
from app.data.types import Candle as ProviderCandle
from app.models.enums import (
    BrokerKind,
    InstrumentKind,
    LifecycleState,
    PriceUnit,
    ProviderKind,
)
from app.models.instrument import BrokerInstrument, Exchange, Instrument, MarketDataMapping
from app.models.market_data import Candle
from app.services.backfill import BackfillService


class BatchMockProvider(MockMarketDataProvider):
    """Mock provider that also answers the batched download the backfill uses."""

    async def get_batch_daily_candles(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        unit_by_symbol: dict[str, PriceUnit] | None = None,
    ) -> dict[str, list[ProviderCandle]]:
        return {s: await self.get_daily_candles(s, start, end) for s in symbols}


@pytest.fixture
async def tradable_instruments(db: AsyncSession) -> list[Instrument]:
    """Three Trading 212-tradable US instruments, unmapped and candle-less."""
    exchange = Exchange(mic="XNAS", name="Nasdaq", country="US", timezone="America/New_York")
    db.add(exchange)
    await db.flush()

    instruments: list[Instrument] = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        inst = Instrument(
            id=uuid.uuid4(),
            isin=f"US000000{i:04d}",
            exchange_id=exchange.id,
            exchange_ticker=ticker,
            name=f"{ticker} Corp.",
            kind=InstrumentKind.STOCK,
            currency="USD",
            price_unit=PriceUnit.USD,
            lifecycle_state=LifecycleState.DISCOVERED,
            is_scanner_eligible=True,
        )
        db.add(inst)
        await db.flush()
        db.add(
            BrokerInstrument(
                instrument_id=inst.id,
                broker=BrokerKind.TRADING212_DEMO,
                broker_ticker=f"{ticker}_US_EQ",
                is_currently_available=True,
            )
        )
        instruments.append(inst)
    await db.commit()
    return instruments


class TestSelection:
    async def test_selects_tradable_unmapped_instruments(
        self, db: AsyncSession, tradable_instruments: list[Instrument]
    ) -> None:
        chosen = await BackfillService(db).select_backfill_candidates(limit=10)
        assert {c.id for c in chosen} == {i.id for i in tradable_instruments}

    async def test_excludes_non_trading212_instruments(
        self, db: AsyncSession, tradable_instruments: list[Instrument]
    ) -> None:
        # An instrument on a supported venue but with no Trading 212 listing.
        exchange = (await db.execute(select(Exchange))).scalars().first()
        orphan = Instrument(
            id=uuid.uuid4(),
            isin="US9999999999",
            exchange_id=exchange.id,
            exchange_ticker="ZZZ",
            name="Not Tradable Co.",
            kind=InstrumentKind.STOCK,
            currency="USD",
            price_unit=PriceUnit.USD,
            is_scanner_eligible=True,
        )
        db.add(orphan)
        await db.commit()

        chosen = await BackfillService(db).select_backfill_candidates(limit=10)
        assert orphan.id not in {c.id for c in chosen}


class TestBackfillRun:
    async def test_maps_and_candles_the_universe(
        self, db: AsyncSession, tradable_instruments: list[Instrument]
    ) -> None:
        service = BackfillService(db)
        candidates = await service.select_backfill_candidates(limit=10)

        result = await service.backfill(candidates, BatchMockProvider())
        await db.commit()

        assert result.mapped == 3
        assert result.ingested == 3
        assert result.candles_written > 0

        # Every instrument now has a yfinance signal-source mapping and candles.
        mappings = (await db.execute(select(MarketDataMapping))).scalars().all()
        assert len(mappings) == 3
        assert all(m.provider is ProviderKind.YFINANCE and m.is_signal_source for m in mappings)

        counts = await service.funnel_counts()
        assert counts.tradable == 3
        assert counts.mapped == 3
        assert counts.candled == 3
        assert counts.scannable == 3  # 400 days of mock history clears the score bar

    async def test_rerun_is_idempotent(
        self, db: AsyncSession, tradable_instruments: list[Instrument]
    ) -> None:
        service = BackfillService(db)
        first = await service.select_backfill_candidates(limit=10)
        await service.backfill(first, BatchMockProvider())
        await db.commit()

        candle_count = len((await db.execute(select(Candle))).scalars().all())

        # Nothing is left to attempt — all are now mapped.
        again = await service.select_backfill_candidates(limit=10)
        assert again == []

        # And the candle store did not grow on a second pass.
        assert len((await db.execute(select(Candle))).scalars().all()) == candle_count
