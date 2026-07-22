"""Intraday ingestion against real PostgreSQL.

The 15-minute strategy reads M15 candles from the store; this proves the store
gets filled through the same validated path as daily ingestion, at an intraday
interval, from the deterministic mock provider (offline).
"""

from __future__ import annotations

import pytest

from app.data.mock_provider import MockMarketDataProvider
from app.data.store import CandleStore
from app.models.enums import Interval
from app.models.instrument import Instrument
from app.services.ingestion import IngestionService

pytestmark = pytest.mark.asyncio


class TestIntradayIngestion:
    async def test_m15_candles_are_written(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        result = await IngestionService(db).ingest_intraday(  # type: ignore[arg-type]
            candled_instrument, MockMarketDataProvider(), interval=Interval.M15, backfill_days=5
        )
        await db.commit()  # type: ignore[attr-defined]

        assert result.succeeded
        assert result.candles_written > 0
        stored = await CandleStore(db).get_candles(  # type: ignore[arg-type]
            candled_instrument.id, Interval.M15, limit=500
        )
        assert len(stored) > 0
        assert all(c.interval is Interval.M15 for c in stored)

    async def test_daily_interval_is_refused(
        self, db: object, candled_instrument: Instrument
    ) -> None:
        result = await IngestionService(db).ingest_intraday(  # type: ignore[arg-type]
            candled_instrument, MockMarketDataProvider(), interval=Interval.D1
        )
        assert not result.succeeded
        assert "not an intraday interval" in (result.skipped_reason or "")
