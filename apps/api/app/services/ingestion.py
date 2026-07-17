"""Market-data ingestion (§4).

Implements the two access patterns the free-tier strategy depends on:

  * **Initial backfill** — one deep history fetch per instrument, once.
  * **Incremental refresh** — thereafter, only a small overlapping tail is
    re-requested and upserted.

The overlap is not redundancy. Providers revise recent bars (late prints,
consolidated corrections, adjustment for an action), so re-reading the last
several sessions is what keeps the store converging on the truth instead of
preserving whatever we happened to read first.

Never re-downloading full history is what keeps a rotating scan of a large
catalogue inside a free tier at all.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.base import MarketDataProvider
from app.data.store import CandleStore
from app.data.types import (
    Candle as CandleDTO,
)
from app.data.types import (
    ProviderError,
    ProviderQuotaExceededError,
)
from app.models.enums import (
    DataQualityEventKind,
    DataSeriesType,
    Interval,
    ProviderKind,
    QualityStatus,
)
from app.models.instrument import Instrument, MarketDataMapping

log = structlog.get_logger(__name__)

#: How much history a fresh instrument gets. Comfortably over the 252 trading
#: days the scanner prefers, with room for indicator warm-up.
DEFAULT_BACKFILL_DAYS = 730

#: Sessions re-requested on each incremental refresh (§4: "five to ten").
DEFAULT_OVERLAP_BARS = 10


@dataclass
class IngestionResult:
    instrument_id: uuid.UUID
    interval: Interval
    provider: ProviderKind | None = None
    candles_written: int = 0
    was_backfill: bool = False
    window_start: datetime | None = None
    window_end: datetime | None = None
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.skipped_reason is None and not self.errors


class IngestionService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = CandleStore(session)

    async def _signal_mapping(
        self, instrument_id: uuid.UUID, provider: ProviderKind
    ) -> MarketDataMapping | None:
        result = await self._session.execute(
            select(MarketDataMapping)
            .where(
                MarketDataMapping.instrument_id == instrument_id,
                MarketDataMapping.provider == provider,
                MarketDataMapping.is_active.is_(True),
            )
            .order_by(MarketDataMapping.priority.asc())
        )
        return result.scalars().first()

    async def ingest_daily(
        self,
        instrument: Instrument,
        provider: MarketDataProvider,
        *,
        backfill_days: int = DEFAULT_BACKFILL_DAYS,
        overlap_bars: int = DEFAULT_OVERLAP_BARS,
        force_full_backfill: bool = False,
    ) -> IngestionResult:
        """Bring one instrument's daily series up to date."""
        result = IngestionResult(
            instrument_id=instrument.id, interval=Interval.D1, provider=provider.kind
        )

        mapping = await self._signal_mapping(instrument.id, provider.kind)
        if mapping is None:
            # Unmapped is not an error — it is a state the UI must surface (§7).
            result.skipped_reason = (
                f"No active {provider.kind} mapping; resolve the symbol before ingesting"
            )
            return result

        if force_full_backfill:
            start, end = datetime.now(UTC) - timedelta(days=backfill_days), datetime.now(UTC)
            result.was_backfill = True
        else:
            start, end = await self._store.incremental_refresh_window(
                instrument.id,
                Interval.D1,
                overlap_bars=overlap_bars,
                full_backfill_days=backfill_days,
            )
            result.was_backfill = (await self._store.count_candles(instrument.id, Interval.D1)) == 0

        result.window_start, result.window_end = start, end

        try:
            candles = await provider.get_daily_candles(mapping.provider_symbol, start, end)
        except ProviderQuotaExceededError as exc:
            # Budget exhaustion is an expected operating condition, not a fault.
            result.skipped_reason = f"Provider budget exhausted: {exc}"
            log.warning(
                "ingestion.quota_exhausted",
                instrument_id=str(instrument.id),
                provider=str(provider.kind),
            )
            return result
        except ProviderError as exc:
            result.errors.append(str(exc))
            mapping.last_error = str(exc)
            await self._store.record_quality_event(
                instrument_id=instrument.id,
                kind=DataQualityEventKind.BACKFILL_GAP,
                severity="error",
                interval=Interval.D1,
                detail=f"{provider.kind} failed for {mapping.provider_symbol}: {exc}",
            )
            return result

        if not candles:
            result.skipped_reason = "Provider returned no candles for the requested window"
            return result

        validated = await self._validate(instrument, candles, result)
        if not validated:
            return result

        result.candles_written = await self._store.upsert_candles(
            instrument.id, validated, series_type=DataSeriesType.RAW
        )
        mapping.last_verified_at = datetime.now(UTC)
        mapping.last_error = None

        log.info(
            "ingestion.completed",
            instrument_id=str(instrument.id),
            symbol=mapping.provider_symbol,
            written=result.candles_written,
            backfill=result.was_backfill,
        )
        return result

    async def _validate(
        self, instrument: Instrument, candles: list[CandleDTO], result: IngestionResult
    ) -> list[CandleDTO]:
        """Reject candles we would not be willing to trade on.

        Runs before persistence: bad data that never lands cannot later be read
        by a strategy that forgot to check quality.
        """
        kept: list[CandleDTO] = []

        for candle in candles:
            if not candle.is_coherent:
                await self._store.record_quality_event(
                    instrument_id=instrument.id,
                    kind=DataQualityEventKind.SUSPICIOUS_MOVE,
                    severity="warning",
                    interval=candle.interval,
                    detail=(
                        f"Incoherent OHLC at {candle.timestamp.isoformat()} "
                        f"(o={candle.open} h={candle.high} l={candle.low} c={candle.close})"
                    ),
                )
                continue

            # A currency disagreement means the mapping points at a different
            # listing than we think. That is an identity failure (§5), and
            # storing the bars would corrupt the series silently.
            if candle.currency != instrument.currency:
                await self._store.record_quality_event(
                    instrument_id=instrument.id,
                    kind=DataQualityEventKind.UNIT_MISMATCH,
                    severity="error",
                    interval=candle.interval,
                    detail=(
                        f"Provider returned {candle.currency} but instrument is denominated "
                        f"in {instrument.currency}. The mapping may point at a different "
                        f"listing; refusing to store."
                    ),
                    context={
                        "provider_symbol": candle.symbol,
                        "provider_currency": candle.currency,
                        "instrument_currency": instrument.currency,
                    },
                )
                result.errors.append(
                    f"Currency mismatch: provider says {candle.currency}, "
                    f"instrument is {instrument.currency}"
                )
                return []

            kept.append(candle)

        if len(kept) < len(candles):
            log.warning(
                "ingestion.candles_rejected",
                instrument_id=str(instrument.id),
                rejected=len(candles) - len(kept),
                kept=len(kept),
            )
        return kept

    async def ingest_many(
        self,
        instruments: list[Instrument],
        provider: MarketDataProvider,
        *,
        backfill_days: int = DEFAULT_BACKFILL_DAYS,
    ) -> list[IngestionResult]:
        """Ingest a batch, sequentially.

        Sequential on purpose: the provider adapter already bounds its own
        concurrency, and stacking another layer here would defeat the rate
        limiting that keeps us inside the free tier.
        """
        results: list[IngestionResult] = []
        for instrument in instruments:
            try:
                results.append(
                    await self.ingest_daily(instrument, provider, backfill_days=backfill_days)
                )
            except ProviderQuotaExceededError:
                # The budget is gone; continuing the loop would just repeat this
                # for every remaining instrument.
                log.warning("ingestion.batch_halted_on_quota", remaining=len(instruments))
                break
            except Exception as exc:
                log.exception("ingestion.instrument_failed", instrument_id=str(instrument.id))
                results.append(
                    IngestionResult(
                        instrument_id=instrument.id,
                        interval=Interval.D1,
                        provider=provider.kind,
                        errors=[str(exc)],
                    )
                )
        return results

    async def has_sufficient_history(self, instrument_id: uuid.UUID, *, minimum_days: int) -> bool:
        """Enough closed daily bars to compute the long indicators?"""
        return (await self._store.count_candles(instrument_id, Interval.D1)) >= minimum_days

    async def assess_freshness(
        self, instrument_id: uuid.UUID, interval: Interval, *, max_age: timedelta
    ) -> QualityStatus:
        """Freshness of a series, for the dashboard and the pre-trade gate."""
        latest = await self._store.latest_candle(instrument_id, interval)
        if latest is None:
            return QualityStatus.UNAVAILABLE
        if (datetime.now(UTC) - latest.timestamp) > max_age:
            return QualityStatus.STALE
        return latest.quality_status
