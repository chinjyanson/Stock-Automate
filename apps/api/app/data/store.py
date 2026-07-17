"""Local candle store (§4).

Once data is here, this is the source of truth. Strategies read this module,
never a provider, so a decision is always made against data we have persisted,
versioned and quality-checked.

Two invariants this module enforces on behalf of every caller:

  * **Upserts are idempotent.** Re-running a backfill, or overlapping the daily
    refresh window, must converge rather than duplicate. Jobs are retried; the
    store has to tolerate that (§16).
  * **Unclosed bars are never served for decisions.** `get_candles` filters
    them out by default. Reading an unclosed bar means reading a `close` that
    is still moving (§4).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import structlog
from sqlalchemy import and_, case, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.types import Candle as CandleDTO
from app.models.enums import (
    DataQualityEventKind,
    DataSeriesType,
    Interval,
    QualityStatus,
)
from app.models.market_data import Candle, DataQualityEvent

log = structlog.get_logger(__name__)

#: Bar durations, used for closed-ness and gap arithmetic.
INTERVAL_DURATION: dict[Interval, timedelta] = {
    Interval.M1: timedelta(minutes=1),
    Interval.M5: timedelta(minutes=5),
    Interval.M15: timedelta(minutes=15),
    Interval.H1: timedelta(hours=1),
    Interval.H4: timedelta(hours=4),
    Interval.D1: timedelta(days=1),
    Interval.W1: timedelta(weeks=1),
}


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """How much history we hold, and how fresh it is.

    A dataclass rather than a dict so callers get typed fields; a `dict[str,
    object]` forced an unchecked cast at every use site.
    """

    candle_count: int
    degraded_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    age: timedelta | None


class CandleStore:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_candles(
        self,
        instrument_id: uuid.UUID,
        candles: list[CandleDTO],
        *,
        series_type: DataSeriesType = DataSeriesType.RAW,
    ) -> int:
        """Insert or update candles. Returns the number written.

        Conflict target is the full uniqueness key from §4
        (instrument+interval+timestamp+series_type). On conflict we overwrite
        rather than ignore: a re-fetch is usually a *correction* — a late
        consolidated print, or a bar that has since closed — and keeping the
        stale first read would defeat the point of refetching.
        """
        if not candles:
            return 0

        now = datetime.now(UTC)
        rows = [
            {
                "id": uuid.uuid4(),
                "instrument_id": instrument_id,
                "provider": candle.provider,
                "provider_symbol": candle.symbol,
                "interval": candle.interval,
                "data_series_type": series_type,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "adjusted_close": candle.adjusted_close,
                "volume": candle.volume,
                "currency": candle.currency,
                "price_unit": candle.price_unit,
                "is_adjusted": candle.is_adjusted,
                "is_closed": candle.is_closed,
                "quality_status": QualityStatus.OK if candle.is_coherent else QualityStatus.SUSPECT,
                "retrieved_at": now,
            }
            for candle in candles
        ]

        stmt = pg_insert(Candle).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["instrument_id", "interval", "timestamp", "data_series_type"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adjusted_close": stmt.excluded.adjusted_close,
                "volume": stmt.excluded.volume,
                "is_closed": stmt.excluded.is_closed,
                "quality_status": stmt.excluded.quality_status,
                "provider": stmt.excluded.provider,
                "provider_symbol": stmt.excluded.provider_symbol,
                "retrieved_at": stmt.excluded.retrieved_at,
            },
        )
        await self._session.execute(stmt)
        await self._session.flush()

        log.info(
            "store.candles_upserted",
            instrument_id=str(instrument_id),
            count=len(rows),
            interval=str(candles[0].interval),
        )
        return len(rows)

    async def get_candles(
        self,
        instrument_id: uuid.UUID,
        interval: Interval,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        series_type: DataSeriesType = DataSeriesType.RAW,
        closed_only: bool = True,
        tradable_quality_only: bool = False,
    ) -> list[Candle]:
        """Read candles ascending by timestamp.

        `closed_only` defaults True — the safe default for anything that will
        inform a decision. Chart rendering is the legitimate exception.
        """
        conditions = [
            Candle.instrument_id == instrument_id,
            Candle.interval == interval,
            Candle.data_series_type == series_type,
        ]
        if start is not None:
            conditions.append(Candle.timestamp >= start)
        if end is not None:
            conditions.append(Candle.timestamp <= end)
        if closed_only:
            conditions.append(Candle.is_closed.is_(True))
        if tradable_quality_only:
            conditions.append(Candle.quality_status == QualityStatus.OK)

        stmt = select(Candle).where(and_(*conditions))

        if limit is not None:
            # For "the last N bars", take the newest N then restore ascending
            # order. Ordering ascending with a LIMIT would return the *oldest* N.
            stmt = stmt.order_by(Candle.timestamp.desc()).limit(limit)
            result = await self._session.execute(stmt)
            return list(reversed(result.scalars().all()))

        stmt = stmt.order_by(Candle.timestamp.asc())
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def latest_candle(
        self,
        instrument_id: uuid.UUID,
        interval: Interval,
        *,
        series_type: DataSeriesType = DataSeriesType.RAW,
        closed_only: bool = True,
    ) -> Candle | None:
        candles = await self.get_candles(
            instrument_id,
            interval,
            limit=1,
            series_type=series_type,
            closed_only=closed_only,
        )
        return candles[0] if candles else None

    async def latest_timestamp(
        self, instrument_id: uuid.UUID, interval: Interval
    ) -> datetime | None:
        """Newest stored bar time, used to compute an incremental refresh window."""
        result = await self._session.execute(
            select(func.max(Candle.timestamp)).where(
                Candle.instrument_id == instrument_id,
                Candle.interval == interval,
            )
        )
        return result.scalar_one_or_none()

    async def count_candles(self, instrument_id: uuid.UUID, interval: Interval) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(Candle)
            .where(Candle.instrument_id == instrument_id, Candle.interval == interval)
        )
        return int(result.scalar_one())

    async def incremental_refresh_window(
        self,
        instrument_id: uuid.UUID,
        interval: Interval,
        *,
        overlap_bars: int = 10,
        full_backfill_days: int = 730,
    ) -> tuple[datetime, datetime]:
        """The date range a refresh should actually request (§4).

        After the initial backfill we re-request only a small overlapping tail
        rather than the whole history. The overlap exists because providers
        revise recent bars (late prints, corrections); without it we would keep
        a first, wrong reading forever.
        """
        now = datetime.now(UTC)
        latest = await self.latest_timestamp(instrument_id, interval)

        if latest is None:
            return now - timedelta(days=full_backfill_days), now

        step = INTERVAL_DURATION[interval]
        return latest - (step * overlap_bars), now

    async def is_stale(
        self,
        instrument_id: uuid.UUID,
        interval: Interval,
        *,
        max_age: timedelta,
        now: datetime | None = None,
    ) -> bool:
        """Is the newest closed bar older than `max_age`?

        Absent data counts as stale. "We have never fetched this" and "we
        fetched it and it is old" are both reasons not to trade it (§17).
        """
        now = now or datetime.now(UTC)
        latest = await self.latest_candle(instrument_id, interval)
        if latest is None:
            return True
        return (now - latest.timestamp) > max_age

    async def detect_gaps(
        self,
        instrument_id: uuid.UUID,
        interval: Interval,
        expected_timestamps: list[datetime],
    ) -> list[datetime]:
        """Which expected bars are missing?

        The caller supplies `expected_timestamps` from the venue's trading
        schedule rather than us deriving them, because only the schedule knows
        about holidays and half-days. Inferring expectation from the data
        itself would make a gap invisible precisely when it matters.
        """
        if not expected_timestamps:
            return []

        result = await self._session.execute(
            select(Candle.timestamp).where(
                Candle.instrument_id == instrument_id,
                Candle.interval == interval,
                Candle.timestamp >= min(expected_timestamps),
                Candle.timestamp <= max(expected_timestamps),
            )
        )
        present = {row[0] for row in result.all()}
        return sorted(ts for ts in expected_timestamps if ts not in present)

    async def record_quality_event(
        self,
        *,
        instrument_id: uuid.UUID | None,
        kind: DataQualityEventKind,
        detail: str,
        severity: str = "warning",
        interval: Interval | None = None,
        context: dict[str, object] | None = None,
    ) -> DataQualityEvent:
        event = DataQualityEvent(
            instrument_id=instrument_id,
            kind=kind,
            severity=severity,
            detail=detail,
            interval=interval,
            context=context,
        )
        self._session.add(event)
        await self._session.flush()
        log.warning(
            "store.data_quality_event",
            kind=str(kind),
            severity=severity,
            instrument_id=str(instrument_id) if instrument_id else None,
            detail=detail,
        )
        return event

    async def mark_stale(
        self, instrument_id: uuid.UUID, interval: Interval, older_than: datetime
    ) -> int:
        """Flag old bars as stale. Returns rows touched."""
        result = await self._session.execute(
            select(Candle).where(
                Candle.instrument_id == instrument_id,
                Candle.interval == interval,
                Candle.timestamp < older_than,
                Candle.quality_status == QualityStatus.OK,
            )
        )
        candles = list(result.scalars().all())
        for candle in candles:
            candle.quality_status = QualityStatus.STALE
        await self._session.flush()
        return len(candles)

    async def purge_unclosed(self, instrument_id: uuid.UUID, interval: Interval) -> int:
        """Delete unclosed bars for this series.

        Used before a refresh: yesterday's partial bar should be replaced by
        today's closed one, and leaving the partial around risks it being read
        if a `closed_only` filter is ever missed.
        """
        result = await self._session.execute(
            delete(Candle).where(
                Candle.instrument_id == instrument_id,
                Candle.interval == interval,
                Candle.is_closed.is_(False),
            )
        )
        # `rowcount` is a CursorResult attribute; execute() is typed as Result.
        return int(cast("CursorResult[Any]", result).rowcount or 0)

    async def coverage_summary(
        self, instrument_id: uuid.UUID, interval: Interval
    ) -> CoverageSummary:
        """Coverage and freshness, for the scanner's data-quality display (§6)."""
        result = await self._session.execute(
            select(
                func.count(Candle.id),
                func.min(Candle.timestamp),
                func.max(Candle.timestamp),
                # COUNT over a filtered CASE: counts only non-OK rows, because
                # COUNT ignores NULLs.
                func.count(case((Candle.quality_status != QualityStatus.OK, 1), else_=None)),
            ).where(
                Candle.instrument_id == instrument_id,
                Candle.interval == interval,
                Candle.is_closed.is_(True),
            )
        )
        count, first, last, degraded = result.one()
        return CoverageSummary(
            candle_count=int(count or 0),
            degraded_count=int(degraded or 0),
            first_timestamp=first,
            last_timestamp=last,
            age=(datetime.now(UTC) - last) if last else None,
        )


def annualisation_factor(interval: Interval) -> Decimal:
    """Bars per year, for scaling volatility. Trading days, not calendar days."""
    factors = {
        Interval.D1: Decimal(252),
        Interval.W1: Decimal(52),
        Interval.H1: Decimal(252 * 7),
        Interval.M15: Decimal(252 * 26),
    }
    return factors.get(interval, Decimal(252))
