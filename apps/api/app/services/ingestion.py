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
    PriceUnit,
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

#: Intraday history is short-lived and provider-budgeted, so a fresh intraday
#: series gets days, not the years a daily series gets.
DEFAULT_INTRADAY_BACKFILL_DAYS = 30

#: Symbols per batched request during a refresh. Matches the yfinance adapter's
#: own default; larger batches return frames big enough that parsing, not the
#: network, becomes the limit.
DEFAULT_REFRESH_CHUNK = 50

#: Consecutive empty fetches before a symbol stops being asked every rotation.
#: Three tolerates a provider hiccup and a genuinely quiet week without writing
#: off a live security.
EMPTY_FETCHES_BEFORE_BACKOFF = 3

#: How long a symbol rests after that. Long enough that ~2,600 dead tickers stop
#: dominating a sweep, short enough that a relisting is picked up within a month
#: rather than never.
EMPTY_FETCH_BACKOFF = timedelta(days=30)


def _record_empty_fetch(mapping: MarketDataMapping, now: datetime) -> None:
    """Count an empty result and, past the threshold, rest the symbol.

    Called for "the provider answered, with nothing" — not for errors. A failed
    request says something about the network; an empty one says something about
    the security.
    """
    mapping.consecutive_empty_fetches += 1
    if mapping.consecutive_empty_fetches >= EMPTY_FETCHES_BEFORE_BACKOFF:
        mapping.retry_after = now + EMPTY_FETCH_BACKOFF


def _record_successful_fetch(mapping: MarketDataMapping, now: datetime) -> None:
    """Clear the empty-fetch state. One good candle is enough to be alive again."""
    mapping.consecutive_empty_fetches = 0
    mapping.retry_after = None
    mapping.last_verified_at = now
    mapping.last_error = None


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
            _record_empty_fetch(mapping, datetime.now(UTC))
            return result

        validated = await self._validate(instrument, candles, result)
        if not validated:
            return result

        result.candles_written = await self._store.upsert_candles(
            instrument.id, validated, series_type=DataSeriesType.RAW
        )
        _record_successful_fetch(mapping, datetime.now(UTC))

        log.info(
            "ingestion.completed",
            instrument_id=str(instrument.id),
            symbol=mapping.provider_symbol,
            written=result.candles_written,
            backfill=result.was_backfill,
        )
        return result

    async def ingest_intraday(
        self,
        instrument: Instrument,
        provider: MarketDataProvider,
        *,
        interval: Interval = Interval.M15,
        backfill_days: int = DEFAULT_INTRADAY_BACKFILL_DAYS,
        overlap_bars: int = DEFAULT_OVERLAP_BARS,
        force_full_backfill: bool = False,
    ) -> IngestionResult:
        """Bring one instrument's intraday series up to date (§4, §8).

        Same shape as `ingest_daily` — the same mapping, validation and upsert
        path — but at an intraday interval and over a short window. The strategy
        that reads it (§8) reads the store, never the provider.
        """
        result = IngestionResult(
            instrument_id=instrument.id, interval=interval, provider=provider.kind
        )
        if not interval.is_intraday:
            result.skipped_reason = f"{interval} is not an intraday interval"
            return result
        if not provider.supports_intraday:
            result.skipped_reason = f"{provider.kind} does not provide intraday data"
            return result

        mapping = await self._signal_mapping(instrument.id, provider.kind)
        if mapping is None:
            result.skipped_reason = (
                f"No active {provider.kind} mapping; resolve the symbol before ingesting"
            )
            return result

        if force_full_backfill:
            start, end = datetime.now(UTC) - timedelta(days=backfill_days), datetime.now(UTC)
            result.was_backfill = True
        else:
            start, end = await self._store.incremental_refresh_window(
                instrument.id, interval, overlap_bars=overlap_bars, full_backfill_days=backfill_days
            )
            result.was_backfill = (await self._store.count_candles(instrument.id, interval)) == 0

        result.window_start, result.window_end = start, end

        try:
            candles = await provider.get_intraday_candles(
                mapping.provider_symbol, interval.value, start, end
            )
        except ProviderQuotaExceededError as exc:
            result.skipped_reason = f"Provider budget exhausted: {exc}"
            log.warning(
                "ingestion.quota_exhausted",
                instrument_id=str(instrument.id),
                provider=str(provider.kind),
                interval=interval.value,
            )
            return result
        except ProviderError as exc:
            result.errors.append(str(exc))
            mapping.last_error = str(exc)
            await self._store.record_quality_event(
                instrument_id=instrument.id,
                kind=DataQualityEventKind.BACKFILL_GAP,
                severity="error",
                interval=interval,
                detail=f"{provider.kind} intraday failed for {mapping.provider_symbol}: {exc}",
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
            "ingestion.intraday_completed",
            instrument_id=str(instrument.id),
            symbol=mapping.provider_symbol,
            interval=interval.value,
            written=result.candles_written,
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

    async def refresh_many_batched(
        self,
        instruments: list[Instrument],
        provider: MarketDataProvider,
        *,
        backfill_days: int = DEFAULT_BACKFILL_DAYS,
        overlap_bars: int = DEFAULT_OVERLAP_BARS,
        chunk_size: int = DEFAULT_REFRESH_CHUNK,
    ) -> list[IngestionResult]:
        """Refresh many instruments using the provider's batched fetch.

        Same contract as `ingest_many` — one `IngestionResult` per instrument,
        same validation, same quality events — but a different cost. That method
        issues one request per instrument, which is why the nightly refresh was
        capped at ~100 names and took months to work through the catalogue. Here
        a chunk of `chunk_size` symbols is one request, so the same request
        budget covers roughly `chunk_size` times as many instruments.

        **The window is shared across a chunk**, because a batched fetch takes
        one date range for every symbol in it. Taking the earliest start any
        member needs means nobody is under-fetched; the ones that needed less
        simply receive bars they already have, and the upsert is idempotent.

        **Which is why chunks are grouped by how much history each instrument
        needs, not by the caller's order.** A single never-ingested name in a
        chunk of fifty drags all fifty to a two-year window — measured, that was
        38,000 candles fetched and re-written to refresh 100 instruments that
        between them needed a few hundred. Worse, it would recur nightly:
        delisted tickers never acquire candles, so they would inflate whichever
        chunk they landed in forever. Grouping confines the deep fetch to the
        instruments that actually need it.

        Priority is preserved *within* each group, and the cheap group runs
        first. The nightly job puts Bot Universe names first precisely so a run
        cut short still refreshed the names a strategy reads tomorrow — and
        those, being refreshed nightly, are always in the cheap group.
        """
        if not instruments:
            return []

        mappings = await self._signal_mappings({i.id for i in instruments}, provider.kind)

        results: list[IngestionResult] = []
        pending: list[Instrument] = []
        for instrument in instruments:
            result = IngestionResult(
                instrument_id=instrument.id, interval=Interval.D1, provider=provider.kind
            )
            if instrument.id not in mappings:
                # Unmapped is a state to surface, not an error (§7).
                result.skipped_reason = (
                    f"No active {provider.kind} mapping; resolve the symbol before ingesting"
                )
                results.append(result)
                continue
            pending.append(instrument)

        now = datetime.now(UTC)
        floor = now - timedelta(days=backfill_days)
        # One query for every instrument's newest bar, rather than one each.
        newest = await self._store.latest_timestamps({i.id for i in pending}, Interval.D1)
        step = timedelta(days=1) * overlap_bars

        windows: dict[uuid.UUID, datetime] = {}
        for instrument in pending:
            latest = newest.get(instrument.id)
            windows[instrument.id] = floor if latest is None else max(latest - step, floor)

        # Group by window start so each chunk fetches only what its members
        # need. Sorted newest-first, with the caller's order broken only within
        # a group, so the cheap tail refreshes run before the deep ones.
        ordered = sorted(pending, key=lambda i: windows[i.id], reverse=True)

        for start_index in range(0, len(ordered), chunk_size):
            chunk = ordered[start_index : start_index + chunk_size]
            try:
                results.extend(
                    await self._refresh_chunk(chunk, mappings, provider, windows, floor, now)
                )
            except ProviderQuotaExceededError:
                # The budget is gone. Every remaining instrument is deliberately
                # left out of the results: the caller stamps its cursor from what
                # it got back, so omitting them keeps them at the front of the
                # queue instead of marking them attempted.
                log.warning(
                    "ingestion.batch_halted_on_quota",
                    remaining=len(ordered) - start_index,
                )
                break
        return results

    async def _refresh_chunk(
        self,
        chunk: list[Instrument],
        mappings: dict[uuid.UUID, MarketDataMapping],
        provider: MarketDataProvider,
        windows: dict[uuid.UUID, datetime],
        floor: datetime,
        now: datetime,
    ) -> list[IngestionResult]:
        """Fetch and persist one chunk. Raises on quota exhaustion."""
        # The earliest start any member needs. Bounded below by the caller's
        # floor so a never-ingested instrument cannot request a decade.
        window_start = max(min(windows[i.id] for i in chunk), floor)

        by_symbol: dict[str, list[Instrument]] = {}
        unit_by_symbol: dict[str, PriceUnit] = {}
        for instrument in chunk:
            symbol = mappings[instrument.id].provider_symbol
            # Two instruments can legitimately share a provider symbol (the same
            # listing mapped twice). Grouping means the symbol is requested once
            # and both receive the bars.
            by_symbol.setdefault(symbol, []).append(instrument)
            unit_by_symbol[symbol] = instrument.price_unit

        fetched = await provider.get_batch_daily_candles(
            list(by_symbol), window_start, now, unit_by_symbol=unit_by_symbol
        )

        results: list[IngestionResult] = []
        for symbol, members in by_symbol.items():
            candles = fetched.get(symbol, [])
            for instrument in members:
                mapping = mappings[instrument.id]
                result = IngestionResult(
                    instrument_id=instrument.id,
                    interval=Interval.D1,
                    provider=provider.kind,
                    window_start=window_start,
                    window_end=now,
                    was_backfill=windows[instrument.id] <= floor,
                )
                if not candles:
                    result.skipped_reason = "Provider returned no candles for the requested window"
                    # The provider answered; this symbol has nothing. Counted so
                    # a permanently dead ticker stops consuming a slot in every
                    # rotation — see `_record_empty_fetch`.
                    _record_empty_fetch(mapping, now)
                    results.append(result)
                    continue

                validated = await self._validate(instrument, candles, result)
                if not validated:
                    results.append(result)
                    continue

                result.candles_written = await self._store.upsert_candles(
                    instrument.id, validated, series_type=DataSeriesType.RAW
                )
                _record_successful_fetch(mapping, now)
                results.append(result)

        log.info(
            "ingestion.chunk_completed",
            symbols=len(by_symbol),
            instruments=len(chunk),
            written=sum(r.candles_written for r in results),
            window_days=(now - window_start).days,
        )
        return results

    async def _signal_mappings(
        self, instrument_ids: set[uuid.UUID], provider: ProviderKind
    ) -> dict[uuid.UUID, MarketDataMapping]:
        """Active mappings for many instruments, in one query.

        The per-instrument path issues one of these per name. At a hundred
        instruments that is invisible; at several thousand it is the second
        bottleneck after the fetch itself.
        """
        if not instrument_ids:
            return {}
        rows = (
            (
                await self._session.execute(
                    select(MarketDataMapping)
                    .where(
                        MarketDataMapping.instrument_id.in_(instrument_ids),
                        MarketDataMapping.provider == provider,
                        MarketDataMapping.is_active.is_(True),
                    )
                    .order_by(MarketDataMapping.priority.asc())
                )
            )
            .scalars()
            .all()
        )
        # Ascending priority with first-write-wins gives the same mapping the
        # single-instrument path picks with `.first()`.
        out: dict[uuid.UUID, MarketDataMapping] = {}
        for row in rows:
            out.setdefault(row.instrument_id, row)
        return out

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
