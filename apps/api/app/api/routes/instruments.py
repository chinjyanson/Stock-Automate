"""Instrument endpoints (§19).

Covers acceptance criteria 2 to 5: sync the catalogue, search it, add an
instrument without touching source code, map it to market-data symbols, and
cache candles locally with incremental updates.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.schemas import (
    CandleResponse,
    ConfirmMappingRequest,
    IngestRequest,
    IngestResponse,
    InstrumentDetailResponse,
    InstrumentListResponse,
    InstrumentResponse,
    MapInstrumentRequest,
    MapInstrumentResponse,
    MarketDataMappingResponse,
    SyncInstrumentsResponse,
)
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.broker.factory import default_paper_broker_kind, resolve_broker
from app.config import get_settings
from app.data.factory import resolve_provider
from app.data.store import CandleStore
from app.db import get_db
from app.models.enums import (
    ActorKind,
    AuditEventKind,
    BrokerKind,
    Interval,
    LifecycleState,
    ProviderKind,
)
from app.models.instrument import Instrument, MarketDataMapping
from app.services.ingestion import IngestionService
from app.services.instrument_sync import InstrumentSyncService
from app.services.mapping import MappingService

router = APIRouter(prefix="/instruments", tags=["instruments"])
log = structlog.get_logger(__name__)


def _parse_provider(raw: str) -> ProviderKind:
    try:
        return ProviderKind(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider {raw!r}. Known: {[p.value for p in ProviderKind]}",
        ) from exc


async def _get_instrument(db: AsyncSession, instrument_id: uuid.UUID) -> Instrument:
    result = await db.execute(
        select(Instrument)
        .where(Instrument.id == instrument_id)
        .options(
            selectinload(Instrument.exchange),
            selectinload(Instrument.data_mappings),
            selectinload(Instrument.broker_instruments),
        )
    )
    instrument = result.scalar_one_or_none()
    if instrument is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instrument not found")
    return instrument


@router.post("/sync", response_model=SyncInstrumentsResponse)
async def sync_instruments(
    broker: str | None = Query(default=None, description="Defaults to the configured paper broker"),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> SyncInstrumentsResponse:
    """Synchronise the broker catalogue.

    Defaults to the paper broker. Live is reachable only by naming it, and even
    then only reads the catalogue — sync never places an order and never grants
    tradability (§7).
    """
    settings = get_settings()
    kind = BrokerKind(broker) if broker else default_paper_broker_kind(settings)

    broker_client = resolve_broker(kind, settings)
    try:
        result = await InstrumentSyncService(db).sync(
            broker_client, actor_label=f"api:{context.user.email}"
        )
        await db.commit()
    finally:
        await broker_client.close()

    return SyncInstrumentsResponse(
        broker=str(result.broker),
        synced_at=result.synced_at,
        total_from_broker=result.total_from_broker,
        broker_instruments_created=result.broker_instruments_created,
        broker_instruments_updated=result.broker_instruments_updated,
        instruments_created=result.instruments_created,
        instruments_needing_confirmation=result.instruments_needing_confirmation,
        delisted=result.delisted,
        errors=result.errors,
    )


@router.get("", response_model=InstrumentListResponse)
async def list_instruments(
    search: str | None = Query(default=None, description="Match name, ticker or ISIN"),
    lifecycle_state: str | None = None,
    bot_universe_only: bool = False,
    tradable_only: bool = Query(
        default=False, description="Only instruments currently available at the broker"
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> InstrumentListResponse:
    """Search the synchronised catalogue (acceptance criterion 2)."""
    conditions = []

    if search:
        pattern = f"%{search.strip()}%"
        conditions.append(
            or_(
                Instrument.name.ilike(pattern),
                Instrument.exchange_ticker.ilike(pattern),
                Instrument.isin.ilike(pattern),
            )
        )
    if lifecycle_state:
        try:
            conditions.append(Instrument.lifecycle_state == LifecycleState(lifecycle_state))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown lifecycle state {lifecycle_state!r}",
            ) from exc
    if bot_universe_only:
        conditions.append(Instrument.is_bot_universe.is_(True))
    if tradable_only:
        conditions.append(Instrument.suspended_at.is_(None))

    base = select(Instrument)
    if conditions:
        base = base.where(*conditions)

    total = await db.scalar(select(func.count()).select_from(base.subquery()))

    result = await db.execute(
        base.options(selectinload(Instrument.exchange))
        .order_by(Instrument.name.asc())
        .limit(limit)
        .offset(offset)
    )
    instruments = list(result.scalars().all())

    return InstrumentListResponse(
        items=[InstrumentResponse.model_validate(i) for i in instruments],
        total=int(total or 0),
        limit=limit,
        offset=offset,
    )


@router.get("/{instrument_id}", response_model=InstrumentDetailResponse)
async def get_instrument(
    instrument_id: uuid.UUID,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> InstrumentDetailResponse:
    """One instrument, with mappings and data coverage."""
    instrument = await _get_instrument(db, instrument_id)
    settings = get_settings()

    coverage = await CandleStore(db).coverage_summary(instrument_id, Interval.D1)

    payload = InstrumentDetailResponse.model_validate(instrument)
    payload.daily_candle_count = coverage.candle_count
    payload.daily_last_timestamp = coverage.last_timestamp
    payload.has_sufficient_history = coverage.candle_count >= settings.min_history_days_for_signal
    return payload


@router.post("/{instrument_id}/map", response_model=MapInstrumentResponse)
async def map_instrument(
    instrument_id: uuid.UUID,
    payload: MapInstrumentRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> MapInstrumentResponse:
    """Resolve a market-data symbol for this instrument (acceptance criterion 4).

    Passing `provider_symbol` skips automatic resolution and records a manual,
    user-confirmed mapping — the tier-5 escape hatch from §5.
    """
    instrument = await _get_instrument(db, instrument_id)
    provider_kind = _parse_provider(payload.provider)
    service = MappingService(db)

    if payload.provider_symbol:
        existing = await service.find_mapping(instrument_id, provider_kind)
        if existing is None:
            existing = MarketDataMapping(
                instrument_id=instrument_id,
                provider=provider_kind,
                provider_symbol=payload.provider_symbol,
                resolution_method="manual_user_override",
                confirmed_by_user=True,
                is_signal_source=payload.is_signal_source,
            )
            db.add(existing)
            await db.flush()
        else:
            existing.provider_symbol = payload.provider_symbol
            existing.resolution_method = "manual_user_override"
            existing.confirmed_by_user = True

        if payload.is_signal_source:
            await service.set_signal_source(
                instrument_id, existing.id, actor_user_id=context.user.id
            )
        await db.commit()
        await db.refresh(existing)

        return MapInstrumentResponse(
            instrument_id=instrument_id,
            provider=str(provider_kind),
            resolved=True,
            requires_confirmation=False,
            mapping=MarketDataMappingResponse.model_validate(existing),
            reason="Manually specified by user",
        )

    provider = resolve_provider(provider_kind, get_settings())
    try:
        result = await service.resolve(
            instrument,
            provider,
            is_signal_source=payload.is_signal_source,
            actor_user_id=context.user.id,
        )
        await db.commit()
    finally:
        await provider.close()

    return MapInstrumentResponse(
        instrument_id=instrument_id,
        provider=str(provider_kind),
        resolved=result.resolved,
        requires_confirmation=result.requires_confirmation,
        mapping=(
            MarketDataMappingResponse.model_validate(result.mapping) if result.mapping else None
        ),
        reason=result.reason,
    )


@router.post(
    "/{instrument_id}/mappings/{mapping_id}/confirm", response_model=MarketDataMappingResponse
)
async def confirm_mapping(
    instrument_id: uuid.UUID,
    mapping_id: uuid.UUID,
    payload: ConfirmMappingRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> MarketDataMappingResponse:
    """Confirm a mapping is correct (§5 tier 5)."""
    service = MappingService(db)
    try:
        mapping = await service.confirm(
            mapping_id, actor_user_id=context.user.id, provider_symbol=payload.provider_symbol
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if mapping.instrument_id != instrument_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mapping does not belong to this instrument",
        )
    await db.commit()
    await db.refresh(mapping)
    return MarketDataMappingResponse.model_validate(mapping)


@router.post("/{instrument_id}/ingest", response_model=IngestResponse)
async def ingest_instrument(
    instrument_id: uuid.UUID,
    payload: IngestRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> IngestResponse:
    """Backfill or incrementally refresh this instrument's daily candles.

    First call backfills; later calls request only a small overlapping tail and
    upsert (acceptance criterion 5).
    """
    instrument = await _get_instrument(db, instrument_id)
    provider_kind = _parse_provider(payload.provider)
    provider = resolve_provider(provider_kind, get_settings())

    try:
        result = await IngestionService(db).ingest_daily(
            instrument,
            provider,
            backfill_days=payload.backfill_days,
            force_full_backfill=payload.force_full_backfill,
        )
        await db.commit()
    finally:
        await provider.close()

    return IngestResponse(
        instrument_id=result.instrument_id,
        interval=str(result.interval),
        provider=str(result.provider) if result.provider else None,
        candles_written=result.candles_written,
        was_backfill=result.was_backfill,
        window_start=result.window_start,
        window_end=result.window_end,
        skipped_reason=result.skipped_reason,
        errors=result.errors,
    )


@router.get("/{instrument_id}/candles", response_model=list[CandleResponse])
async def get_candles(
    instrument_id: uuid.UUID,
    interval: str = Query(default="1d"),
    limit: int = Query(default=250, ge=1, le=2000),
    include_unclosed: bool = Query(
        default=False,
        description="Charting only. Never enable this for anything that informs a decision.",
    ),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[CandleResponse]:
    """Read stored candles."""
    await _get_instrument(db, instrument_id)
    try:
        parsed = Interval(interval)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown interval {interval!r}. Known: {[i.value for i in Interval]}",
        ) from exc

    candles = await CandleStore(db).get_candles(
        instrument_id, parsed, limit=limit, closed_only=not include_unclosed
    )
    return [
        CandleResponse(
            timestamp=c.timestamp,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            adjusted_close=c.adjusted_close,
            volume=c.volume,
            currency=c.currency,
            price_unit=str(c.price_unit),
            is_closed=c.is_closed,
            quality_status=str(c.quality_status),
            provider=str(c.provider),
        )
        for c in candles
    ]


@router.post("/{instrument_id}/suspend", response_model=InstrumentResponse)
async def suspend_instrument(
    instrument_id: uuid.UUID,
    reason: str = Query(min_length=1, max_length=500),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> InstrumentResponse:
    """Suspend an instrument — a per-instrument kill switch (§17)."""
    from datetime import UTC, datetime

    instrument = await _get_instrument(db, instrument_id)
    instrument.suspended_at = datetime.now(UTC)
    instrument.suspension_reason = reason
    instrument.lifecycle_state = LifecycleState.SUSPENDED

    await AuditService(db).record(
        kind=AuditEventKind.INSTRUMENT_SUSPENDED,
        summary=f"Instrument suspended: {instrument.name}",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="instrument",
        subject_id=str(instrument_id),
        payload={"reason": reason},
    )
    await db.commit()
    await db.refresh(instrument)
    return InstrumentResponse.model_validate(instrument)
