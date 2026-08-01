"""Scanner endpoints (§19)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import AliasChoices, BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.schemas import ORMModel, SerializedDecimal
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.broker.factory import default_paper_broker_kind, resolve_broker
from app.broker.read_cache import broker_read_cache
from app.config import get_settings
from app.data.factory import resolve_provider
from app.data.yfinance_provider import YFinanceProvider
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind, Interval, ProviderKind
from app.models.instrument import Instrument
from app.models.market_data import Candle
from app.models.scanner import (
    ScannerConfiguration,
    ScannerResult,
    ScannerRun,
)
from app.scanner import watchlist
from app.scanner.engine import MIN_BARS_TO_SCORE, ScannerEngine
from app.scanner.proposals import ProposalError, ProposalInputs, ProposalService
from app.scanner.rotation import select_instruments
from app.services.backfill import BackfillService
from app.services.system_settings import (
    SCANNER_AUTORUN_KEY,
    scanner_auto_run_enabled,
    set_bool_setting,
)

router = APIRouter(prefix="/scanner", tags=["scanner"])
log = structlog.get_logger(__name__)


# -- Schemas ----------------------------------------------------------------


class RunScannerRequest(BaseModel):
    #: Explicit instruments to scan; omit to use the rotating selection.
    instrument_ids: list[uuid.UUID] | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)


class ScannerRunResponse(ORMModel):
    id: uuid.UUID
    status: str
    started_at: datetime
    completed_at: datetime | None
    instruments_considered: int
    instruments_scored: int
    instruments_skipped: int
    screening_candidates: int
    watchlist_candidates: int
    selection_reason: str | None


class ScannerResultResponse(ORMModel):
    id: uuid.UUID
    instrument_id: uuid.UUID
    #: Instrument identity, so the table can show the stock and its venue without
    #: a second request per row.
    instrument_name: str | None = None
    exchange_name: str | None = None
    exchange_mic: str | None = None
    #: The instrument's sector (yfinance/GICS), so the table can show industry
    #: context alongside the sector score.
    sector: str | None = None
    #: The score driving classification/ranking (momentum, value, or a blend,
    #: per the run's configuration).
    primary_score: SerializedDecimal
    core_score: SerializedDecimal
    trend_score: SerializedDecimal
    momentum_score: SerializedDecimal
    risk_score: SerializedDecimal
    liquidity_score: SerializedDecimal
    positioning_score: SerializedDecimal
    #: Health of the instrument's own sector (via its sector-ETF proxy).
    sector_score: SerializedDecimal | None = None
    #: Two more of the five final-score factors: turning-up strength, and
    #: soundness (fundamentals + low-risk + liquidity).
    reversal_score: SerializedDecimal | None = None
    quality_score: SerializedDecimal | None = None
    #: Insider *buying*, 50 = neutral, up to 100. Null when nobody senior bought.
    insider_score: SerializedDecimal | None = None
    #: Fraction of the score removed for insider selling (0..0.30).
    insider_sell_penalty: SerializedDecimal | None = None
    fundamental_score: SerializedDecimal | None
    #: The valuation lens (0-100): how cheap the instrument looks. Separate from
    #: the momentum core score.
    value_score: SerializedDecimal | None
    price_value_score: SerializedDecimal | None
    fundamental_value_score: SerializedDecimal | None
    classification: str
    data_completeness: SerializedDecimal
    data_freshness_days: SerializedDecimal | None
    confidence: SerializedDecimal
    candles_used: int
    is_trading212_tradable: bool
    #: When this score was computed. The default listing shows each instrument's
    #: latest result across runs, so rows can legitimately carry different dates
    #: — without this the table would present a mixed-age ranking as if every row
    #: were scored today. Both names validate: `created_at` reading the ORM row,
    #: `scanned_at` when `ScannerResultDetail` is rebuilt from this model's dump.
    scanned_at: datetime = Field(validation_alias=AliasChoices("scanned_at", "created_at"))


class ScannerResultDetail(ScannerResultResponse):
    # instrument_name / exchange_* are inherited from ScannerResultResponse.
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    #: Value-lens signals: {"positive": [...], "negative": [...]}.
    value_positive_signals: list[str] = Field(default_factory=list)
    value_negative_signals: list[str] = Field(default_factory=list)
    metrics: dict[str, object] = Field(default_factory=dict)


class ProposeTradeRequest(BaseModel):
    #: Capital the proposal is sized against. Defaults to live account equity.
    account_equity: SerializedDecimal | None = None
    risk_per_trade: SerializedDecimal | None = None


class TradeProposalResponse(ORMModel):
    id: uuid.UUID
    instrument_id: uuid.UUID
    status: str
    side: str
    proposed_quantity: SerializedDecimal
    max_position_value: SerializedDecimal
    risk_amount: SerializedDecimal
    risk_pct: SerializedDecimal
    indicative_entry_price: SerializedDecimal
    proposed_stop_price: SerializedDecimal | None
    currency: str
    reason: str
    expires_at: datetime


# -- Helpers ----------------------------------------------------------------


async def _active_configuration(db: AsyncSession) -> ScannerConfiguration | None:
    result = await db.execute(
        select(ScannerConfiguration).where(ScannerConfiguration.is_active.is_(True)).limit(1)
    )
    return result.scalar_one_or_none()


def _signal_items(payload: dict[str, object] | None) -> list[str]:
    if not payload:
        return []
    items = payload.get("items")
    return [str(i) for i in items] if isinstance(items, list) else []


# -- Endpoints --------------------------------------------------------------


@router.post("/run", response_model=ScannerRunResponse)
async def run_scanner(
    payload: RunScannerRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> ScannerRunResponse:
    """Run a scan over an explicit set, or the rotating selection (§6)."""
    configuration = await _active_configuration(db)

    if payload.instrument_ids:
        result = await db.execute(
            select(Instrument).where(Instrument.id.in_(payload.instrument_ids))
        )
        instruments = list(result.scalars().all())
        reason = f"explicit selection of {len(instruments)} instruments"
        # Named by hand, so it ranks nothing: see `ScannerRun.is_ad_hoc`.
        is_ad_hoc = True
    else:
        instruments, reason = await select_instruments(
            db, configuration=configuration, limit=payload.limit
        )
        is_ad_hoc = False

    if not instruments:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No instruments to scan. Sync a broker catalogue and ingest daily "
                "candles first — the scanner reads the local candle store."
            ),
        )

    summary = await ScannerEngine(db).run(
        instruments,
        configuration=configuration,
        selection_reason=reason,
        actor_user_id=context.user.id,
        is_ad_hoc=is_ad_hoc,
    )
    await db.commit()

    run = await db.get(ScannerRun, summary.run_id)
    assert run is not None
    return ScannerRunResponse.model_validate(run)


@router.get("/results", response_model=list[ScannerResultResponse])
async def list_results(
    run_id: uuid.UUID | None = Query(default=None),
    classification: str | None = Query(default=None),
    min_score: float = Query(default=0.0, ge=0, le=100),
    tradable_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=2000),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[ScannerResultResponse]:
    """List scanner results ranked by score — each instrument's latest by default.

    Naming a `run_id` returns exactly that run. Without one the listing is
    *every instrument's most recent result*, not the most recent run's results.

    Those differ, and the second is what someone looking at a scanner wants. A
    run covers a rotating slice of the catalogue, so "the latest run" is a few
    hundred names chosen by where the sweep happened to be, and a stock rescanned
    on its own would reduce the whole table to one row. Combining across runs is
    sound here only because the scores are absolute rather than cohort-relative
    — a 74 means the same thing whichever night it was computed. `scanned_at`
    carries the age of each row so a stale score cannot pass for a fresh one.
    """
    stmt = select(ScannerResult)

    if run_id is not None:
        stmt = stmt.where(ScannerResult.run_id == run_id)
    else:
        newest_per_instrument = select(
            ScannerResult.id,
            func.row_number()
            .over(
                partition_by=ScannerResult.instrument_id,
                order_by=ScannerResult.created_at.desc(),
            )
            .label("rank"),
        ).subquery()
        stmt = stmt.join(
            newest_per_instrument, newest_per_instrument.c.id == ScannerResult.id
        ).where(newest_per_instrument.c.rank == 1)

    if classification:
        stmt = stmt.where(ScannerResult.classification == classification)
    if tradable_only:
        stmt = stmt.where(ScannerResult.is_trading212_tradable.is_(True))
    # Filter and rank by the primary score — the one that leads under the run's
    # configuration (momentum, value, or a blend).
    stmt = stmt.where(ScannerResult.primary_score >= Decimal(str(min_score)))

    stmt = stmt.order_by(ScannerResult.primary_score.desc()).limit(limit)
    results = list((await db.execute(stmt)).scalars().all())

    # Batch-load instrument identity for the rows in one query, rather than a
    # per-row lookup (which would not scale as the result set grows).
    instrument_ids = {r.instrument_id for r in results}
    instruments = {
        i.id: i
        for i in (
            await db.execute(
                select(Instrument)
                .where(Instrument.id.in_(instrument_ids))
                .options(selectinload(Instrument.exchange))
            )
        )
        .scalars()
        .all()
    }

    responses: list[ScannerResultResponse] = []
    for r in results:
        response = ScannerResultResponse.model_validate(r)
        instrument = instruments.get(r.instrument_id)
        if instrument is not None:
            response.instrument_name = instrument.name
            response.sector = instrument.sector
            if instrument.exchange is not None:
                response.exchange_name = instrument.exchange.name
                response.exchange_mic = instrument.exchange.mic
        responses.append(response)
    return responses


@router.get("/results/{result_id}", response_model=ScannerResultDetail)
async def get_result(
    result_id: uuid.UUID,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> ScannerResultDetail:
    """One result with full score breakdown and provenance (§6)."""
    result = await db.get(ScannerResult, result_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")

    instrument_result = await db.execute(
        select(Instrument)
        .where(Instrument.id == result.instrument_id)
        .options(selectinload(Instrument.exchange))
    )
    instrument = instrument_result.scalar_one_or_none()
    # Validate the base scalar fields from the ORM row, then compose the detail
    # with the signal lists extracted from their JSONB {"items": [...]} shape —
    # those column names collide with the schema's list[str] fields, so they
    # cannot be auto-populated from the ORM.
    base = ScannerResultResponse.model_validate(result)
    base_data = base.model_dump()
    base_data.update(
        instrument_name=instrument.name if instrument else None,
        sector=instrument.sector if instrument else None,
        exchange_name=instrument.exchange.name if instrument and instrument.exchange else None,
        exchange_mic=instrument.exchange.mic if instrument and instrument.exchange else None,
    )
    value_signals = result.value_signals or {}
    return ScannerResultDetail(
        **base_data,
        positive_signals=_signal_items(result.positive_signals),
        negative_signals=_signal_items(result.negative_signals),
        missing_information=_signal_items(result.missing_information),
        value_positive_signals=[str(s) for s in value_signals.get("positive", [])],
        value_negative_signals=[str(s) for s in value_signals.get("negative", [])],
        metrics=result.metrics or {},
    )


@router.post("/results/{result_id}/propose-trade", response_model=TradeProposalResponse)
async def propose_trade(
    result_id: uuid.UUID,
    payload: ProposeTradeRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> TradeProposalResponse:
    """Generate a proposed trade from a candidate (§6).

    This never places an order. It produces a proposal that must be explicitly
    approved, and even then execution waits on the Phase 3 risk engine.
    """
    result = await db.get(ScannerResult, result_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")

    if not result.is_trading212_tradable:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This instrument is not tradable through the connected broker.",
        )

    # Equity: use the request value, else read the (cached) broker account.
    equity = payload.account_equity
    if equity is None:
        settings = get_settings()
        kind = default_paper_broker_kind(settings)

        async def _fetch() -> Decimal:
            broker = resolve_broker(kind, settings)
            try:
                account = await broker.get_account()
                return account.total
            finally:
                await broker.close()

        cached = await broker_read_cache.get_or_fetch(
            f"account:{kind}", settings.broker_read_cache_ttl_seconds, _fetch
        )
        equity = cached.value

    inputs = ProposalInputs(account_equity=equity)
    if payload.risk_per_trade is not None:
        inputs.risk_per_trade = payload.risk_per_trade

    try:
        proposal = await ProposalService(db).propose_from_result(
            result, inputs, actor_user_id=context.user.id
        )
    except ProposalError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    await db.commit()
    await db.refresh(proposal)
    return TradeProposalResponse.model_validate(proposal)


# -- Watchlist ---------------------------------------------------------------


class WatchlistEntryResponse(BaseModel):
    instrument_id: uuid.UUID
    instrument_name: str | None = None
    exchange_mic: str | None = None
    note: str | None = None
    added_at: datetime
    #: Whether the local store holds enough daily history to actually score this
    #: instrument. False means the next scan will *not* cover it, however it is
    #: ranked — reported so a pin that cannot be honoured says so.
    is_scannable: bool


class AddToWatchlistRequest(BaseModel):
    instrument_id: uuid.UUID
    note: str | None = Field(default=None, max_length=500)


async def _scannable_ids(db: AsyncSession, instrument_ids: set[uuid.UUID]) -> set[uuid.UUID]:
    """Which of `instrument_ids` have enough stored daily bars to be scored.

    The same floor the rotation applies, asked here so the watchlist can warn at
    pin time instead of leaving the operator to infer it from an absence.
    """
    if not instrument_ids:
        return set()
    rows = await db.execute(
        select(Candle.instrument_id)
        .where(
            Candle.instrument_id.in_(instrument_ids),
            Candle.interval == Interval.D1,
            Candle.is_closed.is_(True),
        )
        .group_by(Candle.instrument_id)
        .having(func.count(Candle.id) >= MIN_BARS_TO_SCORE)
    )
    return {row[0] for row in rows.all()}


@router.get("/watchlist", response_model=list[WatchlistEntryResponse])
async def list_watchlist(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[WatchlistEntryResponse]:
    """The instruments pinned for the next scan (§6, rotation tier 1)."""
    entries = await watchlist.list_entries(db, context.user.id)
    scannable = await _scannable_ids(db, {e.instrument_id for e, _ in entries})
    return [
        WatchlistEntryResponse(
            instrument_id=entry.instrument_id,
            instrument_name=instrument.name if instrument else None,
            exchange_mic=(instrument.exchange.mic if instrument and instrument.exchange else None),
            note=entry.note,
            added_at=entry.created_at,
            is_scannable=entry.instrument_id in scannable,
        )
        for entry, instrument in entries
    ]


@router.post(
    "/watchlist", response_model=WatchlistEntryResponse, status_code=status.HTTP_201_CREATED
)
async def add_to_watchlist(
    payload: AddToWatchlistRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> WatchlistEntryResponse:
    """Pin an instrument so the next scan covers it before the rotating sweep.

    Idempotent — pinning twice returns the existing entry rather than failing.
    """
    instrument_result = await db.execute(
        select(Instrument)
        .where(Instrument.id == payload.instrument_id)
        .options(selectinload(Instrument.exchange))
    )
    instrument = instrument_result.scalar_one_or_none()
    if instrument is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instrument not found")

    entry, created = await watchlist.add(
        db, context.user.id, payload.instrument_id, note=payload.note
    )
    if created:
        await AuditService(db).record(
            kind=AuditEventKind.WATCHLIST_CHANGED,
            summary=f"Watchlisted '{instrument.name}' for the next scan",
            actor_kind=ActorKind.USER,
            actor_user_id=context.user.id,
            subject_type="instrument",
            subject_id=str(instrument.id),
            payload={"action": "added"},
        )
    await db.commit()

    scannable = await _scannable_ids(db, {payload.instrument_id})
    return WatchlistEntryResponse(
        instrument_id=entry.instrument_id,
        instrument_name=instrument.name,
        exchange_mic=instrument.exchange.mic if instrument.exchange else None,
        note=entry.note,
        added_at=entry.created_at,
        is_scannable=payload.instrument_id in scannable,
    )


class RefreshInstrumentResponse(BaseModel):
    """Outcome of pulling fresh candles for one instrument and rescoring it."""

    instrument_id: uuid.UUID
    instrument_name: str | None = None
    #: Daily bars written by the refresh. Zero is normal when the store was
    #: already current; it does not mean the refresh failed.
    candles_written: int = 0
    #: The new score, or null when the instrument still cannot be scored.
    result: ScannerResultResponse | None = None
    #: Why there is no result — an unsupported venue, no data from the provider,
    #: or too little history. Null when `result` is present.
    reason: str | None = None


@router.post("/instruments/{instrument_id}/refresh", response_model=RefreshInstrumentResponse)
async def refresh_instrument(
    instrument_id: uuid.UUID,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> RefreshInstrumentResponse:
    """Fetch current candles for one instrument and score it immediately (§4, §6).

    The rotation reaches most of the catalogue only over days, so a stock that
    has just been pinned would otherwise sit there carrying whatever score it
    last had — or none at all — until the sweep came round. That is precisely
    when someone is looking at it, so this closes the gap: map the symbol if it
    has never been mapped, pull its daily history, and rescore it now.

    The scan is marked ad hoc. It records a score and nothing more; the nightly
    ranking, and the strategy universe drawn from it, are untouched.
    """
    instrument_row = await db.execute(
        select(Instrument)
        .where(Instrument.id == instrument_id)
        .options(selectinload(Instrument.exchange))
    )
    instrument = instrument_row.scalar_one_or_none()
    if instrument is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instrument not found")

    provider = resolve_provider(ProviderKind.YFINANCE)
    # `backfill` needs the concrete provider: it uses the batched download, which
    # is not part of the generic interface.
    assert isinstance(provider, YFinanceProvider)
    try:
        backfill = await BackfillService(db).backfill([instrument], provider)
    finally:
        await provider.close()

    if backfill.skipped_unsupported:
        await db.commit()
        return RefreshInstrumentResponse(
            instrument_id=instrument_id,
            instrument_name=instrument.name,
            reason=(
                f"{instrument.name} trades on a venue this product cannot map to a "
                f"market-data symbol, so it cannot be scored."
            ),
        )
    if backfill.errors:
        await db.commit()
        return RefreshInstrumentResponse(
            instrument_id=instrument_id,
            instrument_name=instrument.name,
            reason=f"Market data could not be fetched: {backfill.errors[0]}",
        )

    configuration = await _active_configuration(db)
    summary = await ScannerEngine(db).run(
        [instrument],
        configuration=configuration,
        selection_reason="on-demand refresh of a single instrument",
        actor_user_id=context.user.id,
        is_ad_hoc=True,
    )

    scored = (
        await db.execute(
            select(ScannerResult).where(
                ScannerResult.run_id == summary.run_id,
                ScannerResult.instrument_id == instrument_id,
            )
        )
    ).scalar_one_or_none()
    await db.commit()

    if scored is None:
        return RefreshInstrumentResponse(
            instrument_id=instrument_id,
            instrument_name=instrument.name,
            candles_written=backfill.candles_written,
            reason=(
                f"{instrument.name} has fewer than {MIN_BARS_TO_SCORE} daily bars stored, "
                f"which is too little history to score. The provider returned "
                f"{backfill.candles_written} new bar(s)."
            ),
        )

    response = ScannerResultResponse.model_validate(scored)
    response.instrument_name = instrument.name
    response.sector = instrument.sector
    if instrument.exchange is not None:
        response.exchange_name = instrument.exchange.name
        response.exchange_mic = instrument.exchange.mic
    return RefreshInstrumentResponse(
        instrument_id=instrument_id,
        instrument_name=instrument.name,
        candles_written=backfill.candles_written,
        result=response,
    )


@router.delete("/watchlist/{instrument_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_watchlist(
    instrument_id: uuid.UUID,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> None:
    """Unpin an instrument. It falls back to the ordinary rotation."""
    removed = await watchlist.remove(db, context.user.id, instrument_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Instrument is not on the watchlist"
        )
    instrument = await db.get(Instrument, instrument_id)
    await AuditService(db).record(
        kind=AuditEventKind.WATCHLIST_CHANGED,
        summary=f"Removed '{instrument.name if instrument else instrument_id}' from the watchlist",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="instrument",
        subject_id=str(instrument_id),
        payload={"action": "removed"},
    )
    await db.commit()


class ScannerSettingsResponse(BaseModel):
    #: Whether the scheduled rotating scan runs (when the worker is up).
    auto_run_enabled: bool


class ScannerSettingsUpdate(BaseModel):
    auto_run_enabled: bool


@router.get("/settings", response_model=ScannerSettingsResponse)
async def get_scanner_settings(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> ScannerSettingsResponse:
    return ScannerSettingsResponse(auto_run_enabled=await scanner_auto_run_enabled(db))


@router.put("/settings", response_model=ScannerSettingsResponse)
async def update_scanner_settings(
    payload: ScannerSettingsUpdate,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> ScannerSettingsResponse:
    """Turn the scheduled rotating scan on or off. Audited; manual scans are unaffected."""
    await set_bool_setting(
        db,
        SCANNER_AUTORUN_KEY,
        payload.auto_run_enabled,
        description="Whether the scheduled rotating scan runs.",
        is_sensitive=False,
        user_id=context.user.id,
    )
    await AuditService(db).record(
        kind=AuditEventKind.SETTING_CHANGED,
        summary=f"Scheduled scanning {'enabled' if payload.auto_run_enabled else 'disabled'}",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="system_setting",
        subject_id=SCANNER_AUTORUN_KEY,
        payload={"auto_run_enabled": payload.auto_run_enabled},
    )
    await db.commit()
    return ScannerSettingsResponse(auto_run_enabled=await scanner_auto_run_enabled(db))


# Re-exported so the approvals router can reuse the schema.
__all__ = ["TradeProposalResponse", "router"]
