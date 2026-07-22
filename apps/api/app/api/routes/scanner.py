"""Scanner endpoints (§19)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.schemas import ORMModel, SerializedDecimal
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.broker.factory import default_paper_broker_kind, resolve_broker
from app.broker.read_cache import broker_read_cache
from app.config import get_settings
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind
from app.models.instrument import Instrument
from app.models.scanner import (
    ScannerConfiguration,
    ScannerResult,
    ScannerRun,
)
from app.scanner.engine import ScannerEngine
from app.scanner.proposals import ProposalError, ProposalInputs, ProposalService
from app.scanner.rotation import select_instruments
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
    #: The score driving classification/ranking (momentum, value, or a blend,
    #: per the run's configuration).
    primary_score: SerializedDecimal
    core_score: SerializedDecimal
    trend_score: SerializedDecimal
    momentum_score: SerializedDecimal
    risk_score: SerializedDecimal
    liquidity_score: SerializedDecimal
    positioning_score: SerializedDecimal
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
    else:
        instruments, reason = await select_instruments(
            db, configuration=configuration, limit=payload.limit
        )

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
    limit: int = Query(default=50, ge=1, le=500),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[ScannerResultResponse]:
    """List scanner results, most recent run first by default, ranked by score."""
    stmt = select(ScannerResult)

    if run_id is not None:
        stmt = stmt.where(ScannerResult.run_id == run_id)
    else:
        # Default to the latest completed run.
        latest = await db.execute(
            select(ScannerRun.id).order_by(ScannerRun.started_at.desc()).limit(1)
        )
        latest_id = latest.scalar_one_or_none()
        if latest_id is not None:
            stmt = stmt.where(ScannerResult.run_id == latest_id)

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
