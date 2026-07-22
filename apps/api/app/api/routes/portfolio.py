"""Portfolio endpoints for the active venue (§19).

There is exactly one "current account": whichever venue the paper/live toggle
selects — Trading 212 demo for paper, Trading 212 live for real money. The
dashboard and this page both read it, so the product never shows two competing
balances.

Reads go through the rate-limit read cache. Trading 212 limits the portfolio
endpoint tightly (roughly one call every few seconds) and this path is hit on
every page load, so an uncached read here would trip a 429.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AccountResponse,
    OrderResponse,
    ORMModel,
    PositionResponse,
    SerializedDecimal,
)
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.broker.base import Broker
from app.broker.factory import resolve_broker
from app.broker.read_cache import broker_read_cache
from app.broker.types import BrokerAccount, BrokerPosition
from app.config import get_settings
from app.db import get_db
from app.models.reporting import DailyAccountSummary
from app.services.eod import EODSummaryService
from app.services.system_settings import active_broker_kind

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
log = structlog.get_logger(__name__)


class DailySummaryResponse(ORMModel):
    id: uuid.UUID
    broker: str
    summary_date: date
    currency: str
    cash: SerializedDecimal
    equity: SerializedDecimal
    invested: SerializedDecimal
    unrealised_pnl: SerializedDecimal
    realised_pnl: SerializedDecimal
    equity_change: SerializedDecimal | None
    open_positions: int
    trades_today: int
    active_halts: int
    created_at: datetime


async def _active_broker(db: AsyncSession) -> Broker:
    """The venue the paper/live toggle currently selects."""
    return resolve_broker(await active_broker_kind(db), session=db)


@router.get("/account", response_model=AccountResponse)
async def active_account(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> AccountResponse:
    """Cash and value of the active venue's account (paper = Trading 212 demo)."""
    kind = await active_broker_kind(db)
    settings = get_settings()

    async def _fetch() -> BrokerAccount:
        # Built only on a cache miss, so a hit makes no broker call at all.
        client = resolve_broker(kind, settings, session=db)
        try:
            return await client.get_account()
        finally:
            await client.close()

    cached = await broker_read_cache.get_or_fetch(
        f"account:{kind}", settings.broker_read_cache_ttl_seconds, _fetch
    )
    account = cached.value
    return AccountResponse(
        broker=str(kind),
        is_live=kind.is_live,
        account_id=account.masked_account_id,
        currency=account.currency,
        cash=account.cash,
        total=account.total,
        free_for_trading=account.free_for_trading,
        invested=account.invested,
        result=account.result,
        retrieved_at=account.retrieved_at,
        is_stale=cached.is_stale,
        age_seconds=int(cached.age_seconds),
    )


@router.get("/positions", response_model=list[PositionResponse])
async def active_positions(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[PositionResponse]:
    """Open positions on the active venue."""
    kind = await active_broker_kind(db)
    settings = get_settings()

    async def _fetch() -> list[BrokerPosition]:
        client = resolve_broker(kind, settings, session=db)
        try:
            return await client.get_positions()
        finally:
            await client.close()

    cached = await broker_read_cache.get_or_fetch(
        f"positions:{kind}", settings.broker_read_cache_ttl_seconds, _fetch
    )
    positions = cached.value
    return [
        PositionResponse(
            broker_ticker=p.broker_ticker,
            quantity=p.quantity,
            average_price=p.average_price,
            current_price=p.current_price,
            unrealised_pnl=p.unrealised_pnl,
            currency=p.currency,
        )
        for p in positions
    ]


@router.get("/orders", response_model=list[OrderResponse])
async def active_orders(
    include_history: bool = False,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[OrderResponse]:
    """Working (or, with history, all) orders on the active venue.

    Working orders include the resting protective stops placed after each entry.
    """
    broker = await _active_broker(db)
    try:
        orders = (
            await broker.get_order_history()
            if include_history
            else await broker.get_pending_orders()
        )
    finally:
        await broker.close()
    return [
        OrderResponse(
            broker_order_id=o.broker_order_id,
            broker_ticker=o.broker_ticker,
            side=str(o.side),
            order_type=str(o.order_type),
            quantity=o.quantity,
            filled_quantity=o.filled_quantity,
            status=str(o.status),
            average_fill_price=o.average_fill_price,
            created_at=o.created_at,
        )
        for o in orders
    ]


@router.get("/summaries", response_model=list[DailySummaryResponse])
async def paper_summaries(
    limit: int = Query(default=30, ge=1, le=365),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[DailySummaryResponse]:
    """End-of-day summaries for the active venue, most recent first."""
    rows = (
        (
            await db.execute(
                select(DailyAccountSummary)
                .where(DailyAccountSummary.broker == await active_broker_kind(db))
                .order_by(DailyAccountSummary.summary_date.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [DailySummaryResponse.model_validate(r) for r in rows]


@router.post("/summaries/run", response_model=DailySummaryResponse)
async def run_eod_summary(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> DailySummaryResponse:
    """Generate (or refresh) today's EOD summary now, without waiting for the job."""
    broker = await _active_broker(db)
    try:
        summary = await EODSummaryService(db).generate(broker)
    finally:
        await broker.close()
    await db.commit()
    await db.refresh(summary)
    return DailySummaryResponse.model_validate(summary)
