"""Internal paper portfolio endpoints (§19).

The internal paper broker is DB-backed, so — unlike the Trading 212 reads in
`account.py`, which go through the rate-limit read cache — these are plain local
queries and are served directly with the request's session. They read the paper
venue's own truth (cash, positions, orders) written by the execution path.
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
from app.broker.factory import resolve_broker
from app.db import get_db
from app.models.enums import BrokerKind
from app.models.reporting import DailyAccountSummary
from app.services.eod import EODSummaryService

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


@router.get("/account", response_model=AccountResponse)
async def paper_account(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> AccountResponse:
    """Cash and value of the internal paper account."""
    broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=db)
    account = await broker.get_account()
    return AccountResponse(
        broker=str(BrokerKind.INTERNAL_PAPER),
        is_live=False,
        account_id=account.masked_account_id,
        currency=account.currency,
        cash=account.cash,
        total=account.total,
        free_for_trading=account.free_for_trading,
        invested=account.invested,
        result=account.result,
        retrieved_at=account.retrieved_at,
    )


@router.get("/positions", response_model=list[PositionResponse])
async def paper_positions(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[PositionResponse]:
    """Open positions on the internal paper venue."""
    broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=db)
    positions = await broker.get_positions()
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
async def paper_orders(
    include_history: bool = False,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[OrderResponse]:
    """Working (or, with history, all) orders on the internal paper venue.

    Working orders include the resting protective stops placed after each entry.
    """
    broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=db)
    orders = (
        await broker.get_order_history()
        if include_history
        else await broker.get_pending_orders()
    )
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
    """End-of-day summaries for the internal paper account, most recent first."""
    rows = (
        (
            await db.execute(
                select(DailyAccountSummary)
                .where(DailyAccountSummary.broker == BrokerKind.INTERNAL_PAPER)
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
    """Generate (or refresh) today's paper EOD summary now, without waiting for the job."""
    broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=db)
    summary = await EODSummaryService(db).generate(broker)
    await db.commit()
    await db.refresh(summary)
    return DailySummaryResponse.model_validate(summary)
