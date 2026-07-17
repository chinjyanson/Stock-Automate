"""Account, position and reconciliation endpoints (§19)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AccountResponse,
    OrderResponse,
    PositionResponse,
    ReconciliationDiscrepancyResponse,
    ReconciliationResponse,
)
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.broker.factory import default_paper_broker_kind, resolve_broker
from app.broker.types import BrokerError
from app.config import get_settings
from app.db import get_db
from app.models.enums import BrokerKind

router = APIRouter(tags=["account"])
log = structlog.get_logger(__name__)


def _resolve_kind(broker: str | None) -> BrokerKind:
    settings = get_settings()
    if broker is None:
        return default_paper_broker_kind(settings)
    try:
        return BrokerKind(broker)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown broker {broker!r}. Known: {[b.value for b in BrokerKind]}",
        ) from exc


@router.get("/account", response_model=AccountResponse)
async def get_account(
    broker: str | None = Query(default=None),
    context: AuthContext = Depends(get_auth_context),
) -> AccountResponse:
    """Cash and account summary.

    The account identifier is masked before it leaves the server (§17); the
    browser has no use for the full value.
    """
    kind = _resolve_kind(broker)
    client = resolve_broker(kind, get_settings())
    try:
        account = await client.get_account()
    except BrokerError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Broker error: {exc}"
        ) from exc
    finally:
        await client.close()

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
    )


@router.get("/positions", response_model=list[PositionResponse])
async def get_positions(
    broker: str | None = Query(default=None),
    context: AuthContext = Depends(get_auth_context),
) -> list[PositionResponse]:
    kind = _resolve_kind(broker)
    client = resolve_broker(kind, get_settings())
    try:
        positions = await client.get_positions()
    except BrokerError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Broker error: {exc}"
        ) from exc
    finally:
        await client.close()

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
async def get_orders(
    broker: str | None = Query(default=None),
    include_history: bool = Query(default=False),
    context: AuthContext = Depends(get_auth_context),
) -> list[OrderResponse]:
    kind = _resolve_kind(broker)
    client = resolve_broker(kind, get_settings())
    try:
        orders = (
            await client.get_order_history()
            if include_history
            else await client.get_pending_orders()
        )
    except BrokerError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Broker error: {exc}"
        ) from exc
    finally:
        await client.close()

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


@router.post("/broker/reconcile", response_model=ReconciliationResponse)
async def reconcile(
    broker: str | None = Query(default=None),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> ReconciliationResponse:
    """Compare broker state against local records (§10).

    Live arming requires a recent, clean reconciliation, so this is not merely
    diagnostic — it is a precondition for trading.
    """
    kind = _resolve_kind(broker)
    client = resolve_broker(kind, get_settings())
    try:
        result = await client.reconcile()
    except BrokerError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Broker error: {exc}"
        ) from exc
    finally:
        await client.close()

    return ReconciliationResponse(
        broker=str(result.broker),
        reconciled_at=result.reconciled_at,
        positions_checked=result.positions_checked,
        orders_checked=result.orders_checked,
        is_clean=result.is_clean,
        discrepancies=[
            ReconciliationDiscrepancyResponse(
                kind=d.kind,
                broker_ticker=d.broker_ticker,
                detail=d.detail,
                local_value=d.local_value,
                broker_value=d.broker_value,
            )
            for d in result.discrepancies
        ],
    )
