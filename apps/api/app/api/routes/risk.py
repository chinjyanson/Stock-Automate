"""Risk configuration and halt endpoints (§9, §19).

The controls behind live and paper trading, exposed for inspection and change.
Every limit change is audited (no risk limit is a silent edit), and halts —
including the blunt kill switch — are activated and cleared here as explicit,
recorded actions.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ORMModel, SerializedDecimal
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.broker.factory import resolve_broker
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind, BrokerKind
from app.models.risk import RiskHalt
from app.risk.config import load_active_risk_config
from app.risk.halts import HaltService
from app.risk.stops import StopService

router = APIRouter(prefix="/risk", tags=["risk"])
log = structlog.get_logger(__name__)


# -- Schemas ----------------------------------------------------------------


class RiskConfigResponse(ORMModel):
    id: uuid.UUID
    name: str
    is_active: bool
    risk_per_trade_pct: SerializedDecimal
    atr_stop_multiplier: SerializedDecimal
    max_position_pct: SerializedDecimal
    max_instrument_pct: SerializedDecimal
    max_total_open_risk_pct: SerializedDecimal
    max_portfolio_exposure_pct: SerializedDecimal
    monetary_position_cap: SerializedDecimal | None
    max_open_positions: int
    max_trades_per_day: int
    consecutive_loss_cooldown: int
    max_daily_realised_loss_pct: SerializedDecimal
    max_portfolio_drawdown_pct: SerializedDecimal
    correlation_benchmark_symbol: str
    correlation_window_short: int
    correlation_window_long: int
    correlation_threshold: SerializedDecimal
    max_portfolio_sp500_pct: SerializedDecimal
    trailing_stop_enabled: bool
    max_holding_days: int


class RiskConfigUpdate(BaseModel):
    """Partial update — only the fields present are changed."""

    risk_per_trade_pct: SerializedDecimal | None = None
    atr_stop_multiplier: SerializedDecimal | None = None
    max_position_pct: SerializedDecimal | None = None
    max_instrument_pct: SerializedDecimal | None = None
    max_total_open_risk_pct: SerializedDecimal | None = None
    max_portfolio_exposure_pct: SerializedDecimal | None = None
    monetary_position_cap: SerializedDecimal | None = None
    max_open_positions: int | None = None
    max_trades_per_day: int | None = None
    consecutive_loss_cooldown: int | None = None
    max_daily_realised_loss_pct: SerializedDecimal | None = None
    max_portfolio_drawdown_pct: SerializedDecimal | None = None
    correlation_benchmark_symbol: str | None = None
    correlation_window_short: int | None = None
    correlation_window_long: int | None = None
    correlation_threshold: SerializedDecimal | None = None
    max_portfolio_sp500_pct: SerializedDecimal | None = None
    trailing_stop_enabled: bool | None = None
    max_holding_days: int | None = None


class HaltResponse(ORMModel):
    id: uuid.UUID
    kind: str
    scope: str
    instrument_id: uuid.UUID | None
    reason: str
    is_active: bool
    activated_at: datetime
    cleared_at: datetime | None


class KillSwitchRequest(BaseModel):
    reason: str = "manual kill switch"


# -- Config -----------------------------------------------------------------


@router.get("/config", response_model=RiskConfigResponse)
async def get_config(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> RiskConfigResponse:
    config = await load_active_risk_config(db)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active risk configuration. Seed one before trading.",
        )
    return RiskConfigResponse.model_validate(config)


@router.put("/config", response_model=RiskConfigResponse)
async def update_config(
    payload: RiskConfigUpdate,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> RiskConfigResponse:
    """Change limits on the active configuration. Every change is audited."""
    config = await load_active_risk_config(db)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active risk configuration to update.",
        )

    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        return RiskConfigResponse.model_validate(config)

    before = {k: getattr(config, k) for k in changes}
    for key, value in changes.items():
        setattr(config, key, value)
    await db.flush()

    await AuditService(db).record(
        kind=AuditEventKind.SETTING_CHANGED,
        summary=f"Risk configuration updated: {', '.join(changes)}",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="risk_configuration",
        subject_id=str(config.id),
        payload={
            "changed": {k: str(v) for k, v in changes.items()},
            "previous": {k: str(v) for k, v in before.items()},
        },
    )
    await db.commit()
    await db.refresh(config)
    return RiskConfigResponse.model_validate(config)


# -- Halts ------------------------------------------------------------------


@router.get("/halts", response_model=list[HaltResponse])
async def list_halts(
    active_only: bool = Query(default=True),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[HaltResponse]:
    stmt = select(RiskHalt).order_by(RiskHalt.activated_at.desc())
    if active_only:
        stmt = stmt.where(RiskHalt.is_active.is_(True))
    halts = (await db.execute(stmt)).scalars().all()
    return [HaltResponse.model_validate(h) for h in halts]


@router.post("/halts/{halt_id}/clear", response_model=HaltResponse)
async def clear_halt(
    halt_id: uuid.UUID,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> HaltResponse:
    try:
        halt = await HaltService(db).clear(halt_id, actor_user_id=context.user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    await db.commit()
    await db.refresh(halt)
    return HaltResponse.model_validate(halt)


@router.post("/kill-switch", response_model=HaltResponse)
async def kill_switch(
    payload: KillSwitchRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> HaltResponse:
    """Activate the global kill switch — stops all trading at once (§9)."""
    halt = await HaltService(db).kill_switch(payload.reason, actor_user_id=context.user.id)
    await db.commit()
    await db.refresh(halt)
    return HaltResponse.model_validate(halt)


# -- Stops ------------------------------------------------------------------


class StopRunResponse(BaseModel):
    triggered: int
    closed: int
    trailed: int
    time_exits: int


class FlattenRequest(BaseModel):
    reason: str = "manual flatten"


class FlattenResponse(BaseModel):
    positions_closed: int


@router.post("/stops/run", response_model=StopRunResponse)
async def run_stops(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> StopRunResponse:
    """Run a stop-management pass now (trigger, trail, time-stop) without the job."""
    broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=db)
    config = await load_active_risk_config(db)
    result = await StopService(db).manage(broker, config)
    await db.commit()
    return StopRunResponse(**result)


@router.post("/flatten", response_model=FlattenResponse)
async def flatten(
    payload: FlattenRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> FlattenResponse:
    """Emergency exit: close every open paper position at market (§9)."""
    broker = resolve_broker(BrokerKind.INTERNAL_PAPER, session=db)
    closed = await StopService(db).emergency_exit_all(
        broker, payload.reason, actor_user_id=context.user.id
    )
    await db.commit()
    return FlattenResponse(positions_closed=closed)
