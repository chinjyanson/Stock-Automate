"""Strategy configuration, decisions and manual runs (§8, §19).

Thin: list and tune the configured strategies, inspect what they decided, and
trigger an evaluation on demand. Every order a run produces still passes the risk
engine — this router does not execute trades itself, it asks the engine to.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ORMModel, SerializedDecimal
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind
from app.models.strategy import StrategyConfiguration, StrategyDecision
from app.strategies.engine import StrategyEngine

router = APIRouter(prefix="/strategies", tags=["strategies"])
log = structlog.get_logger(__name__)


# -- Schemas ----------------------------------------------------------------


class StrategyConfigResponse(ORMModel):
    id: uuid.UUID
    kind: str
    name: str
    is_active: bool
    interval: str
    operating_mode: str
    auto_execute: bool
    params: dict[str, Any] | None
    universe: dict[str, Any] | None
    account_equity: SerializedDecimal | None


class StrategyConfigUpdate(BaseModel):
    is_active: bool | None = None
    auto_execute: bool | None = None
    interval: str | None = None
    params: dict[str, Any] | None = None
    universe: dict[str, Any] | None = None
    account_equity: SerializedDecimal | None = None


class StrategyDecisionResponse(ORMModel):
    id: uuid.UUID
    run_id: uuid.UUID
    instrument_id: uuid.UUID
    kind: str
    side: str
    conviction: SerializedDecimal
    outcome: str
    reason: str
    metrics: dict[str, Any] | None
    proposal_id: uuid.UUID | None
    created_at: datetime


class StrategyRunResponse(BaseModel):
    run_id: uuid.UUID
    considered: int
    signals: int
    proposals: int
    executed: int
    rejected: int


# -- Config -----------------------------------------------------------------


@router.get("", response_model=list[StrategyConfigResponse])
async def list_strategies(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[StrategyConfigResponse]:
    rows = (
        (await db.execute(select(StrategyConfiguration).order_by(StrategyConfiguration.name)))
        .scalars()
        .all()
    )
    return [StrategyConfigResponse.model_validate(r) for r in rows]


@router.put("/{strategy_id}/config", response_model=StrategyConfigResponse)
async def update_strategy(
    strategy_id: uuid.UUID,
    payload: StrategyConfigUpdate,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> StrategyConfigResponse:
    """Tune a strategy. Activating one, or changing its params, is audited."""
    config = await db.get(StrategyConfiguration, strategy_id)
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        return StrategyConfigResponse.model_validate(config)
    for key, value in changes.items():
        setattr(config, key, value)
    await db.flush()

    await AuditService(db).record(
        kind=AuditEventKind.SETTING_CHANGED,
        summary=f"Strategy '{config.name}' updated: {', '.join(changes)}",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="strategy_configuration",
        subject_id=str(config.id),
        payload={"changed": {k: str(v) for k, v in changes.items()}},
    )
    await db.commit()
    await db.refresh(config)
    return StrategyConfigResponse.model_validate(config)


@router.get("/decisions", response_model=list[StrategyDecisionResponse])
async def list_decisions(
    strategy_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[StrategyDecisionResponse]:
    stmt = select(StrategyDecision).order_by(StrategyDecision.created_at.desc()).limit(limit)
    if strategy_id is not None:
        stmt = stmt.where(StrategyDecision.configuration_id == strategy_id)
    rows = (await db.execute(stmt)).scalars().all()
    return [StrategyDecisionResponse.model_validate(r) for r in rows]


@router.post("/{strategy_id}/run", response_model=StrategyRunResponse)
async def run_strategy(
    strategy_id: uuid.UUID,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> StrategyRunResponse:
    """Evaluate a strategy now, without waiting for its schedule."""
    config = await db.get(StrategyConfiguration, strategy_id)
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")

    summary = await StrategyEngine(db).run(
        config, selection_reason="manual", actor_user_id=context.user.id
    )
    await db.commit()
    return StrategyRunResponse(
        run_id=summary.run_id,
        considered=summary.considered,
        signals=summary.signals,
        proposals=summary.proposals,
        executed=summary.executed,
        rejected=summary.rejected,
    )
