"""Approval workflow endpoints (§19).

The human gate between a scanner candidate and any order. Approval records the
decision and re-runs the §6 checks; it then hands the APPROVED proposal to the
risk engine, which sizes and gates it and — for the paper venue — places the
order. The risk engine can still refuse: an approved proposal that the engine
rejects lands in REJECTED_BY_RISK (200, with that status) rather than silently,
so "the user said yes but the system said no" is visible.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.scanner import TradeProposalResponse
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.db import get_db
from app.models.scanner import ProposalStatus, TradeProposal
from app.risk.execution import ExecutionError, ExecutionService
from app.scanner.proposals import ProposalError, ProposalService

router = APIRouter(prefix="/approvals", tags=["approvals"])
log = structlog.get_logger(__name__)


class DecisionRequest(BaseModel):
    note: str | None = None


@router.get("", response_model=list[TradeProposalResponse])
async def list_approvals(
    pending_only: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=200),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[TradeProposalResponse]:
    """List trade proposals awaiting (or past) a decision."""
    stmt = select(TradeProposal)
    if pending_only:
        stmt = stmt.where(TradeProposal.status == ProposalStatus.PENDING_APPROVAL)
    stmt = stmt.order_by(TradeProposal.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return [TradeProposalResponse.model_validate(p) for p in result.scalars().all()]


@router.post("/{proposal_id}/approve", response_model=TradeProposalResponse)
async def approve(
    proposal_id: uuid.UUID,
    payload: DecisionRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> TradeProposalResponse:
    """Approve a proposal, then size, gate and (for paper) execute it (§6, §9).

    Requires an authenticated session (the dependency above) — a proposal must
    never be executable from an unauthenticated context such as a notification
    action. Execution runs against the internal paper venue; the risk engine can
    still reduce or reject the order.
    """
    try:
        proposal = await ProposalService(db).approve(
            proposal_id, actor_user_id=context.user.id, note=payload.note
        )
    except ProposalError as exc:
        await db.commit()  # persist any audit event written during the failed attempt
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    try:
        proposal = await ExecutionService(db).execute_approved(
            proposal, actor_user_id=context.user.id
        )
    except ExecutionError as exc:
        # The intent state + audit written during the failed attempt must persist.
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    await db.commit()
    await db.refresh(proposal)
    return TradeProposalResponse.model_validate(proposal)


@router.post("/{proposal_id}/reject", response_model=TradeProposalResponse)
async def reject(
    proposal_id: uuid.UUID,
    payload: DecisionRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> TradeProposalResponse:
    try:
        proposal = await ProposalService(db).reject(
            proposal_id, actor_user_id=context.user.id, note=payload.note
        )
    except ProposalError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    await db.commit()
    await db.refresh(proposal)
    return TradeProposalResponse.model_validate(proposal)
