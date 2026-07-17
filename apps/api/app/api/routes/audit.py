"""Audit log endpoints (§19, acceptance criterion 20)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import AuditEventResponse, AuditListResponse, AuditVerifyResponse
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context
from app.db import get_db
from app.models.enums import AuditEventKind

router = APIRouter(prefix="/audit", tags=["audit"])
log = structlog.get_logger(__name__)


@router.get("", response_model=AuditListResponse)
async def list_audit_events(
    kind: str | None = Query(default=None),
    subject_type: str | None = None,
    subject_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> AuditListResponse:
    """Recent audit events, newest first.

    Read-only by construction: there is no endpoint to write or amend one, and
    the table rejects UPDATE and DELETE at the database level (§17).
    """
    parsed_kind: AuditEventKind | None = None
    if kind:
        try:
            parsed_kind = AuditEventKind(kind)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown audit event kind {kind!r}",
            ) from exc

    service = AuditService(db)
    events = await service.recent(
        limit=limit, kind=parsed_kind, subject_type=subject_type, subject_id=subject_id
    )
    total = await service.count()

    return AuditListResponse(
        items=[AuditEventResponse.model_validate(e) for e in events], total=total
    )


@router.get("/verify", response_model=AuditVerifyResponse)
async def verify_audit_chain(
    limit: int | None = Query(
        default=None, description="Verify only the first N events; omit for the whole chain"
    ),
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> AuditVerifyResponse:
    """Verify the tamper-evidence chain.

    A failure here is an incident, not a routine check going red: the append
    path and the database trigger should jointly make a break impossible.
    """
    service = AuditService(db)
    is_intact, problems = await service.verify_chain(limit=limit)
    total = await service.count()

    if not is_intact:
        log.error("audit.chain_broken", problems=problems[:10], total_events=total)

    return AuditVerifyResponse(
        is_intact=is_intact,
        events_checked=limit if limit is not None else total,
        problems=problems,
    )
