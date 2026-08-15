"""Reading the crash overlay (§9).

Read-only. The overlay records what exposure it thinks the index sleeve should
carry; acting on that is a human decision, so there is no endpoint here that
places, sizes or approves anything.

`GET /crash-overlay` is the operator's question — "where does it stand today,
and why?" — answered with the reason in words as well as the number, because a
bare exposure of 0.30 tells nobody whether a warning fired this morning or a
week ago.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import AuthContext, get_auth_context
from app.db import get_db
from app.models.crash_overlay import CrashOverlayReading
from app.services.crash_overlay import CrashOverlayService

router = APIRouter(prefix="/crash-overlay", tags=["crash-overlay"])


class OverlayReadingOut(BaseModel):
    as_of: date
    index_close: Decimal | None
    probability: Decimal | None
    trigger: Decimal | None
    is_warning: bool
    target_exposure: Decimal | None
    days_out: int
    reason: str | None

    @classmethod
    def of(cls, row: CrashOverlayReading) -> OverlayReadingOut:
        return cls(
            as_of=row.as_of,
            index_close=row.index_close,
            probability=row.probability,
            trigger=row.trigger,
            is_warning=row.is_warning,
            target_exposure=row.target_exposure,
            days_out=row.days_out,
            reason=row.reason,
        )


class OverlayStatusOut(BaseModel):
    """Today's answer, plus enough context to judge whether to believe it."""

    latest: OverlayReadingOut | None
    #: Days on record. The model is refitted from this history every night, so
    #: a thin history is a weak model and the caller should be able to see that.
    days_of_history: int


@router.get("", response_model=OverlayStatusOut)
async def read_status(
    session: AsyncSession = Depends(get_db),
    _auth: AuthContext = Depends(get_auth_context),
) -> OverlayStatusOut:
    service = CrashOverlayService(session)
    latest = await service.latest()
    return OverlayStatusOut(
        latest=OverlayReadingOut.of(latest) if latest else None,
        days_of_history=await service.count(),
    )


@router.get("/history", response_model=list[OverlayReadingOut])
async def read_history(
    limit: int = Query(90, ge=1, le=2000),
    session: AsyncSession = Depends(get_db),
    _auth: AuthContext = Depends(get_auth_context),
) -> list[OverlayReadingOut]:
    rows = await CrashOverlayService(session).history(limit=limit)
    return [OverlayReadingOut.of(row) for row in rows]
