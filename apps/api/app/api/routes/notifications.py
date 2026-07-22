"""Notification preferences (§13, §19).

Thin settings surface for outbound notifications. Today that is just the daily
end-of-day email digest; the toggle is a runtime `SystemSetting`, so it changes
without a redeploy and is audited like every other operator setting.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind
from app.services.system_settings import (
    EOD_DIGEST_KEY,
    eod_digest_enabled,
    set_bool_setting,
)

router = APIRouter(prefix="/notifications", tags=["notifications"])
log = structlog.get_logger(__name__)


class NotificationSettingsResponse(BaseModel):
    #: Whether the daily end-of-day summary is emailed to the account.
    eod_digest_enabled: bool


class NotificationSettingsUpdate(BaseModel):
    eod_digest_enabled: bool


@router.get("/settings", response_model=NotificationSettingsResponse)
async def get_notification_settings(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> NotificationSettingsResponse:
    return NotificationSettingsResponse(eod_digest_enabled=await eod_digest_enabled(db))


@router.put("/settings", response_model=NotificationSettingsResponse)
async def update_notification_settings(
    payload: NotificationSettingsUpdate,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> NotificationSettingsResponse:
    """Turn the daily EOD email digest on or off. Audited."""
    await set_bool_setting(
        db,
        EOD_DIGEST_KEY,
        payload.eod_digest_enabled,
        description="Whether the daily end-of-day summary is emailed.",
        is_sensitive=False,
        user_id=context.user.id,
    )
    await AuditService(db).record(
        kind=AuditEventKind.SETTING_CHANGED,
        summary=f"EOD email digest {'enabled' if payload.eod_digest_enabled else 'disabled'}",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="system_setting",
        subject_id=EOD_DIGEST_KEY,
        payload={"eod_digest_enabled": payload.eod_digest_enabled},
    )
    await db.commit()
    return NotificationSettingsResponse(eod_digest_enabled=await eod_digest_enabled(db))
