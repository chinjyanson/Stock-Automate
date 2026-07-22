"""The trading-venue toggle: paper or live (§7, §14, §19).

The product points at exactly one venue at a time — **paper** (the Trading 212
demo account) or **live** (real money) — and this module owns that switch.

Selecting live is one click, deliberately: no typed phrase. What it is *not* is
unguarded. Live can only be selected when the server itself permits it —
`LIVE_TRADING_ENABLED` plus live credentials — and those are deployment-level
facts a browser session cannot change. That is the backstop that makes accidental
real-money trading impossible from the UI, and it is checked again by the
execution preflight before any order is built (§7 depth-in-defence).

Going back to paper is always allowed and never blocked: stopping must never be
harder than starting.
"""

from __future__ import annotations

from decimal import Decimal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import LiveStatusResponse, SetLiveModeRequest
from app.audit.service import AuditService
from app.auth.dependencies import AuthContext, get_auth_context, require_csrf
from app.config import get_settings
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind
from app.risk.config import load_active_risk_config
from app.risk.halts import HaltService
from app.services.system_settings import (
    TRADING_LIVE_MODE_KEY,
    autonomous_live_enabled,
    live_mode_enabled,
    set_bool_setting,
)

router = APIRouter(prefix="/live", tags=["live"])
log = structlog.get_logger(__name__)


async def _live_blockers(db: AsyncSession, context: AuthContext) -> list[str]:
    """Everything standing between here and real-money trading.

    Reported as a list rather than short-circuiting, so the UI can show
    everything that needs fixing at once.
    """
    settings = get_settings()
    blockers: list[str] = []

    if not settings.live_trading_enabled:
        blockers.append("Live trading is disabled on the server (LIVE_TRADING_ENABLED is false).")

    key = settings.trading212_live_api_key
    if key is None or not key.get_secret_value():
        blockers.append("No live Trading 212 credentials are configured.")

    for halt in await HaltService(db).active_halts():
        blockers.append(f"Active risk halt: {halt.kind.value} — {halt.reason}")

    return blockers


async def _status(db: AsyncSession, context: AuthContext) -> LiveStatusResponse:
    settings = get_settings()
    config = await load_active_risk_config(db)
    return LiveStatusResponse(
        live_trading_enabled_on_server=settings.live_trading_enabled,
        autonomous_enabled_on_server=await autonomous_live_enabled(db),
        live_mode=await live_mode_enabled(db),
        max_live_capital=(
            Decimal(config.max_live_capital)
            if config is not None and config.max_live_capital is not None
            else None
        ),
        max_daily_loss=(
            Decimal(config.max_daily_loss)
            if config is not None and config.max_daily_loss is not None
            else None
        ),
        blockers=await _live_blockers(db, context),
    )


@router.get("/status", response_model=LiveStatusResponse)
async def live_status(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> LiveStatusResponse:
    """Which venue is active, and everything blocking live."""
    return await _status(db, context)


@router.post("/mode", response_model=LiveStatusResponse)
async def set_live_mode(
    payload: SetLiveModeRequest,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> LiveStatusResponse:
    """Point the product at paper or live.

    Turning live *on* re-checks the server gate here rather than trusting the
    status the client read a moment ago — the answer can change. Turning it
    *off* is unconditional.
    """
    if payload.live:
        blockers = await _live_blockers(db, context)
        if blockers:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "live_trading_blocked", "blockers": blockers},
            )

    await set_bool_setting(
        db,
        TRADING_LIVE_MODE_KEY,
        payload.live,
        description="Whether the product trades the live venue rather than paper.",
        is_sensitive=True,
        user_id=context.user.id,
    )
    await AuditService(db).record(
        kind=AuditEventKind.LIVE_ARMED if payload.live else AuditEventKind.LIVE_DISARMED,
        summary=f"Trading venue switched to {'LIVE (real money)' if payload.live else 'paper'}",
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="system_setting",
        subject_id=TRADING_LIVE_MODE_KEY,
        payload={"live_mode": payload.live},
    )
    await db.commit()

    if payload.live:
        log.warning("live.mode_enabled", user_id=str(context.user.id))
    else:
        log.info("live.mode_disabled", user_id=str(context.user.id))

    return await _status(db, context)
