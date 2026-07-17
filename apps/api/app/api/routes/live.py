"""Live-trading arming and disarming (§7, §14, §19).

Arming is the moment this system becomes capable of losing real money, so it is
gated by five independent conditions that must *all* hold (§7):

  1. LIVE_TRADING_ENABLED on the server
  2. Live credentials configured
  3. A recent broker reconciliation
  4. No active risk halt
  5. A re-authenticated user typing an exact confirmation phrase, and affirming
     capital and loss ceilings

They are checked as a *list of blockers* rather than short-circuiting on the
first failure, because the UI should tell the user everything standing in the
way at once, not make them fix five things one refresh at a time.

Arming is time-boxed and expires on its own. A live session that stays armed
because someone forgot to turn it off is the failure mode this design refuses.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ArmLiveRequest, LiveStatusResponse
from app.audit.service import AuditService
from app.auth.dependencies import (
    AuthContext,
    get_auth_context,
    require_csrf,
    require_recent_reauth,
)
from app.broker.factory import resolve_broker
from app.broker.types import BrokerError
from app.config import get_settings
from app.db import get_db
from app.models.enums import ActorKind, AuditEventKind, BrokerKind
from app.models.system import LiveArmingSession

router = APIRouter(prefix="/live", tags=["live"])
log = structlog.get_logger(__name__)

#: Must be typed exactly. Deliberately awkward — muscle memory should not be
#: able to produce it.
CONFIRMATION_PHRASE = "I UNDERSTAND THIS TRADES REAL MONEY"

#: How recently the broker must have been reconciled for arming to proceed.
RECONCILIATION_MAX_AGE = timedelta(minutes=15)


async def _active_arming(db: AsyncSession, user_id: object) -> LiveArmingSession | None:
    result = await db.execute(
        select(LiveArmingSession)
        .where(
            LiveArmingSession.user_id == user_id,
            LiveArmingSession.disarmed_at.is_(None),
            LiveArmingSession.expires_at > datetime.now(UTC),
        )
        .order_by(LiveArmingSession.armed_at.desc())
    )
    return result.scalars().first()


async def _collect_blockers(db: AsyncSession, context: AuthContext) -> list[str]:
    """Everything currently preventing live trading."""
    settings = get_settings()
    blockers: list[str] = []

    if not settings.live_trading_enabled:
        blockers.append("Live trading is disabled on the server (LIVE_TRADING_ENABLED is false).")

    key = settings.trading212_live_api_key
    if key is None or not key.get_secret_value():
        blockers.append("No live Trading 212 credentials are configured.")

    if not context.is_recently_reauthenticated:
        blockers.append("Password re-confirmation is required.")

    # Risk halts arrive with the risk engine in Phase 3. Until then there is no
    # halt state to consult, and saying so is more honest than reporting a
    # green check for a system that does not exist yet.
    blockers.append(
        "Risk engine is not yet implemented (Phase 3); live trading remains unavailable."
    )

    return blockers


@router.get("/status", response_model=LiveStatusResponse)
async def live_status(
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> LiveStatusResponse:
    """Current live-trading state and everything blocking it."""
    settings = get_settings()
    arming = await _active_arming(db, context.user.id)
    blockers = await _collect_blockers(db, context)

    return LiveStatusResponse(
        live_trading_enabled_on_server=settings.live_trading_enabled,
        is_armed=arming is not None,
        armed_at=arming.armed_at if arming else None,
        expires_at=arming.expires_at if arming else None,
        max_live_capital=Decimal(arming.max_live_capital) if arming else None,
        max_daily_loss=Decimal(arming.max_daily_loss) if arming else None,
        blockers=blockers,
    )


@router.post("/arm", response_model=LiveStatusResponse)
async def arm_live(
    payload: ArmLiveRequest,
    context: AuthContext = Depends(require_recent_reauth),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> LiveStatusResponse:
    """Arm live trading for a bounded window.

    Every precondition is re-checked here rather than trusted from the status
    endpoint: the client may have read status minutes ago, and the answer can
    change.
    """
    settings = get_settings()

    if payload.confirmation_phrase.strip() != CONFIRMATION_PHRASE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Confirmation phrase must be exactly: {CONFIRMATION_PHRASE}",
        )

    blockers = await _collect_blockers(db, context)
    if blockers:
        # 409: the request is well-formed, the system state forbids it.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "live_trading_blocked", "blockers": blockers},
        )

    # Reconciliation must be recent *and* clean. Arming against an unreconciled
    # broker means arming without knowing what is already open (§7).
    broker = resolve_broker(BrokerKind.TRADING212_LIVE, settings)
    try:
        reconciliation = await broker.reconcile()
    except BrokerError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Cannot arm: broker reconciliation failed: {exc}",
        ) from exc
    finally:
        await broker.close()

    if not reconciliation.is_clean:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "reconciliation_required",
                "message": "Broker state diverges from local records. Resolve before arming.",
                "discrepancies": [d.detail for d in reconciliation.discrepancies],
            },
        )

    now = datetime.now(UTC)
    arming = LiveArmingSession(
        user_id=context.user.id,
        armed_at=now,
        expires_at=now + timedelta(minutes=payload.duration_minutes),
        max_live_capital=str(payload.max_live_capital),
        max_daily_loss=str(payload.max_daily_loss),
    )
    db.add(arming)

    await AuditService(db).record(
        kind=AuditEventKind.LIVE_ARMED,
        summary=(
            f"Live trading armed for {payload.duration_minutes} minutes "
            f"(max capital {payload.max_live_capital}, max daily loss {payload.max_daily_loss})"
        ),
        actor_kind=ActorKind.USER,
        actor_user_id=context.user.id,
        subject_type="live_arming_session",
        subject_id=str(arming.id),
        payload={
            "max_live_capital": str(payload.max_live_capital),
            "max_daily_loss": str(payload.max_daily_loss),
            "duration_minutes": payload.duration_minutes,
            "expires_at": arming.expires_at.isoformat(),
            "reconciliation_clean": True,
        },
    )
    await db.commit()
    await db.refresh(arming)

    log.warning(
        "live.armed",
        user_id=str(context.user.id),
        expires_at=arming.expires_at.isoformat(),
        max_live_capital=str(payload.max_live_capital),
    )

    return LiveStatusResponse(
        live_trading_enabled_on_server=settings.live_trading_enabled,
        is_armed=True,
        armed_at=arming.armed_at,
        expires_at=arming.expires_at,
        max_live_capital=payload.max_live_capital,
        max_daily_loss=payload.max_daily_loss,
        blockers=[],
    )


@router.post("/disarm", response_model=LiveStatusResponse)
async def disarm_live(
    reason: str = "user requested",
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> LiveStatusResponse:
    """Disarm live trading immediately.

    Deliberately the easiest action in this module: no re-authentication, no
    confirmation phrase. Stopping must never be harder than starting.
    Disarming when not armed succeeds quietly — in an emergency, an error
    response is not a useful answer to "make it stop".
    """
    settings = get_settings()
    arming = await _active_arming(db, context.user.id)

    if arming is not None:
        arming.disarmed_at = datetime.now(UTC)
        arming.disarm_reason = reason[:200]

        await AuditService(db).record(
            kind=AuditEventKind.LIVE_DISARMED,
            summary=f"Live trading disarmed: {reason}",
            actor_kind=ActorKind.USER,
            actor_user_id=context.user.id,
            subject_type="live_arming_session",
            subject_id=str(arming.id),
            payload={"reason": reason},
        )
        await db.commit()
        log.warning("live.disarmed", user_id=str(context.user.id), reason=reason)

    return LiveStatusResponse(
        live_trading_enabled_on_server=settings.live_trading_enabled,
        is_armed=False,
        blockers=await _collect_blockers(db, context),
    )
