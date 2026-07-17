"""Authentication endpoints (§17, §19)."""

from __future__ import annotations

import secrets

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    LoginRequest,
    ReauthRequest,
    SessionResponse,
    UserResponse,
)
from app.auth.dependencies import CSRF_COOKIE, AuthContext, get_auth_context, require_csrf
from app.auth.service import AuthError, AuthService
from app.config import get_settings
from app.db import get_db
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])
log = structlog.get_logger(__name__)


def _set_session_cookies(response: Response, token: str, csrf_token: str) -> None:
    """Install the session and CSRF cookies.

    The session cookie is HttpOnly: JavaScript must not be able to read it, so
    an XSS bug cannot exfiltrate a session that can place orders. The CSRF
    cookie is deliberately *not* HttpOnly — the frontend has to read it to echo
    it back in a header, which is what makes the double-submit check work.
    """
    settings = get_settings()

    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_ttl_seconds,
        path="/",
    )
    response.set_cookie(
        key=CSRF_COOKIE,
        value=csrf_token,
        httponly=False,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_ttl_seconds,
        path="/",
    )


@router.post("/login", response_model=SessionResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Sign in and open a session.

    Not CSRF-protected: there is no session to forge a request from yet, and a
    forged login only logs the victim into the attacker's own account, which
    this endpoint's response does not leak anything useful about.
    """
    service = AuthService(db)
    try:
        result = await service.authenticate(
            email=payload.email,
            password=payload.password,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None,
        )
    except AuthError as exc:
        await db.commit()  # Persist the failed-login audit event.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    csrf_token = secrets.token_urlsafe(32)

    user = await db.get(User, result.session.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    await db.commit()

    _set_session_cookies(response, result.token, csrf_token)
    return SessionResponse(
        user=UserResponse.model_validate(user),
        expires_at=result.expires_at,
        csrf_token=csrf_token,
        # A fresh sign-in is by definition a fresh password proof.
        is_recently_reauthenticated=True,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> Response:
    """Revoke the current session server-side and clear the cookies."""
    await AuthService(db).revoke_session(context.session, reason="logout")
    await db.commit()

    settings = get_settings()
    response.delete_cookie(settings.session_cookie_name, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=SessionResponse)
async def me(request: Request, context: AuthContext = Depends(get_auth_context)) -> SessionResponse:
    """Current session state, used by the frontend to bootstrap."""
    return SessionResponse(
        user=UserResponse.model_validate(context.user),
        expires_at=context.session.expires_at,
        csrf_token=request.cookies.get(CSRF_COOKIE, ""),
        is_recently_reauthenticated=context.is_recently_reauthenticated,
    )


@router.post("/reauthenticate", response_model=SessionResponse)
async def reauthenticate(
    payload: ReauthRequest,
    request: Request,
    context: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_csrf),
) -> SessionResponse:
    """Re-prove the password without opening a new session.

    Required before arming live trading (§17). Marks the existing session
    fresh rather than issuing a new token, so the freshness expires with the
    session and cannot be carried elsewhere.
    """
    service = AuthService(db)
    ok = await service.reauthenticate(context.session, context.user, payload.password)
    await db.commit()

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Password is incorrect"
        )

    return SessionResponse(
        user=UserResponse.model_validate(context.user),
        expires_at=context.session.expires_at,
        csrf_token=request.cookies.get(CSRF_COOKIE, ""),
        is_recently_reauthenticated=True,
    )
