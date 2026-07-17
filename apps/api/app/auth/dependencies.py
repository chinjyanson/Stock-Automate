"""FastAPI authentication and CSRF dependencies (§17)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import AuthService
from app.config import get_settings
from app.db import get_db
from app.models.user import User, UserSession
from app.security.crypto import constant_time_compare

#: Header carrying the CSRF token on state-changing requests.
CSRF_HEADER = "X-CSRF-Token"
#: Cookie mirroring it, for the double-submit check.
CSRF_COOKIE = "trading_csrf"


@dataclass(slots=True)
class AuthContext:
    """The authenticated caller, plus the session backing them."""

    user: User
    session: UserSession

    @property
    def is_recently_reauthenticated(self) -> bool:
        return AuthService.is_recently_reauthenticated(self.session)


async def get_auth_context(request: Request, db: AsyncSession = Depends(get_db)) -> AuthContext:
    """Resolve the caller, or 401.

    The cookie is read directly rather than via a bearer header: the browser
    must not be able to hold this token in JavaScript-readable storage, so it
    is an HttpOnly cookie, and CSRF is handled separately by `require_csrf`.
    """
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Cookie"},
        )

    resolved = await AuthService(db).resolve_session(token)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session is invalid or has expired",
        )

    user, session = resolved
    return AuthContext(user=user, session=session)


async def get_optional_auth_context(
    request: Request, db: AsyncSession = Depends(get_db)
) -> AuthContext | None:
    """Like `get_auth_context`, but tolerates anonymity."""
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    resolved = await AuthService(db).resolve_session(token)
    if resolved is None:
        return None
    user, session = resolved
    return AuthContext(user=user, session=session)


async def require_admin(context: AuthContext = Depends(get_auth_context)) -> AuthContext:
    if not context.user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required"
        )
    return context


async def require_recent_reauth(
    context: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    """Demand a fresh password proof.

    Guards live arming and other irreversible actions. 403 with a machine-
    readable code so the UI can prompt for a password rather than logging the
    user out, which is what a bare 401 would imply.
    """
    if not context.is_recently_reauthenticated:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "reauthentication_required",
                "message": "Confirm your password to continue.",
            },
        )
    return context


async def require_csrf(request: Request) -> None:
    """Double-submit CSRF check on state-changing requests (§17).

    The session cookie is SameSite=Lax, which already blocks cross-site POSTs
    from a plain form. This is the second layer, because Lax has exceptions and
    the endpoints behind it place orders.

    Safe methods are exempt: they must not change state, and requiring a token
    on them would break ordinary navigation.
    """
    if request.method in {"GET", "HEAD", "OPTIONS", "TRACE"}:
        return

    header_token = request.headers.get(CSRF_HEADER)
    cookie_token = request.cookies.get(CSRF_COOKIE)

    if not header_token or not cookie_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token missing")
    if not constant_time_compare(header_token, cookie_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")
