"""Authentication."""

from app.auth.dependencies import (
    CSRF_COOKIE,
    CSRF_HEADER,
    AuthContext,
    get_auth_context,
    get_optional_auth_context,
    require_admin,
    require_csrf,
    require_recent_reauth,
)
from app.auth.service import REAUTH_WINDOW, AuthError, AuthService

__all__ = [
    "CSRF_COOKIE",
    "CSRF_HEADER",
    "REAUTH_WINDOW",
    "AuthContext",
    "AuthError",
    "AuthService",
    "get_auth_context",
    "get_optional_auth_context",
    "require_admin",
    "require_csrf",
    "require_recent_reauth",
]
