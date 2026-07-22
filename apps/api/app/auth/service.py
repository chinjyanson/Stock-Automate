"""Session-based authentication (§1, §17).

Server-side sessions rather than stateless JWTs. The deciding factor is
revocation: this application has a kill switch and a live-disarm requirement,
and both need an existing session to stop working *now*. A self-validating
token cannot be withdrawn before it expires without building the very session
table a JWT was supposed to avoid.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.service import AuditService
from app.config import get_settings
from app.models.enums import ActorKind, AuditEventKind
from app.models.user import User, UserSession
from app.security.crypto import (
    generate_session_token,
    hash_password,
    hash_session_token,
    needs_rehash,
    verify_password,
)

log = structlog.get_logger(__name__)

#: How recently the user must have proved their password to arm live trading.
#: Short by design: arming risks real money, and "logged in this morning" is not
#: evidence that the person at the keyboard now is the account holder (§17).
REAUTH_WINDOW = timedelta(minutes=5)


class AuthError(Exception):
    """Authentication failed. Message is deliberately non-specific."""


class SessionCreationResult:
    __slots__ = ("expires_at", "session", "token")

    def __init__(self, token: str, session: UserSession, expires_at: datetime) -> None:
        self.token = token
        self.session = session
        self.expires_at = expires_at


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._audit = AuditService(session)

    async def create_user(
        self,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
        is_admin: bool = False,
    ) -> User:
        email = email.strip().lower()
        if len(password) < 12:
            # Length over composition rules: this account can eventually place
            # real orders.
            raise ValueError("Password must be at least 12 characters")

        user = User(
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
            is_admin=is_admin,
        )
        self._session.add(user)
        await self._session.flush()

        await self._audit.record(
            kind=AuditEventKind.CREDENTIAL_CHANGED,
            summary=f"User account created: {email}",
            actor_kind=ActorKind.SYSTEM,
            subject_type="user",
            subject_id=str(user.id),
            # No password material, hashed or otherwise.
            payload={"email": email, "is_admin": is_admin},
        )
        return user

    async def update_display_name(self, user: User, display_name: str | None) -> User:
        """Change the account's display name. Not a credential — no re-auth needed."""
        cleaned = display_name.strip() if display_name else None
        user.display_name = cleaned or None
        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.SETTING_CHANGED,
            summary="Display name updated",
            actor_kind=ActorKind.USER,
            actor_user_id=user.id,
            subject_type="user",
            subject_id=str(user.id),
        )
        return user

    async def update_email(self, user: User, new_email: str) -> User:
        """Change the login email. Caller must have enforced a fresh re-auth.

        The email is the login identifier, so a change is a credential change:
        it is audited, and a collision with another account is refused rather
        than silently swallowed.
        """
        cleaned = new_email.strip().lower()
        existing = (
            await self._session.execute(select(User).where(User.email == cleaned))
        ).scalar_one_or_none()
        if existing is not None and existing.id != user.id:
            raise AuthError("That email is already in use")

        old_email = user.email
        user.email = cleaned
        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.CREDENTIAL_CHANGED,
            summary=f"Account email changed from {old_email} to {cleaned}",
            actor_kind=ActorKind.USER,
            actor_user_id=user.id,
            subject_type="user",
            subject_id=str(user.id),
            payload={"old_email": old_email, "new_email": cleaned},
        )
        return user

    async def change_password(self, user: User, new_password: str) -> User:
        """Set a new password. Caller must have enforced a fresh re-auth."""
        if len(new_password) < 12:
            raise AuthError("Password must be at least 12 characters")
        user.password_hash = hash_password(new_password)
        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.CREDENTIAL_CHANGED,
            summary="Account password changed",
            actor_kind=ActorKind.USER,
            actor_user_id=user.id,
            subject_type="user",
            subject_id=str(user.id),
            # Never any password material, hashed or otherwise.
        )
        return user

    async def authenticate(
        self,
        *,
        email: str,
        password: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> SessionCreationResult:
        """Verify credentials and open a session.

        Every failure path raises the identical error. Distinguishing "no such
        user" from "wrong password" would turn this endpoint into an account
        enumeration oracle.
        """
        email = email.strip().lower()
        result = await self._session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user is None:
            # Hash anyway so a missing account does not return measurably faster
            # than a wrong password.
            hash_password(password)
            await self._record_failed_login(email, "unknown account")
            raise AuthError("Invalid email or password")

        if not verify_password(password, user.password_hash):
            await self._record_failed_login(email, "wrong password", user_id=user.id)
            raise AuthError("Invalid email or password")

        if not user.is_active:
            await self._record_failed_login(email, "inactive account", user_id=user.id)
            raise AuthError("Invalid email or password")

        # Transparently upgrade a hash made under weaker parameters.
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)

        user.last_login_at = datetime.now(UTC)
        session_result = await self._open_session(
            user, user_agent=user_agent, ip_address=ip_address, reauthenticated=True
        )

        await self._audit.record(
            kind=AuditEventKind.USER_LOGIN,
            summary=f"User signed in: {email}",
            actor_kind=ActorKind.USER,
            actor_user_id=user.id,
            subject_type="user",
            subject_id=str(user.id),
            payload={"ip_address": ip_address, "user_agent": user_agent},
        )
        return session_result

    async def _record_failed_login(
        self, email: str, reason: str, user_id: uuid.UUID | None = None
    ) -> None:
        await self._audit.record(
            kind=AuditEventKind.USER_LOGIN_FAILED,
            summary=f"Failed sign-in attempt for {email}",
            actor_kind=ActorKind.USER,
            actor_user_id=user_id,
            subject_type="user",
            subject_id=str(user_id) if user_id else None,
            payload={"email": email, "reason": reason},
        )

    async def _open_session(
        self,
        user: User,
        *,
        user_agent: str | None,
        ip_address: str | None,
        reauthenticated: bool,
    ) -> SessionCreationResult:
        settings = get_settings()
        token = generate_session_token()
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=settings.session_ttl_seconds)

        session = UserSession(
            user_id=user.id,
            # Only the hash is stored: a database read must not be enough to
            # impersonate the user.
            token_hash=hash_session_token(token),
            expires_at=expires_at,
            user_agent=user_agent[:400] if user_agent else None,
            ip_address=ip_address,
            reauthenticated_at=now if reauthenticated else None,
        )
        self._session.add(session)
        await self._session.flush()

        return SessionCreationResult(token=token, session=session, expires_at=expires_at)

    async def resolve_session(self, token: str) -> tuple[User, UserSession] | None:
        """Look up a live session by its token. None when absent or invalid."""
        if not token:
            return None

        result = await self._session.execute(
            select(UserSession, User)
            .join(User, User.id == UserSession.user_id)
            .where(UserSession.token_hash == hash_session_token(token))
        )
        row = result.one_or_none()
        if row is None:
            return None

        session, user = row
        if not session.is_valid_at(datetime.now(UTC)):
            return None
        if not user.is_active:
            return None
        return user, session

    async def revoke_session(self, session: UserSession, *, reason: str = "logout") -> None:
        session.revoked_at = datetime.now(UTC)
        await self._session.flush()
        await self._audit.record(
            kind=AuditEventKind.USER_LOGOUT,
            summary=f"Session revoked ({reason})",
            actor_kind=ActorKind.USER,
            actor_user_id=session.user_id,
            subject_type="user_session",
            subject_id=str(session.id),
            payload={"reason": reason},
        )

    async def revoke_all_sessions(self, user_id: uuid.UUID, *, reason: str) -> int:
        """Revoke every session for a user. Used by the kill switch."""
        result = await self._session.execute(
            select(UserSession).where(
                UserSession.user_id == user_id, UserSession.revoked_at.is_(None)
            )
        )
        sessions = list(result.scalars().all())
        now = datetime.now(UTC)
        for session in sessions:
            session.revoked_at = now
        await self._session.flush()
        return len(sessions)

    async def reauthenticate(self, session: UserSession, user: User, password: str) -> bool:
        """Re-prove the password on an existing session.

        Required before arming live trading (§17). Marks the session rather than
        issuing a new one, so the freshness is attached to *this* browser
        session and expires with it.
        """
        if not verify_password(password, user.password_hash):
            await self._record_failed_login(user.email, "reauth failed", user_id=user.id)
            return False
        session.reauthenticated_at = datetime.now(UTC)
        await self._session.flush()
        return True

    @staticmethod
    def is_recently_reauthenticated(session: UserSession, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        if session.reauthenticated_at is None:
            return False
        return (now - session.reauthenticated_at) <= REAUTH_WINDOW
