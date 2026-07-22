"""Account profile mutations and the notification digest setting.

Service-level against real PostgreSQL: the email/display-name/password changes go
through `AuthService`, and the digest toggle through the `SystemSetting` overlay.
The HTTP re-auth guard on the email/password routes is exercised separately; here
we assert the underlying behaviour those routes delegate to.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import AuthError, AuthService
from app.models.enums import AuditEventKind
from app.security.crypto import verify_password
from app.services.system_settings import (
    EOD_DIGEST_KEY,
    eod_digest_enabled,
    set_bool_setting,
)


class TestProfileMutations:
    async def test_update_display_name_sets_and_clears(self, db: AsyncSession) -> None:
        service = AuthService(db)
        user = await service.create_user(email="p1@example.com", password="a-long-enough-password")
        await db.commit()

        await service.update_display_name(user, "  Ada  ")
        await db.commit()
        assert user.display_name == "Ada"

        # Blank clears back to null rather than storing an empty string.
        await service.update_display_name(user, "   ")
        await db.commit()
        assert user.display_name is None

    async def test_update_email_changes_login_and_audits(self, db: AsyncSession) -> None:
        service = AuthService(db)
        user = await service.create_user(email="p2@example.com", password="a-long-enough-password")
        await db.commit()

        await service.update_email(user, "New@Example.com")
        await db.commit()
        assert user.email == "new@example.com"  # normalised to lower-case

        events = await service._audit.recent(kind=AuditEventKind.CREDENTIAL_CHANGED)
        assert any("email changed" in e.summary for e in events)

    async def test_update_email_rejects_a_collision(self, db: AsyncSession) -> None:
        service = AuthService(db)
        await service.create_user(email="taken@example.com", password="a-long-enough-password")
        mover = await service.create_user(email="mover@example.com", password="a-long-enough-password")
        await db.commit()

        with pytest.raises(AuthError):
            await service.update_email(mover, "taken@example.com")

    async def test_change_password_rejects_short_and_accepts_valid(self, db: AsyncSession) -> None:
        service = AuthService(db)
        user = await service.create_user(email="p3@example.com", password="a-long-enough-password")
        await db.commit()

        with pytest.raises(AuthError):
            await service.change_password(user, "too-short")

        await service.change_password(user, "another-long-password")
        await db.commit()
        assert verify_password("another-long-password", user.password_hash)


class TestDigestSetting:
    async def test_default_is_off(self, db: AsyncSession) -> None:
        assert await eod_digest_enabled(db) is False

    async def test_toggle_on_and_off(self, db: AsyncSession) -> None:
        await set_bool_setting(
            db,
            EOD_DIGEST_KEY,
            True,
            description="test",
            is_sensitive=False,
            user_id=None,
        )
        await db.commit()
        assert await eod_digest_enabled(db) is True

        await set_bool_setting(
            db,
            EOD_DIGEST_KEY,
            False,
            description="test",
            is_sensitive=False,
            user_id=None,
        )
        await db.commit()
        assert await eod_digest_enabled(db) is False
