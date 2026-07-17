"""Runtime settings and encrypted credential storage."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, StrEnumType, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import BrokerKind, ProviderKind


class SystemSetting(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Admin-editable runtime setting.

    Overlays the environment defaults from `app.config` so operational limits
    can be retuned without a redeploy (§4). Env supplies the value on first
    boot; once a row exists here it wins.
    """

    __tablename__ = "system_settings"
    __table_args__ = (UniqueConstraint("key", name="uq_system_settings_key"),)

    key: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    value: Mapped[dict[str, Any]] = mapped_column(nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    #: Guards settings whose change should require re-authentication.
    is_sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )

    def __repr__(self) -> str:
        return f"<SystemSetting {self.key}>"


class BrokerCredential(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Broker API key, encrypted at rest (§17).

    The ciphertext is Fernet (AES-128-CBC + HMAC) keyed by
    SECRETS_ENCRYPTION_KEY. The plaintext exists only transiently in the API
    process when constructing a broker client, and is never serialised to a
    response, a log line, or an audit payload.

    Demo and live credentials are separate rows with distinct `broker` values,
    so there is no code path that can reach for "the" key and get the wrong
    environment.
    """

    __tablename__ = "broker_credentials"
    __table_args__ = (
        UniqueConstraint("user_id", "broker", name="uq_broker_credentials_user_broker"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    broker: Mapped[BrokerKind] = mapped_column(StrEnumType(BrokerKind, 24), nullable=False)
    encrypted_api_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    #: Last 4 characters only, for "is this the key I think it is?" in the UI.
    key_fingerprint: Mapped[str | None] = mapped_column(String(8))
    label: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_verification_error: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        # Deliberately omits anything key-derived beyond the fingerprint.
        return f"<BrokerCredential {self.broker} ***{self.key_fingerprint or '????'}>"


class ProviderCredential(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Market-data provider API key, encrypted at rest. See BrokerCredential."""

    __tablename__ = "provider_credentials"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_provider_credentials_user_provider"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[ProviderKind] = mapped_column(StrEnumType(ProviderKind, 16), nullable=False)
    encrypted_api_key: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_fingerprint: Mapped[str | None] = mapped_column(String(8))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_verification_error: Mapped[str | None] = mapped_column(Text)


class LiveArmingSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An explicit, expiring authorisation to place live orders (§14).

    Live trading requires *all* of: the server flag LIVE_TRADING_ENABLED, live
    credentials configured, a recent reconciliation, no active risk halt, and
    an unexpired row here created by a re-authenticated user. The row exists so
    that arming is auditable, bounded in time, and revocable — a boolean toggle
    would be none of those.
    """

    __tablename__ = "live_arming_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    armed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    disarmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    disarm_reason: Mapped[str | None] = mapped_column(String(200))

    #: Ceilings the user affirmed at arming time. The risk engine treats these
    #: as hard caps for the duration of the session.
    max_live_capital: Mapped[Any] = mapped_column(String(32), nullable=False)
    max_daily_loss: Mapped[Any] = mapped_column(String(32), nullable=False)

    def is_active_at(self, now: datetime) -> bool:
        return self.disarmed_at is None and self.expires_at > now
