"""Application configuration.

Every operational limit in this module is an environment variable with a
conservative default, never a hard-coded constant (spec §4). Values that are
also editable at runtime through admin settings are read via
`app.settings_store`, which falls back to these values on first boot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Core ---------------------------------------------------------------
    environment: Environment = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    web_base_url: str = "http://localhost:3000"
    cors_allow_origins: str = "http://localhost:3000"

    # Defaults are parsed into the Dsn types rather than left as `str`, so a
    # malformed URL fails at construction instead of at first connection.
    database_url: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://trading:trading_dev_password_change_me@localhost:5433/trading_platform"
    )
    redis_url: RedisDsn = RedisDsn("redis://localhost:6380/0")

    # -- Security -----------------------------------------------------------
    #
    # There is deliberately no SESSION_SECRET. Sessions are server-side records
    # keyed by a 256-bit random token stored as a SHA-256 hash (see
    # `app.auth.service`), so there is nothing to sign and no signing key to
    # rotate. Declaring one would imply a dependency that does not exist — and
    # imply that rotating it revokes sessions, which it would not. Revocation is
    # `UserSession.revoked_at`.
    secrets_encryption_key: SecretStr = SecretStr("dev-only-insecure-encryption-key")
    session_ttl_seconds: int = 86_400
    session_cookie_name: str = "trading_session"
    session_cookie_secure: bool = False

    # -- Trading 212 --------------------------------------------------------
    #
    # Trading 212 authenticates with an API key AND an API secret, combined as
    # HTTP Basic auth. Both halves are required; a key alone returns 401. The
    # secret is shown only once at generation time — if it was not saved, the
    # key must be regenerated.
    trading212_demo_api_key: SecretStr | None = None
    trading212_demo_api_secret: SecretStr | None = None
    trading212_demo_base_url: str = "https://demo.trading212.com/api/v0"
    trading212_live_api_key: SecretStr | None = None
    trading212_live_api_secret: SecretStr | None = None
    trading212_live_base_url: str = "https://live.trading212.com/api/v0"
    trading212_max_requests_per_minute: int = 30
    trading212_timeout_seconds: float = 20.0

    #: Server-side master switch for live trading. This alone does not permit a
    #: live order: the session must additionally be armed through the UI (§14).
    live_trading_enabled: bool = False

    # -- Market data: yfinance ---------------------------------------------
    yfinance_enabled: bool = True
    yfinance_max_concurrency: int = 4
    yfinance_batch_size: int = 50
    yfinance_backoff_base_seconds: float = 2.0
    yfinance_max_retries: int = 5

    # -- Market data: Twelve Data ------------------------------------------
    twelve_data_api_key: SecretStr | None = None
    twelve_data_base_url: str = "https://api.twelvedata.com"
    twelve_data_daily_max: int = 800
    twelve_data_daily_operational_limit: int = 720
    twelve_data_daily_emergency_reserve: int = 80
    twelve_data_per_minute_max: int = 8
    twelve_data_per_minute_operational_limit: int = 7

    # -- Market data: EODHD -------------------------------------------------
    eodhd_api_key: SecretStr | None = None
    eodhd_base_url: str = "https://eodhd.com/api"
    eodhd_daily_operational_limit: int = 18
    eodhd_daily_emergency_reserve: int = 2

    # -- Data quality gates -------------------------------------------------
    stale_daily_candle_max_age_hours: int = 36
    stale_intraday_candle_max_age_minutes: int = 45
    require_closed_candles: bool = True
    min_history_days_for_signal: int = 252

    # -- Push notifications -------------------------------------------------
    vapid_public_key: str | None = None
    vapid_private_key: SecretStr | None = None
    vapid_subject: str = "mailto:you@example.com"

    # -- Observability ------------------------------------------------------
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "trading-platform-api"

    # -- Derived ------------------------------------------------------------

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def sync_database_url(self) -> str:
        """Alembic and other sync consumers need a non-async driver."""
        return str(self.database_url).replace("+asyncpg", "+psycopg2", 1)

    @property
    def twelve_data_safe_daily_budget(self) -> int:
        """Requests we may spend before the emergency reserve is touched."""
        return min(
            self.twelve_data_daily_operational_limit,
            self.twelve_data_daily_max - self.twelve_data_daily_emergency_reserve,
        )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @model_validator(mode="after")
    def _validate_budget_coherence(self) -> Settings:
        if self.twelve_data_daily_operational_limit > self.twelve_data_daily_max:
            raise ValueError(
                "TWELVE_DATA_DAILY_OPERATIONAL_LIMIT must not exceed TWELVE_DATA_DAILY_MAX"
            )
        if self.twelve_data_per_minute_operational_limit > self.twelve_data_per_minute_max:
            raise ValueError(
                "TWELVE_DATA_PER_MINUTE_OPERATIONAL_LIMIT must not exceed "
                "TWELVE_DATA_PER_MINUTE_MAX"
            )
        if self.twelve_data_safe_daily_budget <= 0:
            raise ValueError(
                "Twelve Data emergency reserve consumes the whole daily maximum; "
                "no requests could ever be spent"
            )
        return self

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> Settings:
        """Refuse to boot production with a development placeholder secret.

        Failing at startup is deliberate: an encryption key left at its default
        would mean every stored broker credential on a live-capable deployment
        is encrypted under a value published in this repository.
        """
        if not self.is_production:
            return self

        encryption_key = self.secrets_encryption_key.get_secret_value()
        if encryption_key.startswith("dev-only-") or encryption_key.startswith("CHANGE_ME"):
            raise ValueError(
                "Refusing to start in production with a placeholder "
                "SECRETS_ENCRYPTION_KEY. Generate one with `openssl rand -base64 32`."
            )
        if not self.session_cookie_secure:
            raise ValueError("SESSION_COOKIE_SECURE must be true in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
