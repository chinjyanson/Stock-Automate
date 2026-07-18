"""Seed script: `uv run python -m app.seed`.

Idempotent — safe to run against an existing database. Re-running must not
duplicate exchanges or reset a password that has been changed.

Seeds only what the application cannot discover for itself:

  * **Exchanges.** Instrument sync needs venues to exist before it can attach
    instruments to them, and MIC/timezone/calendar are reference data, not
    something to be inferred from a broker payload.
  * **A development user.** So a fresh clone can sign in.
  * **System settings.** The runtime-editable overlay for §4's budgets.

It does *not* seed instruments. Those come from a real broker sync, which is
the flow that actually needs exercising (acceptance criterion 3: add an
instrument without changing source code).
"""

from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal

import structlog
from sqlalchemy import select

from app.audit.service import AuditService
from app.auth.service import AuthService
from app.config import get_settings
from app.db import session_scope
from app.models.enums import ActorKind, AuditEventKind
from app.models.instrument import Exchange
from app.models.system import SystemSetting
from app.models.user import User
from app.observability.logging import configure_logging

log = structlog.get_logger(__name__)

#: Venues Trading 212 commonly lists. MIC is the identity key (§5); the IANA
#: timezone drives session boundaries and DST; calendar_name ties to
#: pandas_market_calendars where a calendar exists.
_EXCHANGES: list[dict[str, str | None]] = [
    {
        "mic": "XLON",
        "name": "London Stock Exchange",
        "country": "GB",
        "timezone": "Europe/London",
        "currency": "GBP",
        "calendar_name": "LSE",
    },
    {
        "mic": "XNAS",
        "name": "Nasdaq Stock Market",
        "country": "US",
        "timezone": "America/New_York",
        "currency": "USD",
        "calendar_name": "NASDAQ",
    },
    {
        "mic": "XNYS",
        "name": "New York Stock Exchange",
        "country": "US",
        "timezone": "America/New_York",
        "currency": "USD",
        "calendar_name": "NYSE",
    },
    {
        "mic": "ARCX",
        "name": "NYSE Arca",
        "country": "US",
        "timezone": "America/New_York",
        "currency": "USD",
        "calendar_name": "NYSE",
    },
    {
        "mic": "XETR",
        "name": "Xetra",
        "country": "DE",
        "timezone": "Europe/Berlin",
        "currency": "EUR",
        "calendar_name": "XETR",
    },
    {
        "mic": "XAMS",
        "name": "Euronext Amsterdam",
        "country": "NL",
        "timezone": "Europe/Amsterdam",
        "currency": "EUR",
        "calendar_name": "XAMS",
    },
    {
        "mic": "XPAR",
        "name": "Euronext Paris",
        "country": "FR",
        "timezone": "Europe/Paris",
        "currency": "EUR",
        "calendar_name": "XPAR",
    },
    {
        "mic": "XMIL",
        "name": "Borsa Italiana",
        "country": "IT",
        "timezone": "Europe/Rome",
        "currency": "EUR",
        "calendar_name": "XMIL",
    },
    {
        "mic": "XMAD",
        "name": "Bolsa de Madrid",
        "country": "ES",
        "timezone": "Europe/Madrid",
        "currency": "EUR",
        "calendar_name": "XMAD",
    },
    {
        "mic": "XSWX",
        "name": "SIX Swiss Exchange",
        "country": "CH",
        "timezone": "Europe/Zurich",
        "currency": "CHF",
        "calendar_name": "SIX",
    },
]


#: Runtime-editable settings. Seeded from environment defaults; once a row
#: exists it wins over the environment (§4: budgets must be admin-editable).
def _default_settings() -> list[dict[str, object]]:
    settings = get_settings()
    return [
        {
            "key": "twelve_data.daily_operational_limit",
            "value": {"value": settings.twelve_data_daily_operational_limit},
            "description": "Twelve Data requests per UTC day before non-critical work stops.",
        },
        {
            "key": "twelve_data.per_minute_operational_limit",
            "value": {"value": settings.twelve_data_per_minute_operational_limit},
            "description": "Twelve Data requests per minute.",
        },
        {
            "key": "eodhd.daily_operational_limit",
            "value": {"value": settings.eodhd_daily_operational_limit},
            "description": "EODHD requests per UTC day. Verification and gap-fill only.",
        },
        {
            "key": "scanner.candidate_threshold",
            "value": {"screening": 75, "watchlist": 60},
            "description": (
                "Score thresholds. 75+ is a screening candidate, 60-74 a watchlist "
                "candidate, below 60 does not pass the screen."
            ),
        },
        {
            "key": "data_quality.min_history_days_for_signal",
            "value": {"value": settings.min_history_days_for_signal},
            "description": "Closed daily candles required before an instrument may signal.",
        },
    ]


async def seed_exchanges() -> int:
    created = 0
    async with session_scope() as session:
        for row in _EXCHANGES:
            existing = await session.execute(select(Exchange).where(Exchange.mic == row["mic"]))
            if existing.scalar_one_or_none() is not None:
                continue
            session.add(Exchange(**row))
            created += 1
    log.info("seed.exchanges", created=created, total=len(_EXCHANGES))
    return created


async def seed_settings() -> int:
    created = 0
    async with session_scope() as session:
        for row in _default_settings():
            existing = await session.execute(
                select(SystemSetting).where(SystemSetting.key == row["key"])
            )
            if existing.scalar_one_or_none() is not None:
                # Never clobber a value an operator has tuned.
                continue
            session.add(SystemSetting(**row))
            created += 1
    log.info("seed.settings", created=created)
    return created


async def seed_scanner_configuration() -> int:
    """Seed the default scanner configuration (§6) if none exists.

    Weights, thresholds and universe filters match the §6 defaults. Idempotent:
    an operator's tuned configuration is never overwritten.
    """
    from app.models.scanner import ScannerConfiguration
    from app.scanner.scoring import DEFAULT_THRESHOLDS, DEFAULT_WEIGHTS

    created = 0
    async with session_scope() as session:
        existing = await session.execute(
            select(ScannerConfiguration).where(ScannerConfiguration.name == "default")
        )
        if existing.scalar_one_or_none() is None:
            session.add(
                ScannerConfiguration(
                    name="default",
                    is_active=True,
                    include_stocks=True,
                    include_etfs=True,
                    trading212_only=True,
                    max_instruments_per_scan=200,
                    weights=dict(DEFAULT_WEIGHTS),
                    thresholds=dict(DEFAULT_THRESHOLDS),
                    benchmark_symbol="SPY",
                    # Value-primary ("buy low"): the value lens leads and momentum
                    # is a secondary signal. Flip to momentum_weight=1/value=0 for
                    # a momentum-primary ("buy strength") screen.
                    momentum_weight=Decimal("0.3"),
                    value_weight=Decimal("0.7"),
                )
            )
            created = 1
    log.info("seed.scanner_configuration", created=created)
    return created


async def seed_dev_user() -> bool:
    """Create a development user if none exists.

    Refuses outside development. A known-password account on a live-capable
    deployment is an open door, and seeding is exactly the kind of step that
    gets run against the wrong environment.
    """
    settings = get_settings()
    if settings.is_production:
        log.warning("seed.user_skipped", reason="Refusing to seed a user in production")
        return False

    email = os.environ.get("SEED_USER_EMAIL", "dev@example.com")
    password = os.environ.get("SEED_USER_PASSWORD", "development-password")

    async with session_scope() as session:
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none() is not None:
            log.info("seed.user_exists", email=email)
            return False

        service = AuthService(session)
        user = await service.create_user(
            email=email, password=password, display_name="Development User", is_admin=True
        )
        await AuditService(session).record(
            kind=AuditEventKind.SETTING_CHANGED,
            summary="Database seeded",
            actor_kind=ActorKind.SYSTEM,
            actor_label="seed_script",
            payload={"seeded_user": email},
        )
        log.info("seed.user_created", email=email, user_id=str(user.id))

    print(f"\n  Development user created: {email}")
    print(f"  Password: {password}")
    print("  Change this before exposing the application to a network.\n")
    return True


async def main() -> int:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, json_output=False)

    log.info("seed.starting", environment=settings.environment)

    await seed_exchanges()
    await seed_settings()
    await seed_scanner_configuration()
    await seed_dev_user()

    log.info("seed.completed")
    print("Seed complete. Next: sign in, then POST /instruments/sync to pull the catalogue.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
