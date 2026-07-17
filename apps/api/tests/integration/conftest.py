"""Integration test fixtures.

These run against a real PostgreSQL, not SQLite. That is deliberate: the schema
depends on PostgreSQL behaviour that SQLite does not have — the audit
immutability trigger, `nextval`, advisory locks, `ON CONFLICT` upserts and
JSONB. A suite that passed on SQLite would be testing a different system than
the one that runs.

Each test gets a dedicated database created from the migrations and dropped
afterwards, so tests cannot leak state into one another.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings
from app.models import Base


def _admin_url() -> str:
    """Connection URL for the maintenance database.

    CREATE/DROP DATABASE cannot run inside a transaction or while connected to
    the target, so this points at `postgres` instead.
    """
    base = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://trading:trading_dev_password_change_me@localhost:5433/postgres",
    )
    return base


@pytest_asyncio.fixture
async def test_database() -> AsyncIterator[str]:
    """Create a throwaway database; yield its URL; drop it afterwards."""
    db_name = f"test_{uuid.uuid4().hex[:12]}"
    admin_engine = create_async_engine(_admin_url(), isolation_level="AUTOCOMMIT")

    async with admin_engine.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{db_name}"'))

    url = _admin_url().rsplit("/", 1)[0] + f"/{db_name}"

    try:
        yield url
    finally:
        async with admin_engine.connect() as conn:
            # Terminate stragglers first; DROP DATABASE fails while any
            # connection remains, which would leak databases across a run.
            await conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": db_name},
            )
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
        await admin_engine.dispose()


@pytest_asyncio.fixture
async def engine(test_database: str) -> AsyncIterator[object]:
    """Engine against the throwaway database, with the schema applied.

    The schema is built from `Base.metadata` plus the raw DDL that Alembic
    carries (the audit triggers and the chain sequence). Running the real
    migration chain per-test would be slower and, more importantly, would test
    Alembic rather than the application.
    """
    test_engine = create_async_engine(test_database, poolclass=None)

    async with test_engine.begin() as conn:
        await conn.execute(text("CREATE SEQUENCE IF NOT EXISTS audit_events_sequence_seq"))
        await conn.run_sync(Base.metadata.create_all)
        # Mirror the migration's immutability guarantee, so tests exercise the
        # same constraint production has.
        await conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION audit_events_reject_mutation()
                RETURNS TRIGGER AS $$
                BEGIN
                    RAISE EXCEPTION 'audit_events is append-only: % is not permitted', TG_OP
                        USING ERRCODE = 'restrict_violation';
                END;
                $$ LANGUAGE plpgsql;
                """
            )
        )
        await conn.execute(
            text(
                "CREATE TRIGGER audit_events_immutable BEFORE UPDATE OR DELETE "
                "ON audit_events FOR EACH ROW "
                "EXECUTE FUNCTION audit_events_reject_mutation()"
            )
        )
        await conn.execute(
            text(
                "CREATE TRIGGER audit_events_no_truncate BEFORE TRUNCATE "
                "ON audit_events FOR EACH STATEMENT "
                "EXECUTE FUNCTION audit_events_reject_mutation()"
            )
        )

    yield test_engine
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db(engine: object) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[arg-type]
    async with factory() as session:
        yield session


@pytest.fixture
def test_settings(test_database: str) -> Settings:
    return Settings(
        environment="test",
        database_url=test_database,  # type: ignore[arg-type]
        live_trading_enabled=False,
        trading212_demo_api_key=None,
        trading212_live_api_key=None,
    )


@pytest_asyncio.fixture
async def client(engine: object, test_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client bound to the app, with the database dependency overridden."""
    from app.db import get_db
    from app.main import app

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[arg-type]

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    get_settings.cache_clear()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()
    get_settings.cache_clear()
