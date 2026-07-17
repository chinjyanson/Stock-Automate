"""Health and readiness endpoints (§18, §19)."""

from __future__ import annotations

import asyncio
import time

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ComponentHealth, HealthResponse
from app.broker.factory import resolve_broker
from app.config import get_settings
from app.db import get_db
from app.models.enums import BrokerKind

router = APIRouter(tags=["health"])
log = structlog.get_logger(__name__)


async def _check_database(db: AsyncSession) -> ComponentHealth:
    started = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        return ComponentHealth(name="database", healthy=False, detail=str(exc)[:200])
    return ComponentHealth(
        name="database",
        healthy=True,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


async def _check_redis() -> ComponentHealth:
    settings = get_settings()
    started = time.perf_counter()
    client: aioredis.Redis | None = None
    try:
        client = aioredis.from_url(str(settings.redis_url))
        await client.ping()
    except Exception as exc:
        return ComponentHealth(name="redis", healthy=False, detail=str(exc)[:200])
    finally:
        if client is not None:
            await client.aclose()
    return ComponentHealth(
        name="redis", healthy=True, latency_ms=int((time.perf_counter() - started) * 1000)
    )


async def _check_broker(kind: BrokerKind, name: str) -> ComponentHealth:
    settings = get_settings()
    started = time.perf_counter()
    try:
        broker = resolve_broker(kind, settings)
    except Exception as exc:
        # Not configured is a *state*, not a failure. A demo-only deployment
        # reporting unhealthy for live would be noise.
        return ComponentHealth(name=name, healthy=True, detail=f"Not configured: {exc}"[:200])

    try:
        healthy = await broker.health_check()
        return ComponentHealth(
            name=name,
            healthy=healthy,
            detail=None if healthy else "Broker did not answer an account read",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        return ComponentHealth(name=name, healthy=False, detail=str(exc)[:200])
    finally:
        await broker.close()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness. Must stay dependency-free and always cheap.

    This is what an orchestrator polls to decide whether to restart the
    process. If it touched the database, a database blip would trigger a
    restart loop that cannot possibly fix it.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.environment,
        live_trading_enabled=settings.live_trading_enabled,
    )


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(response: Response, db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Readiness: can this instance actually serve requests?

    Checks are concurrent — serially probing five components would make the
    endpoint's latency their sum, and it gets polled often.
    """
    settings = get_settings()

    components = await asyncio.gather(
        _check_database(db),
        _check_redis(),
        return_exceptions=False,
    )

    all_healthy = all(component.healthy for component in components)
    if not all_healthy:
        # 503 so a load balancer stops routing here, rather than a 200 with a
        # sad payload nobody reads.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if all_healthy else "degraded",
        version="0.1.0",
        environment=settings.environment,
        live_trading_enabled=settings.live_trading_enabled,
        components=list(components),
    )


@router.get("/health/dependencies", response_model=HealthResponse)
async def dependencies(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Full dependency sweep, including outbound integrations (§18).

    Separate from readiness because it makes real network calls to brokers and
    providers. Polling this on a health-check interval would burn provider
    budget and trip broker rate limits.
    """
    settings = get_settings()

    components = list(
        await asyncio.gather(
            _check_database(db),
            _check_redis(),
            _check_broker(BrokerKind.TRADING212_DEMO, "trading212_demo"),
            _check_broker(BrokerKind.TRADING212_LIVE, "trading212_live"),
        )
    )

    all_healthy = all(component.healthy for component in components)
    return HealthResponse(
        status="ok" if all_healthy else "degraded",
        version="0.1.0",
        environment=settings.environment,
        live_trading_enabled=settings.live_trading_enabled,
        components=components,
    )
