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
from app.broker.factory import (
    BrokerNotConfiguredError,
    LiveTradingDisabledError,
    resolve_broker,
)
from app.broker.types import BrokerAuthError, BrokerRateLimitError, BrokerUnavailableError
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
    """Probe one broker, classifying *why* it is unhealthy.

    Deliberately calls `get_account()` rather than `Broker.health_check()`: the
    latter swallows every failure into a bare False, so the probe could only
    ever say "did not answer". Distinguishing a rejected credential from a
    network outage from a rate limit is the whole point of a dependency check —
    it is the difference between "fix your key" and "wait and retry".

    Not-configured and live-disabled are reported as healthy=True: a demo-only
    deployment is not broken because it has no live key, and a health check that
    went red for that would be noise that trains operators to ignore it.
    """
    settings = get_settings()
    started = time.perf_counter()

    try:
        broker = resolve_broker(kind, settings)
    except BrokerNotConfiguredError:
        return ComponentHealth(name=name, healthy=True, detail="not configured")
    except LiveTradingDisabledError:
        return ComponentHealth(name=name, healthy=True, detail="live trading disabled")
    except Exception as exc:
        return ComponentHealth(name=name, healthy=False, detail=str(exc)[:200])

    def _latency() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        account = await broker.get_account()
        return ComponentHealth(
            name=name,
            healthy=True,
            detail=f"authenticated — account {account.masked_account_id}",
            latency_ms=_latency(),
        )
    except BrokerAuthError:
        return ComponentHealth(
            name=name,
            healthy=False,
            detail="credentials rejected (401) — check the API key and secret",
            latency_ms=_latency(),
        )
    except BrokerRateLimitError:
        # Reachable and almost certainly authenticated — Trading 212 rate-limits
        # recognised keys. Not a failure state for a health check.
        return ComponentHealth(
            name=name,
            healthy=True,
            detail="rate limited, but reachable",
            latency_ms=_latency(),
        )
    except BrokerUnavailableError as exc:
        return ComponentHealth(
            name=name, healthy=False, detail=f"unreachable: {exc}"[:200], latency_ms=_latency()
        )
    except Exception as exc:
        return ComponentHealth(
            name=name, healthy=False, detail=str(exc)[:200], latency_ms=_latency()
        )
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
