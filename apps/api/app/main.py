"""FastAPI application entrypoint."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import account, audit, auth, health, instruments, live
from app.config import get_settings
from app.db import dispose_engine
from app.observability.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(
        log_level=settings.log_level,
        json_output=settings.environment != "development",
    )

    log.info(
        "api.starting",
        environment=settings.environment,
        # Surfaced at boot on purpose: an operator should never have to guess
        # whether this process can place real orders.
        live_trading_enabled=settings.live_trading_enabled,
    )
    if settings.live_trading_enabled:
        log.warning(
            "api.live_trading_enabled",
            detail="LIVE_TRADING_ENABLED is true. Live orders are possible once a "
            "user arms a session.",
        )

    yield

    log.info("api.shutting_down")
    await dispose_engine()


app = FastAPI(
    title="Trading Platform API",
    description=(
        "Stock scanner and autonomous trading platform.\n\n"
        "Defaults to paper trading. Live trading is disabled unless explicitly "
        "enabled server-side and armed by a re-authenticated user.\n\n"
        "Nothing this API returns is investment advice, and no screening result "
        "asserts that an instrument is a good investment."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # Served under /docs in development; the frontend consumes the OpenAPI
    # schema to generate its typed client (§1).
    openapi_url="/openapi.json",
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    # Explicit origins, never "*": credentials must be allowed, and the two are
    # mutually exclusive for good reason.
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)


@app.middleware("http")
async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach a request id to logs and responses (§18).

    Honours an inbound X-Request-ID so a trace can span the web app and the API,
    and generates one otherwise.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    clear_request_context()
    bind_request_context(request_id=request_id)

    try:
        response = await call_next(request)
    except Exception:
        # Log with the request id bound, then re-raise for the handler below.
        log.exception("api.unhandled_exception", path=request.url.path, method=request.method)
        raise
    finally:
        clear_request_context()

    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return an opaque 500.

    Exception text can carry connection strings, SQL and occasionally
    credentials. It belongs in the logs, not in an HTTP response (§17).
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    log.exception("api.unhandled_exception", path=request.url.path, request_id=request_id)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(account.router)
app.include_router(instruments.router)
app.include_router(audit.router)
app.include_router(live.router)
