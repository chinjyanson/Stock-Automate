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
from app.broker.factory import BrokerNotConfiguredError, LiveTradingDisabledError
from app.broker.types import BrokerAuthError, BrokerError
from app.config import get_settings
from app.data.factory import ProviderNotConfiguredError
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


@app.exception_handler(BrokerNotConfiguredError)
@app.exception_handler(ProviderNotConfiguredError)
async def not_configured_handler(request: Request, exc: Exception) -> JSONResponse:
    """A required credential is absent.

    503 rather than 500: the request was valid and the code is fine — the
    deployment is missing configuration. The message is safe to return verbatim
    because these exceptions are constructed by us and name environment
    variables, never values.

    This exists so a missing key surfaces as "set TRADING212_DEMO_API_KEY"
    rather than an opaque 500 the user has to read server logs to understand.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    log.warning(
        "api.not_configured",
        path=request.url.path,
        request_id=request_id,
        error=str(exc),
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "code": "not_configured",
            "request_id": request_id,
        },
    )


@app.exception_handler(BrokerAuthError)
async def broker_auth_handler(request: Request, exc: Exception) -> JSONResponse:
    """The broker rejected our credentials.

    Distinct from `not_configured`: a key is present, it is simply not accepted.
    502 because the failure is upstream, not in this request.

    The message deliberately does not echo the exception verbatim — broker error
    text occasionally reflects request context back, and this response is not a
    place to risk that. The specifics are in the logs.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    log.warning(
        "api.broker_credentials_rejected",
        path=request.url.path,
        request_id=request_id,
        error=str(exc),
    )
    return JSONResponse(
        status_code=502,
        content={
            "detail": (
                "Trading 212 rejected the configured API key. Check that the key is "
                "current and was generated on the matching account type — a key made "
                "on a Live account is not accepted by the demo endpoint, and vice "
                "versa. See docs/trading212-setup.md."
            ),
            "code": "broker_credentials_rejected",
            "request_id": request_id,
        },
    )


@app.exception_handler(BrokerError)
async def broker_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """The broker failed in some way we did not classify more specifically.

    Registered *after* the BrokerAuthError handler above. Starlette dispatches
    on the exception's MRO, so the most specific registered handler wins and a
    rejected credential still gets its own actionable message rather than this
    generic one.

    The exception text is not echoed: broker errors can reflect request context
    back at us, and this is not the place to gamble on that.
    """
    request_id = request.headers.get("X-Request-ID", "unknown")
    log.warning("api.broker_error", path=request.url.path, request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=502,
        content={
            "detail": "The broker could not be reached or returned an error.",
            "code": "broker_error",
            "request_id": request_id,
        },
    )


@app.exception_handler(LiveTradingDisabledError)
async def live_disabled_handler(request: Request, exc: Exception) -> JSONResponse:
    """Live trading was requested while the server forbids it."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    log.warning("api.live_trading_disabled", path=request.url.path, request_id=request_id)
    return JSONResponse(
        status_code=403,
        content={
            "detail": str(exc),
            "code": "live_trading_disabled",
            "request_id": request_id,
        },
    )


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
