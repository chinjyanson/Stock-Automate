"""API error-handler classification.

The broker failure taxonomy is only useful if the HTTP layer preserves it. A
rate limit that surfaces as "could not be reached" (the bug this pins against)
sends the user to debug the wrong thing. These drive the real ASGI app with a
stubbed broker so the exception-handler dispatch is exercised end to end.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

import app.api.routes.account as account_routes
from app.auth.dependencies import AuthContext, get_auth_context
from app.broker.read_cache import broker_read_cache
from app.broker.types import (
    BrokerAuthError,
    BrokerError,
    BrokerRateLimitError,
    BrokerUnavailableError,
)
from app.main import app
from app.models.enums import BrokerKind
from app.models.user import User


class _StubBroker:
    """A broker whose reads raise a chosen error."""

    kind = BrokerKind.MOCK

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_positions(self) -> list[object]:
        raise self._error

    async def close(self) -> None:
        return None


@pytest_asyncio.fixture
async def client_with_broker_error() -> AsyncIterator[httpx.AsyncClient]:
    """ASGI client with auth bypassed; each test sets the broker's failure."""
    broker_read_cache.clear()

    fake_user = User(
        id=uuid.uuid4(),
        email="handler-test@example.com",
        password_hash="x",
        is_admin=True,
        is_active=True,
    )

    class _Session:
        pass

    app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user=fake_user,
        session=_Session(),  # type: ignore[arg-type]
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()
    broker_read_cache.clear()


def _stub(error: Exception) -> None:
    account_routes.resolve_broker = lambda kind, settings: _StubBroker(error)  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _restore_resolve_broker() -> AsyncIterator[None]:
    original = account_routes.resolve_broker
    yield
    account_routes.resolve_broker = original


class TestRateLimit:
    async def test_rate_limit_is_429_not_502(
        self, client_with_broker_error: httpx.AsyncClient
    ) -> None:
        _stub(BrokerRateLimitError("rate limit hit", retry_after_seconds=30))
        response = await client_with_broker_error.get("/positions?broker=mock")

        assert response.status_code == 429
        body = response.json()
        assert body["code"] == "broker_rate_limited"
        # The message must not claim the broker was unreachable — it answered.
        assert "reached" not in body["detail"].lower()
        assert "wait" in body["detail"].lower()

    async def test_retry_after_header_is_set(
        self, client_with_broker_error: httpx.AsyncClient
    ) -> None:
        _stub(BrokerRateLimitError("rate limit hit", retry_after_seconds=30))
        response = await client_with_broker_error.get("/positions?broker=mock")
        assert response.headers.get("retry-after") == "30"


class TestOtherBrokerErrors:
    async def test_auth_error_is_502_with_credential_message(
        self, client_with_broker_error: httpx.AsyncClient
    ) -> None:
        _stub(BrokerAuthError("401"))
        response = await client_with_broker_error.get("/positions?broker=mock")
        assert response.status_code == 502
        assert response.json()["code"] == "broker_credentials_rejected"

    async def test_generic_broker_error_is_502(
        self, client_with_broker_error: httpx.AsyncClient
    ) -> None:
        _stub(BrokerError("something odd"))
        response = await client_with_broker_error.get("/positions?broker=mock")
        assert response.status_code == 502
        assert response.json()["code"] == "broker_error"

    async def test_unavailable_is_treated_as_generic_broker_error(
        self, client_with_broker_error: httpx.AsyncClient
    ) -> None:
        # BrokerUnavailableError has no dedicated handler; "could not be reached"
        # is accurate for it, unlike for a rate limit.
        _stub(BrokerUnavailableError("connection refused"))
        response = await client_with_broker_error.get("/positions?broker=mock")
        assert response.status_code == 502
        assert response.json()["code"] == "broker_error"
