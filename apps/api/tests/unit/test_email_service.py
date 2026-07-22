"""Unit tests for the Brevo transactional email service.

No network: `respx` serves the Brevo endpoint, and the unconfigured path is
asserted to make no HTTP call at all.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import SecretStr

from app.config import Settings
from app.services.email import _BREVO_SEND_URL, BrevoEmailService


def _settings(*, key: str | None) -> Settings:
    return Settings(
        brevo_api_key=SecretStr(key) if key else None,
        eod_digest_from_email="from@example.com",
        eod_digest_from_name="Test Sender",
    )


class TestBrevoEmailService:
    async def test_unconfigured_is_a_noop(self) -> None:
        service = BrevoEmailService(_settings(key=None))
        assert service.is_configured is False

        # respx with assert_all_mocked would raise if any request were made.
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(_BREVO_SEND_URL).mock(return_value=httpx.Response(201))
            sent = await service.send(
                to_email="to@example.com", to_name="Recipient", subject="s", html="<p>x</p>"
            )
        assert sent is False
        assert route.called is False

    @respx.mock
    async def test_configured_posts_to_brevo_with_key_header(self) -> None:
        route = respx.post(_BREVO_SEND_URL).mock(return_value=httpx.Response(201, json={"messageId": "1"}))
        service = BrevoEmailService(_settings(key="secret-key"))

        sent = await service.send(
            to_email="to@example.com",
            to_name="Recipient",
            subject="EOD summary",
            html="<p>hi</p>",
        )

        assert sent is True
        assert route.called
        request = route.calls.last.request
        assert request.headers["api-key"] == "secret-key"
        import json

        body = json.loads(request.content)
        assert body["sender"] == {"email": "from@example.com", "name": "Test Sender"}
        assert body["to"] == [{"email": "to@example.com", "name": "Recipient"}]
        assert body["subject"] == "EOD summary"
        assert body["htmlContent"] == "<p>hi</p>"

    @respx.mock
    async def test_rejection_returns_false_not_raise(self) -> None:
        respx.post(_BREVO_SEND_URL).mock(return_value=httpx.Response(400, json={"message": "bad"}))
        service = BrevoEmailService(_settings(key="secret-key"))
        sent = await service.send(
            to_email="to@example.com", to_name=None, subject="s", html="<p>x</p>"
        )
        assert sent is False

    @respx.mock
    async def test_transport_error_returns_false_not_raise(self) -> None:
        respx.post(_BREVO_SEND_URL).mock(side_effect=httpx.ConnectError("boom"))
        service = BrevoEmailService(_settings(key="secret-key"))
        sent = await service.send(
            to_email="to@example.com", to_name=None, subject="s", html="<p>x</p>"
        )
        assert sent is False
