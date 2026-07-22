"""Transactional email via Brevo (§13, §16).

The only outbound email today is the end-of-day digest. Delivery is best-effort
and deliberately *fail-soft*: a send that errors is logged and swallowed, never
raised, so a mail outage can never fail the job that produced the summary.

When `brevo_api_key` is unset the service is a logged no-op — the same "quiet
when unconfigured" contract the market-data providers follow. This lets the
platform run end-to-end without mail credentials, and start sending the moment a
key is added, with no code change.
"""

from __future__ import annotations

import structlog
import httpx

from app.config import Settings, get_settings

log = structlog.get_logger(__name__)

#: Brevo's transactional email endpoint.
_BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"


class BrevoEmailService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        return self._settings.brevo_api_key is not None

    async def send(
        self,
        *,
        to_email: str,
        to_name: str | None,
        subject: str,
        html: str,
    ) -> bool:
        """Send one transactional email. Returns True only on a 2xx from Brevo.

        Never raises: a delivery failure must not propagate into the caller (the
        EOD job), which has already done its real work by the time we send.
        """
        if self._settings.brevo_api_key is None:
            log.info("email.skipped", reason="brevo_api_key not set", to=to_email)
            return False

        payload = {
            "sender": {
                "email": self._settings.eod_digest_from_email,
                "name": self._settings.eod_digest_from_name,
            },
            "to": [{"email": to_email, **({"name": to_name} if to_name else {})}],
            "subject": subject,
            "htmlContent": html,
        }
        headers = {
            "api-key": self._settings.brevo_api_key.get_secret_value(),
            "accept": "application/json",
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                response = await client.post(_BREVO_SEND_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            log.warning("email.send_failed", to=to_email, error=str(exc))
            return False

        if response.is_success:
            log.info("email.sent", to=to_email, subject=subject, status=response.status_code)
            return True

        # Brevo returns a JSON body with a message; log it but never surface it.
        log.warning(
            "email.rejected",
            to=to_email,
            status=response.status_code,
            body=response.text[:500],
        )
        return False
