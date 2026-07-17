"""Trading 212 authentication header construction.

Trading 212 uses HTTP Basic auth over an API key and secret. Getting this wrong
produces a 401 with no other symptom, so the exact header is pinned here — a
regression to the old raw-key scheme would fail these rather than surface as a
mystery auth failure in production.
"""

from __future__ import annotations

import base64

import pytest

from app.broker.trading212 import Trading212DemoBroker, _Trading212Client
from app.broker.types import BrokerAuthError


def _auth_header(client: _Trading212Client) -> str:
    # The header is set once at construction on the underlying httpx client.
    return client._client.headers["Authorization"]


class TestBasicAuthHeader:
    def test_header_is_basic_base64_of_key_colon_secret(self) -> None:
        client = _Trading212Client("my-key", "my-secret", "https://demo.trading212.com/api/v0")
        expected = base64.b64encode(b"my-key:my-secret").decode("ascii")
        assert _auth_header(client) == f"Basic {expected}"

    def test_header_decodes_back_to_key_colon_secret(self) -> None:
        client = _Trading212Client("abc123", "shhh", "https://demo.trading212.com/api/v0")
        _, token = _auth_header(client).split(" ", 1)
        assert base64.b64decode(token).decode("ascii") == "abc123:shhh"

    def test_raw_key_is_never_sent_alone(self) -> None:
        """Guards against reverting to `Authorization: <key>`."""
        client = _Trading212Client("k", "s", "https://demo.trading212.com/api/v0")
        header = _auth_header(client)
        assert header.startswith("Basic ")
        assert header != "k"

    def test_no_trailing_newline_in_the_encoded_credential(self) -> None:
        """`base64.b64encode` over bytes never appends a newline.

        The shell `base64` command does, which is why the docs use `echo -n`.
        A stray newline in the credential would 401.
        """
        client = _Trading212Client("k", "s", "https://demo.trading212.com/api/v0")
        _, token = _auth_header(client).split(" ", 1)
        assert base64.b64decode(token) == b"k:s"


class TestCredentialValidation:
    def test_missing_secret_is_rejected_at_construction(self) -> None:
        with pytest.raises(BrokerAuthError, match="both required"):
            _Trading212Client("key-only", "", "https://demo.trading212.com/api/v0")

    def test_missing_key_is_rejected_at_construction(self) -> None:
        with pytest.raises(BrokerAuthError, match="both required"):
            _Trading212Client("", "secret-only", "https://demo.trading212.com/api/v0")

    def test_broker_forwards_both_halves_to_the_client(self) -> None:
        broker = Trading212DemoBroker(
            api_key="bk", api_secret="bs", base_url="https://demo.trading212.com/api/v0"
        )
        expected = base64.b64encode(b"bk:bs").decode("ascii")
        assert _auth_header(broker._client) == f"Basic {expected}"
