"""VAPID keypair generation (RFC 8292).

The encoding is the whole risk here. A wrongly-encoded key does not fail at
generation — it fails much later, in a browser, as an opaque
`applicationServerKey` error that names none of the actual causes. These tests
pin the format so that failure cannot reach a browser.
"""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric import ec

from app.scripts.generate_vapid_keys import _b64url, generate_vapid_keypair


def _b64url_decode(value: str) -> bytes:
    """Decode base64url, restoring the padding the encoder stripped."""
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class TestEncoding:
    def test_padding_is_stripped(self) -> None:
        assert "=" not in _b64url(b"\x00" * 32)

    def test_uses_urlsafe_alphabet(self) -> None:
        """`+` and `/` would corrupt a key placed in a URL or header."""
        # This input encodes to "+/+/" under the standard alphabet.
        encoded = _b64url(bytes([0xFB, 0xFF, 0xBF]))
        assert "+" not in encoded
        assert "/" not in encoded
        assert encoded == "-_-_"

    def test_roundtrips(self) -> None:
        raw = bytes(range(65))
        assert _b64url_decode(_b64url(raw)) == raw


class TestKeypair:
    def test_public_key_is_an_uncompressed_p256_point(self) -> None:
        """0x04 || X(32) || Y(32) — not DER, not PEM, not compressed."""
        public_key, _ = generate_vapid_keypair()
        raw = _b64url_decode(public_key)

        assert len(raw) == 65
        assert raw[0] == 0x04

    def test_public_key_starts_with_capital_b(self) -> None:
        """A sanity check anyone can apply by eye.

        The leading 0x04 byte means the first base64url character is always "B".
        A key that does not start with "B" is not an uncompressed point.
        """
        public_key, _ = generate_vapid_keypair()
        assert public_key.startswith("B")

    def test_private_key_is_a_raw_32_byte_scalar(self) -> None:
        _, private_key = generate_vapid_keypair()
        assert len(_b64url_decode(private_key)) == 32

    def test_key_lengths_match_the_vapid_wire_format(self) -> None:
        public_key, private_key = generate_vapid_keypair()
        # 65 bytes -> 87 chars unpadded; 32 bytes -> 43 chars unpadded.
        assert len(public_key) == 87
        assert len(private_key) == 43

    def test_public_key_is_derived_from_the_private_key(self) -> None:
        """The pair must actually correspond — otherwise every push is rejected."""
        public_key, private_key = generate_vapid_keypair()

        scalar = int.from_bytes(_b64url_decode(private_key), "big")
        reconstructed = ec.derive_private_key(scalar, ec.SECP256R1())
        numbers = reconstructed.public_key().public_numbers()
        expected = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")

        assert _b64url_decode(public_key) == expected

    def test_keys_are_unique_per_call(self) -> None:
        # A generator that returned a fixed key would be catastrophic and is the
        # kind of thing a careless refactor could introduce.
        first, _ = generate_vapid_keypair()
        second, _ = generate_vapid_keypair()
        assert first != second

    def test_coordinates_are_left_padded_to_32_bytes(self) -> None:
        """Guards the once-in-256 short-encoding bug.

        A coordinate whose leading byte is zero must still encode as 32 bytes.
        Generating a batch makes an unpadded implementation fail reliably rather
        than intermittently.
        """
        for _ in range(64):
            public_key, private_key = generate_vapid_keypair()
            assert len(_b64url_decode(public_key)) == 65
            assert len(_b64url_decode(private_key)) == 32
