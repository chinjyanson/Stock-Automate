"""Generate a VAPID keypair.

    uv --directory apps/api run python -m app.scripts.generate_vapid_keys

VAPID (RFC 8292) keys are an ECDSA P-256 keypair, encoded base64url without
padding. Nothing about them is web-push-library-specific, so this uses the
`cryptography` package the API already depends on rather than pulling in an npm
package for a subsystem whose sender is Python.

Encoding is the part that is easy to get subtly wrong:

  * The **public** key is the *uncompressed point* — `0x04 || X || Y`, 65 bytes.
    Not DER, not PEM, not the compressed form. Browsers reject anything else,
    and the failure surfaces as an opaque `applicationServerKey` error at
    subscribe time rather than anything that names the real problem.
  * The **private** key is the raw 32-byte scalar, not a PKCS#8 blob.
  * Both are base64**url** (`-` and `_`, not `+` and `/`) with `=` padding
    stripped, because these values travel in URLs and headers.
"""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric import ec

#: A P-256 public point is 0x04 || X(32) || Y(32).
_EXPECTED_PUBLIC_BYTES = 65
#: A P-256 private scalar is 32 bytes.
_EXPECTED_PRIVATE_BYTES = 32


def _b64url(raw: bytes) -> str:
    """base64url-encode without padding, per RFC 8292."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_vapid_keypair() -> tuple[str, str]:
    """Return (public_key, private_key), both base64url-encoded.

    The public key is safe to expose to the browser; the private key
    authenticates *us* to the push service and must stay server-side.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())

    public_numbers = private_key.public_key().public_numbers()
    # Fixed 32-byte big-endian coordinates. `to_bytes` with an explicit length
    # left-pads, which matters: a coordinate that happens to start with a zero
    # byte would otherwise encode short and produce an invalid 64-byte point
    # roughly one time in 256 — an intermittent bug that would be miserable to
    # track down later.
    x = public_numbers.x.to_bytes(32, "big")
    y = public_numbers.y.to_bytes(32, "big")
    public_bytes = b"\x04" + x + y

    private_bytes = private_key.private_numbers().private_value.to_bytes(32, "big")

    # Cheap assertions over a rare, silent encoding fault.
    assert len(public_bytes) == _EXPECTED_PUBLIC_BYTES, len(public_bytes)
    assert len(private_bytes) == _EXPECTED_PRIVATE_BYTES, len(private_bytes)

    return _b64url(public_bytes), _b64url(private_bytes)


def main() -> int:
    public_key, private_key = generate_vapid_keypair()

    print()
    print("VAPID keypair generated. Add these to your .env:")
    print()
    print(f"VAPID_PUBLIC_KEY={public_key}")
    print(f"VAPID_PRIVATE_KEY={private_key}")
    print(f"NEXT_PUBLIC_VAPID_PUBLIC_KEY={public_key}")
    print()
    print("VAPID_SUBJECT must also be set to a mailto: or https: URL you control.")
    print()
    print("The public key is safe to expose — it is sent to the browser, which is")
    print("why it appears twice (the NEXT_PUBLIC_ copy is the browser's).")
    print("The private key authenticates this server to the push service. Keep it")
    print("server-side, and treat a leak as you would a broker credential.")
    print()
    print("Note: push delivery is Phase 2 and is not implemented yet. These keys")
    print("are inert until then.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
