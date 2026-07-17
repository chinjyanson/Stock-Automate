"""Secret encryption at rest and password hashing (§17).

Broker and provider API keys are encrypted with Fernet (AES-128-CBC +
HMAC-SHA256), keyed by a value derived from SECRETS_ENCRYPTION_KEY. Fernet is
chosen over raw AES because it is authenticated and misuse-resistant: it will
not let us accidentally ship an unauthenticated mode or reuse a nonce.

The key is *derived* rather than used directly so that any sufficiently random
SECRETS_ENCRYPTION_KEY works, whatever its encoding. `openssl rand -base64 32`
produces standard base64, which is not the URL-safe base64 Fernet demands;
forcing users to know that difference invites them to paste something Fernet
rejects at boot, or worse, to pick a shorter value that it accepts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings

#: Domain separation: the same master secret must never produce the same key
#: for two different purposes.
_ENCRYPTION_INFO = b"trading-platform:secret-encryption:v1"

#: Argon2id with parameters chosen for an interactive login on modest hardware.
#: Deliberately not the library defaults, which are tuned lower.
_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MiB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def _derive_fernet_key(master_secret: str) -> bytes:
    """Stretch an arbitrary secret into a Fernet key.

    HKDF-Expand with a fixed info string. There is no salt because the master
    secret is already required to be high-entropy (32 random bytes); the job
    here is encoding normalisation and domain separation, not password
    stretching.
    """
    derived = hmac.new(
        key=master_secret.encode("utf-8"),
        msg=_ENCRYPTION_INFO,
        digestmod=hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(derived)


def _fernet() -> Fernet:
    settings = get_settings()
    return Fernet(_derive_fernet_key(settings.secrets_encryption_key.get_secret_value()))


class SecretDecryptionError(Exception):
    """Ciphertext could not be decrypted — usually a rotated encryption key."""


def encrypt_secret(plaintext: str) -> bytes:
    """Encrypt a credential for storage."""
    if not plaintext:
        raise ValueError("Refusing to encrypt an empty secret")
    return _fernet().encrypt(plaintext.encode("utf-8"))


def decrypt_secret(ciphertext: bytes) -> str:
    """Decrypt a stored credential.

    Raises rather than returning a sentinel: a caller that received "" and
    treated it as "no key configured" would silently fall back, and §17 forbids
    silent fallback around credentials.
    """
    try:
        return _fernet().decrypt(ciphertext).decode("utf-8")
    except InvalidToken as exc:
        raise SecretDecryptionError(
            "Stored credential could not be decrypted. This usually means "
            "SECRETS_ENCRYPTION_KEY was rotated; re-enter the credential."
        ) from exc


def fingerprint_secret(plaintext: str) -> str:
    """Last four characters, for 'is this the key I think it is?' in the UI.

    Not a hash — a hash of a short key would be brute-forceable, and this is
    only ever shown to the key's owner.
    """
    return plaintext[-4:] if len(plaintext) >= 4 else "****"


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time-ish password check. Never raises on a wrong password."""
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash predates the current Argon2 parameters."""
    return _password_hasher.check_needs_rehash(password_hash)


def generate_session_token() -> str:
    """A new opaque session token. 32 bytes of CSPRNG entropy."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """SHA-256 of a session token, for storage.

    Plain SHA-256 rather than Argon2 is correct here: the token is already 256
    bits of uniform entropy, so there is no dictionary to defend against, and
    session lookup happens on every request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
