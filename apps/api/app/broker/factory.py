"""Broker selection.

Two rules govern this module:

  1. **A venue is never substituted.** If the requested broker cannot be
     honoured, this raises. There is no implicit fallback of any kind — a caller
     that asks for Trading 212 demo and receives a broker back knows it is
     talking to Trading 212, and a caller that asks for live knows it is live.

     The mock broker is still available, but only when *explicitly named*
     (`BrokerKind.MOCK`). Silently degrading to it on a missing key was worse
     than an error: the system would report filled orders and moving cash while
     talking to nothing, and the fixture prices are invented — a paper run
     against them proves nothing about Trading 212's tickers, rate limits or
     order semantics, which is the entire reason to use demo.

  2. **Live construction is guarded in depth.** The server flag, the credential
     and the arming session are checked by different layers; this one owns the
     first two and refuses to build the adapter at all without them.
"""

from __future__ import annotations

import structlog
from pydantic import SecretStr

from app.broker.base import Broker
from app.broker.mock import MockBroker
from app.broker.trading212 import Trading212DemoBroker, Trading212LiveBroker
from app.config import Settings, get_settings
from app.models.enums import BrokerKind

log = structlog.get_logger(__name__)


def _secret_value(secret: SecretStr | None) -> str:
    """Unwrap a SecretStr to its value, or empty string when unset."""
    return secret.get_secret_value() if secret else ""


def _missing_credentials(key: str, key_name: str, secret: str, secret_name: str) -> str | None:
    """Name the absent half (or both) of a key/secret pair, or None if complete.

    Both are required for Basic auth, and reporting exactly which is missing is
    the difference between a user fixing it in one step and guessing.
    """
    missing = [name for value, name in ((key, key_name), (secret, secret_name)) if not value]
    if not missing:
        return None
    return " and ".join(missing) + (" is" if len(missing) == 1 else " are")


class LiveTradingDisabledError(Exception):
    """Raised when a live broker is requested but the server forbids it."""


class BrokerNotConfiguredError(Exception):
    """A broker was requested whose credentials are absent.

    Distinct from `BrokerAuthError`, which means credentials were *present and
    rejected*. The remedy differs: this one is a configuration gap, that one is
    a bad or revoked key.
    """


def resolve_broker(kind: BrokerKind, settings: Settings | None = None) -> Broker:
    """Construct the adapter for `kind`, or raise.

    Never returns a different venue than the one asked for.
    """
    settings = settings or get_settings()

    match kind:
        case BrokerKind.MOCK:
            # Reachable only by explicitly asking for it. Nothing falls back here.
            return MockBroker()

        case BrokerKind.INTERNAL_PAPER:
            # The internal simulator lands with the risk engine in Phase 3.
            # Raising keeps the gap visible instead of handing back another venue.
            raise NotImplementedError(
                "InternalPaperBroker arrives in Phase 3. Use BrokerKind.TRADING212_DEMO "
                "for paper trading, or BrokerKind.MOCK for offline tests."
            )

        case BrokerKind.TRADING212_DEMO:
            key = _secret_value(settings.trading212_demo_api_key)
            secret = _secret_value(settings.trading212_demo_api_secret)
            missing = _missing_credentials(
                key, "TRADING212_DEMO_API_KEY", secret, "TRADING212_DEMO_API_SECRET"
            )
            if missing:
                raise BrokerNotConfiguredError(
                    f"Trading 212 demo was requested but {missing} not configured. "
                    "Generate a key on a Practice account (Settings -> API (Beta)); it "
                    "yields both a key and a secret, and both are required. Set them in "
                    ".env — see docs/trading212-setup.md. To run offline against "
                    "deterministic fixtures instead, request BrokerKind.MOCK explicitly."
                )
            return Trading212DemoBroker(
                api_key=key,
                api_secret=secret,
                base_url=settings.trading212_demo_base_url,
                timeout_seconds=settings.trading212_timeout_seconds,
                max_requests_per_minute=settings.trading212_max_requests_per_minute,
            )

        case BrokerKind.TRADING212_LIVE:
            if not settings.live_trading_enabled:
                raise LiveTradingDisabledError(
                    "Live trading is disabled on this server. Set LIVE_TRADING_ENABLED=true "
                    "and restart to permit live mode. This alone does not arm live trading."
                )
            key = _secret_value(settings.trading212_live_api_key)
            secret = _secret_value(settings.trading212_live_api_secret)
            missing = _missing_credentials(
                key, "TRADING212_LIVE_API_KEY", secret, "TRADING212_LIVE_API_SECRET"
            )
            if missing:
                raise BrokerNotConfiguredError(
                    f"Live trading was requested but {missing} not configured."
                )
            log.warning("broker.live_adapter_constructed", detail="Real-money venue selected")
            return Trading212LiveBroker(
                api_key=key,
                api_secret=secret,
                base_url=settings.trading212_live_base_url,
                timeout_seconds=settings.trading212_timeout_seconds,
                max_requests_per_minute=settings.trading212_max_requests_per_minute,
            )

    raise ValueError(f"Unknown broker kind: {kind}")


def default_paper_broker_kind(settings: Settings | None = None) -> BrokerKind:
    """The broker used when a caller does not name one.

    Always Trading 212 demo, and never live (§7: "Default mode must be Trading
    212 demo or internal paper mode"). If demo credentials are absent, the
    caller gets an error from `resolve_broker` rather than a mock — the default
    must not quietly become a different venue.

    Note this returns a *kind* and does not check credentials: the check belongs
    to `resolve_broker`, so there is exactly one place that decides whether a
    venue can be honoured.
    """
    _ = settings  # Retained for signature stability; selection is unconditional.
    return BrokerKind.TRADING212_DEMO


__all__ = [
    "BrokerNotConfiguredError",
    "LiveTradingDisabledError",
    "default_paper_broker_kind",
    "resolve_broker",
]
