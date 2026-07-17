"""Broker selection.

Two rules govern this module, both from §17:

  1. Fallback is one-directional. Missing demo credentials degrade to the mock
     broker, because the cost is a less realistic practice environment. Missing
     *live* credentials raise. There is no path here from a paper intent to a
     live venue — `resolve_broker` cannot return a live adapter unless the
     caller explicitly asked for one and every gate passed.

  2. Live construction is guarded in depth. The server flag, the credential and
     the arming session are checked by different layers; this one owns the
     first two and refuses to build the adapter at all without them.
"""

from __future__ import annotations

import structlog

from app.broker.base import Broker
from app.broker.mock import MockBroker
from app.broker.trading212 import Trading212DemoBroker, Trading212LiveBroker
from app.broker.types import BrokerAuthError
from app.config import Settings, get_settings
from app.models.enums import BrokerKind

log = structlog.get_logger(__name__)


class LiveTradingDisabledError(Exception):
    """Raised when a live broker is requested but the server forbids it."""


def resolve_broker(kind: BrokerKind, settings: Settings | None = None) -> Broker:
    """Construct the adapter for `kind`.

    Raises rather than substituting whenever the requested venue cannot be
    honoured, so a caller asking for live and getting a broker back knows it is
    live.
    """
    settings = settings or get_settings()

    match kind:
        case BrokerKind.MOCK:
            return MockBroker()

        case BrokerKind.INTERNAL_PAPER:
            # The internal simulator lands with the risk engine in Phase 3; the
            # demo broker covers the paper path until then. Raising keeps the
            # gap visible instead of silently handing back a different venue.
            raise NotImplementedError(
                "InternalPaperBroker arrives in Phase 3. Use BrokerKind.TRADING212_DEMO "
                "or BrokerKind.MOCK for paper trading today."
            )

        case BrokerKind.TRADING212_DEMO:
            key = settings.trading212_demo_api_key
            if key is None or not key.get_secret_value():
                # Safe degradation: mock is strictly less privileged than demo.
                log.warning(
                    "broker.demo_credentials_absent_using_mock",
                    detail="TRADING212_DEMO_API_KEY is unset; using the deterministic "
                    "mock broker. No orders reach Trading 212.",
                )
                return MockBroker()
            return Trading212DemoBroker(
                api_key=key.get_secret_value(),
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
            key = settings.trading212_live_api_key
            if key is None or not key.get_secret_value():
                # Deliberately NOT falling back. A caller that asked for live
                # must never receive a simulator and believe it traded.
                raise BrokerAuthError(
                    "Live trading was requested but TRADING212_LIVE_API_KEY is not configured."
                )
            log.warning("broker.live_adapter_constructed", detail="Real-money venue selected")
            return Trading212LiveBroker(
                api_key=key.get_secret_value(),
                base_url=settings.trading212_live_base_url,
                timeout_seconds=settings.trading212_timeout_seconds,
                max_requests_per_minute=settings.trading212_max_requests_per_minute,
            )

    raise ValueError(f"Unknown broker kind: {kind}")


def default_paper_broker_kind(settings: Settings | None = None) -> BrokerKind:
    """The broker used when nothing more specific is requested.

    Never live, regardless of configuration (§7: "Default mode must be Trading
    212 demo or internal paper mode").
    """
    settings = settings or get_settings()
    key = settings.trading212_demo_api_key
    if key is not None and key.get_secret_value():
        return BrokerKind.TRADING212_DEMO
    return BrokerKind.MOCK
