"""Broker integrations.

Nothing outside this package may import a concrete adapter or a broker's wire
format. Consumers take `Broker` and the DTOs from `app.broker.types` (§3).
"""

from app.broker.base import Broker
from app.broker.factory import (
    LiveTradingDisabledError,
    default_paper_broker_kind,
    resolve_broker,
)
from app.broker.mock import MockBroker
from app.broker.trading212 import Trading212DemoBroker, Trading212LiveBroker
from app.broker.types import (
    BrokerAccount,
    BrokerAmbiguousResponseError,
    BrokerAuthError,
    BrokerError,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRejectedError,
    BrokerOrderRequest,
    BrokerPosition,
    BrokerRateLimitError,
    BrokerUnavailableError,
    ReconciliationDiscrepancy,
    ReconciliationResult,
)

__all__ = [
    "Broker",
    "BrokerAccount",
    "BrokerAmbiguousResponseError",
    "BrokerAuthError",
    "BrokerError",
    "BrokerInstrument",
    "BrokerOrder",
    "BrokerOrderRejectedError",
    "BrokerOrderRequest",
    "BrokerPosition",
    "BrokerRateLimitError",
    "BrokerUnavailableError",
    "LiveTradingDisabledError",
    "MockBroker",
    "ReconciliationDiscrepancy",
    "ReconciliationResult",
    "Trading212DemoBroker",
    "Trading212LiveBroker",
    "default_paper_broker_kind",
    "resolve_broker",
]
