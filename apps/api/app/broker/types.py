"""Provider-neutral broker DTOs.

These are the only broker shapes permitted outside `app.broker`. Trading 212's
JSON never escapes the adapter (§3): if it did, every consumer would quietly
grow a dependency on one broker's field names and the interface would be
decorative.

All DTOs are frozen. A broker response is an observation of a past fact; code
that wants to change one has misunderstood something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from app.models.enums import BrokerKind, OrderSide, OrderStatus, OrderType


@dataclass(frozen=True, slots=True)
class BrokerInstrument:
    """An instrument as the broker's catalogue describes it."""

    broker_ticker: str
    name: str
    isin: str | None
    currency: str | None
    #: Exchange MIC where derivable; brokers often report their own venue codes.
    exchange_mic: str | None
    kind: str
    is_currently_available: bool
    min_quantity: Decimal | None = None
    quantity_step: Decimal | None = None
    supports_fractional: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BrokerAccount:
    account_id: str
    currency: str
    cash: Decimal
    #: Total account value including open positions.
    total: Decimal
    free_for_trading: Decimal
    invested: Decimal | None = None
    result: Decimal | None = None
    blocked: Decimal | None = None
    retrieved_at: datetime | None = None

    @property
    def masked_account_id(self) -> str:
        """Account identifiers are masked in logs and UI where not essential (§17)."""
        if len(self.account_id) <= 4:
            return "****"
        return f"****{self.account_id[-4:]}"


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    broker_ticker: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal | None
    currency: str | None = None
    unrealised_pnl: Decimal | None = None
    initial_fill_at: datetime | None = None
    max_buy: Decimal | None = None
    max_sell: Decimal | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    broker_order_id: str
    broker_ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    status: OrderStatus
    filled_quantity: Decimal = Decimal(0)
    average_fill_price: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    #: Our correlation token echoed back where the broker supports it. Trading
    #: 212 does not, which is precisely why local trade intents exist (§10).
    client_reference: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


@dataclass(frozen=True, slots=True)
class BrokerOrderRequest:
    broker_ticker: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "DAY"
    #: Local trade-intent id. Carried for our own reconciliation matching even
    #: when the broker will not store it.
    client_reference: str | None = None

    def validate(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires limit_price")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and self.stop_price is None:
            raise ValueError(f"{self.order_type} requires stop_price")


@dataclass(frozen=True, slots=True)
class ReconciliationDiscrepancy:
    kind: str
    broker_ticker: str | None
    detail: str
    local_value: str | None = None
    broker_value: str | None = None


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Outcome of comparing local state against the broker's."""

    broker: BrokerKind
    reconciled_at: datetime
    positions_checked: int
    orders_checked: int
    discrepancies: list[ReconciliationDiscrepancy] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies


class BrokerError(Exception):
    """Base class for broker failures."""


class BrokerAuthError(BrokerError):
    """Credentials rejected. Never include the key in the message."""


class BrokerRateLimitError(BrokerError):
    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class BrokerUnavailableError(BrokerError):
    """Broker unreachable or returning 5xx. Callers must fail closed (§17)."""


class BrokerAmbiguousResponseError(BrokerError):
    """A submission whose outcome is unknown — timeout, or an unparseable reply.

    This is the dangerous case: the order may or may not exist. Callers must
    never retry on this. Escalate to reconciliation instead (§10).
    """


class BrokerOrderRejectedError(BrokerError):
    """The broker positively refused the order. Safe: nothing was placed."""
