"""The Broker interface (§3).

Strategies never see this type directly — orders reach a broker only after the
risk engine and the trade-intent guard have both cleared them (§9, §10). The
interface is deliberately small and free of Trading 212 concepts so that the
internal simulator is a first-class peer rather than a test double.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.broker.types import (
    BrokerAccount,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerPosition,
    ReconciliationResult,
)
from app.models.enums import BrokerKind


class Broker(ABC):
    """A venue we can query and, subject to authorisation, transact against."""

    #: Which concrete broker this is. Used for audit records and for the
    #: is-this-live checks that gate order submission.
    kind: BrokerKind

    @property
    def is_live(self) -> bool:
        """True only for real-money venues.

        Every call site that can move money consults this. It is a property of
        the adapter, not a configuration flag, so a demo adapter cannot be
        talked into claiming it is live (or the reverse).
        """
        return self.kind.is_live

    @abstractmethod
    async def sync_instruments(self) -> list[BrokerInstrument]:
        """Fetch the catalogue available to this account."""

    @abstractmethod
    async def get_account(self) -> BrokerAccount:
        """Fetch cash and account summary."""

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Fetch currently open positions."""

    @abstractmethod
    async def get_pending_orders(self) -> list[BrokerOrder]:
        """Fetch working (not yet terminal) orders."""

    @abstractmethod
    async def get_order_history(self) -> list[BrokerOrder]:
        """Fetch recent historical orders, used for ambiguity resolution."""

    @abstractmethod
    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrder:
        """Submit an order.

        Implementations MUST treat this as non-idempotent and MUST NOT retry
        internally. A timeout raises `BrokerAmbiguousResponseError`; deciding
        what happened is the caller's job, via reconciliation (§10).
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> None:
        """Request cancellation. Cancelling an already-terminal order is a no-op."""

    @abstractmethod
    async def reconcile(self) -> ReconciliationResult:
        """Compare broker truth against local records and report divergence."""

    async def process_stops(self) -> int:
        """Trigger resting stops the latest data has breached; return fills (§9).

        A real venue triggers its own stops server-side, so there is nothing for
        us to simulate — the default does nothing. The internal paper broker
        overrides this to fill breached stops against the local candle store.
        """
        return 0

    async def health_check(self) -> bool:
        """Cheap liveness probe (§18). Default: can we read the account?"""
        try:
            await self.get_account()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        """Release any held resources. Safe to call more than once."""
        return None
