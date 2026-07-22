"""The Strategy interface and its evaluation context (§8).

A strategy reads candles from the local store and returns `StrategySignal`s. That
is all it does — sizing, risk, and execution live downstream, so a strategy stays
pure and unit-testable against fixture candles, with no broker or database mock.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.broker.types import BrokerPosition
from app.data.store import CandleStore
from app.indicators.series import PriceSeries, candles_to_series
from app.models.enums import Interval, OrderSide, StrategyKind
from app.models.instrument import Instrument
from app.models.strategy import StrategyConfiguration


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """One strategy's intent for one instrument.

    A signal is not an order. `conviction` (0..1) lets the engine and the UI rank
    signals; it does not size the trade — the risk engine does that.
    """

    instrument_id: uuid.UUID
    side: OrderSide
    conviction: float
    reason: str
    #: Target quantity for an exit/trim (SELL). None means "the natural amount":
    #: close the position for an exit, or let sizing decide for an entry.
    target_quantity: Decimal | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyContext:
    """Everything a strategy pass needs, resolved once by the engine."""

    config: StrategyConfiguration
    store: CandleStore
    instruments: list[Instrument]
    positions: list[BrokerPosition]

    def held_quantity(self, instrument_id: uuid.UUID) -> Decimal:
        """How much of `instrument_id` the paper venue currently holds."""
        ticker = str(instrument_id)
        return sum(
            (Decimal(p.quantity) for p in self.positions if p.broker_ticker == ticker),
            start=Decimal(0),
        )

    async def series(
        self, instrument_id: uuid.UUID, interval: Interval, *, limit: int = 250
    ) -> PriceSeries | None:
        """Closed candles for an instrument at `interval`, as a `PriceSeries`.

        None when there is not enough history to form even a short series — the
        strategy then simply produces no signal for that instrument (fail closed).
        """
        candles = await self.store.get_candles(
            instrument_id, interval, limit=limit, closed_only=True
        )
        if len(candles) < 20:
            return None
        return candles_to_series(candles)


class Strategy(ABC):
    """A configured, reproducible opinion about when to trade (§8)."""

    kind: StrategyKind
    #: The bar size this strategy reads. The engine seeds `config.interval` from
    #: it, but the config wins if an operator overrides it.
    interval: Interval = Interval.D1

    def __init__(self, config: StrategyConfiguration) -> None:
        self.config = config
        self.params: dict[str, Any] = dict(config.params or {})

    def param(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    @property
    def read_interval(self) -> Interval:
        return self.config.interval or self.interval

    @abstractmethod
    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        """Return the signals this strategy produces for the current data."""
