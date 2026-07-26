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


#: Floor below which no indicator in this codebase is meaningful, whatever a
#: strategy asks for. A strategy needing more says so via `series(required=...)`.
_MIN_SERIES_BARS = 20


@dataclass
class StrategyContext:
    """Everything a strategy pass needs, resolved once by the engine."""

    config: StrategyConfiguration
    store: CandleStore
    instruments: list[Instrument]
    positions: list[BrokerPosition]
    #: Instruments `series()` could not serve, and why. Populated as a side
    #: effect so strategies stay branch-free about it; the engine drains this
    #: into SKIPPED decisions after the pass, which is what makes a data outage
    #: visible instead of looking like a run that found no setup.
    insufficient_history: dict[uuid.UUID, str] = field(default_factory=dict)

    def held_quantity(self, instrument_id: uuid.UUID) -> Decimal:
        """How much of `instrument_id` the paper venue currently holds."""
        ticker = str(instrument_id)
        return sum(
            (Decimal(p.quantity) for p in self.positions if p.broker_ticker == ticker),
            start=Decimal(0),
        )

    async def series(
        self,
        instrument_id: uuid.UUID,
        interval: Interval,
        *,
        limit: int = 250,
        required: int = _MIN_SERIES_BARS,
    ) -> PriceSeries | None:
        """Closed candles for an instrument at `interval`, as a `PriceSeries`.

        `required` is the strategy's own minimum — the longest lookback it will
        index into. Passing it here rather than re-checking `series.length`
        afterwards keeps the shortfall recordable: a strategy that discards a
        too-short series itself leaves no trace of having done so.

        Returns None when the store cannot meet it, recording the shortfall in
        `insufficient_history`. The strategy then produces no signal for that
        instrument (fail closed).
        """
        floor = max(required, _MIN_SERIES_BARS)
        candles = await self.store.get_candles(
            instrument_id, interval, limit=max(limit, floor), closed_only=True
        )
        if len(candles) < floor:
            self.insufficient_history[instrument_id] = (
                f"{len(candles)} closed {interval.value} bars available, needs {floor}"
            )
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
