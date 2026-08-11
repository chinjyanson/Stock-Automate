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
from app.models_ml.logistic import FittedModel


@dataclass(frozen=True, slots=True)
class InsiderPressure:
    """Insider selling on one instrument, and whether it has been priced in yet.

    `move_atr` is the signed move since the filing, in ATR units. The sign is
    what an exit decision turns on: already fallen means the market has acted
    and selling now realises the loss at the bottom; flat or higher means the
    move has not happened and there is still time to leave.
    """

    sell_penalty: float
    move_atr: float | None


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


@dataclass(frozen=True)
class IndexConditions:
    """Index-level market state, resolved once per run by the engine.

    Everything here describes the market rather than any instrument in it, so it
    is read once and shared. `options_available` is separate from the fields
    being None because "no usable chain today" and "a chain that priced no
    skew" are different situations: the first means a strategy has nothing to
    act on, the second is a real reading that happens to be partial.
    """

    regime_factor: float = 1.0
    gamma_exposure: float | None = None
    #: Net dealer gamma as a fraction of gross, in [-1, +1]. **The figure a model
    #: reads**, because it means the same thing whichever proxy produced the row
    #: — open interest and the contract multiplier cancel out of a ratio, and
    #: `gamma_exposure` does not.
    gamma_tilt: float | None = None
    charm_exposure: float | None = None
    #: Net dealer charm as a fraction of gross. The model's figure, as above.
    charm_tilt: float | None = None
    skew_25delta: float | None = None
    atm_iv: float | None = None
    contracts_used: int = 0
    options_available: bool = False


@dataclass
class StrategyContext:
    """Everything a strategy pass needs, resolved once by the engine."""

    config: StrategyConfiguration
    store: CandleStore
    instruments: list[Instrument]
    positions: list[BrokerPosition]
    #: Index-level conditions (market regime, dealer gamma, skew). Resolved by
    #: the engine and passed in for the same reason as `insider_sell_pressure`:
    #: a strategy reads only what it is handed, so it stays testable against
    #: fixtures with no database behind them.
    index_conditions: IndexConditions = field(default_factory=IndexConditions)
    #: Insider selling pressure per instrument, 0..0.40 — the same figure the
    #: scanner uses as a score penalty. Resolved once by the engine and passed
    #: in, exactly like `positions`, so a strategy still reads only what is
    #: handed to it and stays testable against fixtures with no database.
    #: Absent key means no qualifying selling.
    insider_sell_pressure: dict[uuid.UUID, InsiderPressure] = field(default_factory=dict)
    #: Post-earnings drift per instrument, 0-100 with 50 neutral — below it means
    #: the market reacted badly to the last report and, per PEAD, is not finished
    #: repricing. Resolved once by the engine and passed in, exactly like
    #: `insider_sell_pressure`. An absent key means no live earnings event, which
    #: is the common case and reads as "no objection" rather than as bad news.
    pead_scores: dict[uuid.UUID, float] = field(default_factory=dict)
    #: Kronos forecast features per instrument, as `kronos_return`,
    #: `kronos_prob_up`, `kronos_dispersion`, `kronos_drawdown`. Resolved once
    #: by the engine from the predictions table — never generated here, because
    #: generating one imports torch. An absent key means no recent forecast,
    #: which reads as missing information rather than as bad news.
    kronos_predictions: dict[uuid.UUID, dict[str, float]] = field(default_factory=dict)
    #: The fitted model this strategy serves, loaded once per run. None means
    #: no model has been fitted yet, and a strategy with no model emits nothing
    #: rather than falling back to a default weighting.
    stock_model: FittedModel | None = None
    #: Instruments `series()` could not serve, and why. Populated as a side
    #: effect so strategies stay branch-free about it; the engine drains this
    #: into SKIPPED decisions after the pass, which is what makes a data outage
    #: visible instead of looking like a run that found no setup.
    insufficient_history: dict[uuid.UUID, str] = field(default_factory=dict)

    def sell_pressure(self, instrument_id: uuid.UUID) -> InsiderPressure | None:
        """Insider selling on one instrument, or None when there is none."""
        return self.insider_sell_pressure.get(instrument_id)

    def kronos_features(self, instrument_id: uuid.UUID) -> dict[str, float] | None:
        """Kronos features for one instrument, or None when there is no forecast."""
        return self.kronos_predictions.get(instrument_id)

    def pead_score(self, instrument_id: uuid.UUID) -> float | None:
        """Post-earnings drift on one instrument, or None when there is no event."""
        return self.pead_scores.get(instrument_id)

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
