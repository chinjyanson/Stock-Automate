"""What the replay needs to know about a bar, and the two ways of deciding it.

The replay models execution — next-open fills, intrabar stops, gaps filling at
the open — and that is worth keeping whatever decides the entries. So the two
are separated here: `replay` walks the bars and models the fills, and an
`EntryReader` says whether this bar admits a trade and where its target sits.

Two readers, for the two halves of building a model.

**`EveryBarReader` labels the training set.** It admits every bar it can read.
That is not a strategy — it is how an unbiased sample of outcomes is obtained.
Labelling only the entries some existing rule liked would teach the model to
discriminate *within* that rule's selection and tell it nothing about what the
rule was already refusing, which is most of the space and exactly where a better
entry would have to come from.

**`ModelReader` measures the fitted model.** It admits when the model's
probability clears a threshold, so a sweep over that threshold is a sweep over
the real strategy.

Both compute the stop and target identically, and identically to the live
strategy: the stop is a multiple of ATR and the target is the middle Bollinger
band. A replay whose exits differed from the deployed ones would be measuring a
strategy nobody runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.backtest.features import compute
from app.indicators import functions as ind
from app.indicators.functions import FloatArray
from app.indicators.series import PriceSeries
from app.models_ml.logistic import FittedModel

#: Multiple of ATR the stop sits below entry. Mirrors the live risk engine and
#: `ReplayConfig.atr_stop_multiplier`; all three describe one quantity.
DEFAULT_ATR_STOP_MULTIPLIER = 5.0

DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_ATR_PERIOD = 14


@dataclass(frozen=True, slots=True)
class BarReading:
    """Everything the replay needs from one bar, whoever decided it."""

    atr: float
    #: The level a profitable exit is taken at — the middle band.
    target: float
    #: The reader's own conviction, recorded on the trade for later analysis.
    score: float
    #: Distance to target over distance to stop, at entry.
    reward_risk: float
    #: Whether this bar admits opening a position.
    admits: bool


#: Called on `series.head(i + 1)` — bars up to and including `i`, never after.
#: That slicing is the replay's only guard against look-ahead, and it is done in
#: one place so a reader cannot accidentally undo it.
EntryReader = Callable[[PriceSeries], BarReading | None]


def _base_reading(
    series: PriceSeries,
    *,
    bb_period: int,
    bb_std: float,
    atr_period: int,
    atr_stop_multiplier: float,
) -> tuple[float, float, float] | None:
    """(atr, target, reward_risk) — the parts every reader shares.

    None when no opinion is possible: too few bars, a flat window with no
    meaningful band, an unusable price. Distinct from a reading that declines.
    """
    bands = ind.bollinger_bands(series.close, bb_period, bb_std)
    if bands is None:
        return None
    _, middle, _ = bands

    atr = ind.average_true_range(series.high, series.low, series.close, period=atr_period)
    last = float(series.close[-1])
    if atr is None or atr <= 0 or last <= 0:
        return None

    stop_distance = atr * atr_stop_multiplier
    reward_risk = (middle - last) / stop_distance if stop_distance > 0 else 0.0
    return atr, middle, reward_risk


@dataclass(frozen=True, slots=True)
class EveryBarReader:
    """Admits every readable bar. Used to label a training set, never to trade.

    Deliberately has no threshold and no gates. Its output is the unconditional
    answer to "what happens if a position is opened here?", across the whole
    space of bars — which is what a model needs to learn where the good entries
    are rather than merely to rank the ones something else already chose.
    """

    bb_period: int = DEFAULT_BB_PERIOD
    bb_std: float = DEFAULT_BB_STD
    atr_period: int = DEFAULT_ATR_PERIOD
    atr_stop_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER

    @property
    def required_bars(self) -> int:
        return max(self.bb_period, self.atr_period + 1)

    @property
    def preferred_bars(self) -> int:
        """Enough for the 200-day slope and its 21-bar fit window."""
        return 300

    def __call__(self, series: PriceSeries) -> BarReading | None:
        base = _base_reading(
            series,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            atr_period=self.atr_period,
            atr_stop_multiplier=self.atr_stop_multiplier,
        )
        if base is None:
            return None
        atr, target, reward_risk = base
        return BarReading(atr=atr, target=target, score=0.0, reward_risk=reward_risk, admits=True)


@dataclass
class ModelReader:
    """Admits when the fitted model's probability clears `threshold`.

    The same features the live strategy reads, on the same fitted model, so a
    swept threshold here is a swept threshold there. Kronos is absent from a
    historical replay — generating a forecast at every bar would take weeks — so
    its features impute to their training means, which is the same thing that
    happens live on a night the forecasting job did not run.

    **Stateful, unlike `EveryBarReader`.** `prepare` computes the feature matrix
    once per instrument; see it for why that is both necessary and safe.
    """

    model: FittedModel
    threshold: float = 0.55
    min_atr_pct: float = 0.02
    bb_period: int = DEFAULT_BB_PERIOD
    bb_std: float = DEFAULT_BB_STD
    atr_period: int = DEFAULT_ATR_PERIOD
    atr_stop_multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER
    _columns: dict[str, FloatArray] = field(default_factory=dict, repr=False)

    @property
    def required_bars(self) -> int:
        return max(self.bb_period, self.atr_period + 1)

    @property
    def preferred_bars(self) -> int:
        return 300

    def prepare(self, series: PriceSeries) -> None:
        """Compute the feature matrix once, for the whole series.

        `replay` walks a series by calling its reader with `head(start)`,
        `head(start + 1)`, and so on. Recomputing every feature column on each
        of those calls is quadratic in the series length — about 1.4M column
        evaluations for a 1,200-bar instrument, per threshold — which turns a
        routine sweep into twenty minutes.

        Computing once over the full series and indexing at `length - 1` gives
        **identical** numbers, because every feature is point-in-time: bar `i`'s
        value depends on bars up to `i` and none after it. That is asserted
        rather than assumed — `TestTrainServeIdentity` pins that the two paths
        agree to floating point, and `test_rewriting_the_future_cannot_change
        _the_present` pins the property they depend on. If that ever stopped
        holding, this would be wrong and so would the live strategy.

        Optional: a reader without this method is simply called per bar.
        """
        self._columns = compute(series.open, series.high, series.low, series.close, series.volume)

    def __call__(self, series: PriceSeries) -> BarReading | None:
        base = _base_reading(
            series,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            atr_period=self.atr_period,
            atr_stop_multiplier=self.atr_stop_multiplier,
        )
        if base is None:
            return None
        atr, target, reward_risk = base

        probability = self.model.probability(self._features(series))
        atr_ok = atr / float(series.close[-1]) >= self.min_atr_pct
        return BarReading(
            atr=atr,
            target=target,
            score=probability,
            reward_risk=reward_risk,
            admits=probability >= self.threshold and atr_ok,
        )

    def _features(self, series: PriceSeries) -> dict[str, float]:
        from app.strategies.logistic_stock import PRICE_FEATURES, read_features

        # Un-prepared — a direct call rather than a replay. Correct either way;
        # only the cost differs.
        if not self._columns:
            return read_features(series, None)

        index = series.length - 1
        out: dict[str, float] = {}
        for name in PRICE_FEATURES:
            column = self._columns.get(name)
            if column is None or index >= column.size:
                continue
            value = float(column[index])
            if np.isfinite(value):
                out[name] = value
        return out
