"""Timed exposure to an S&P tracker, on a fitted probability (§9).

This answers a question mean reversion cannot: "is this cheap relative to its
own range?" is meaningful for a stock and meaningless for an index fund, whose
price falling *is* the market falling rather than a dislocation within it. So
the decision here is exposure — be in or be out — and the model estimates the
probability that the next twenty days are positive.

**Some of its features cannot be backfilled, and that shapes everything.** An
option chain is published for today and for no other day: per-strike open
interest is simply gone once the session passes, and no provider sells it back.
So dealer gamma and charm have no history to fit on, and cannot acquire one
except by this system recording a row a night from the day it starts.

Waiting for that would mean no model for years. Instead they ship carried by
**priors** — market-structure theory with error bars — while every feature that
*does* have history is fitted normally. As rows accumulate the data takes over,
and `shrinkage` reports exactly how far that has got.

**Three safeguards, because a prior is an assumption wearing a number.**

  * `shrinkage` per coefficient says what fraction of it came from data. A
    number nobody measured must never be indistinguishable from one somebody
    did.
  * The **clamp** bounds the summed log-odds those coefficients may contribute
    (`FittedModel.low_shrinkage_clamp`). On 20-day non-overlapping windows a
    coefficient stays prior-driven for *years*, so "the data will fix it" is not
    a safeguard and magnitude has to be: a wrong prior can tilt the decision and
    can never drive it.
  * A feature whose **scale** was never learned is not served at all. The prior
    supplies a coefficient, but standardising a raw reading against a fabricated
    mean and standard deviation would produce a number with no meaning that
    still looks like one. Roughly sixty rows are needed before dealer gamma can
    be z-scored, at which point it switches itself on.

The regime factor still gates. A model that likes the market cannot overrule a
risk-off regime, for the same reason the stock model cannot buy past its ATR
floor: "is this likely to work" and "is this allowed" are different questions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.backtest.features import compute as compute_features
from app.indicators.series import PriceSeries
from app.models.enums import Interval, OrderSide, StrategyKind
from app.models_ml.logistic import FittedModel
from app.strategies.base import IndexConditions, Strategy, StrategyContext, StrategySignal

#: Price features, from the same module the fit runs on.
PRICE_FEATURES = (
    "sma50_over_sma200",
    "discount_sma200",
    "sma200_slope",
    "atr_pct",
    "volatility_20d",
)

#: Option-derived features. These are the ones carried by priors until the daily
#: job has accumulated enough rows to fit and to standardise them.
OPTION_FEATURES = ("gamma_tilt", "charm_tilt", "skew_25delta", "atm_iv")

PREFERRED_BARS = 400
REQUIRED_BARS = 260


@dataclass(frozen=True, slots=True)
class IndexReading:
    probability: float
    features: dict[str, float]
    contributions: dict[str, float]
    close: float


def read_index_features(
    series: PriceSeries,
    conditions: IndexConditions,
    kronos: dict[str, float] | None = None,
) -> dict[str, float]:
    """Point-in-time features for the last bar, plus today's option reading.

    The option fields are only offered when a chain was actually read today.
    `options_available` is separate from the fields being None on purpose: "no
    usable chain" and "a chain that priced no skew" are different situations,
    and only the second is a real reading that happens to be partial.
    """
    computed = compute_features(series.open, series.high, series.low, series.close, series.volume)
    out: dict[str, float] = {}
    for name in PRICE_FEATURES:
        column = computed.get(name)
        if column is None or column.size == 0:
            continue
        value = float(column[-1])
        if np.isfinite(value):
            out[name] = value

    if conditions.options_available:
        optional: tuple[tuple[str, float | None], ...] = (
            ("gamma_tilt", conditions.gamma_tilt),
            ("charm_tilt", conditions.charm_tilt),
            ("skew_25delta", conditions.skew_25delta),
            ("atm_iv", conditions.atm_iv),
        )
        for name, reading in optional:
            if reading is not None and np.isfinite(reading):
                out[name] = reading

    if kronos:
        out.update(kronos)
    return out


def read_index(
    series: PriceSeries,
    model: FittedModel,
    conditions: IndexConditions,
    kronos: dict[str, float] | None = None,
) -> IndexReading | None:
    """Evaluate the model, or None when no opinion is possible."""
    if series.length < REQUIRED_BARS:
        return None
    last = float(series.close[-1])
    if last <= 0:
        return None

    features = read_index_features(series, conditions, kronos)
    if not features:
        return None
    return IndexReading(
        probability=model.probability(features),
        features=features,
        contributions=model.contributions(features),
        close=last,
    )


class LogisticIndexStrategy(Strategy):
    kind = StrategyKind.LOGISTIC_INDEX
    interval = Interval.D1

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        model = ctx.stock_model
        if model is None:
            return []

        entry_probability = float(self.param("entry_probability", 0.55))
        exit_probability = float(self.param("exit_probability", 0.45))
        regime_floor = float(self.param("regime_floor", 0.5))

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            series = await ctx.series(
                instrument.id, self.read_interval, limit=PREFERRED_BARS, required=REQUIRED_BARS
            )
            if series is None:
                continue
            reading = read_index(
                series, model, ctx.index_conditions, ctx.kronos_features(instrument.id)
            )
            if reading is None:
                continue

            held = ctx.held_quantity(instrument.id)
            regime = ctx.index_conditions.regime_factor
            metrics = {
                "probability": reading.probability,
                "regime_factor": regime,
                "close": reading.close,
                "options_available": float(ctx.index_conditions.options_available),
                **{f"feature_{k}": v for k, v in reading.features.items()},
                **{f"logodds_{k}": v for k, v in reading.contributions.items()},
            }

            # A hysteresis band rather than one threshold. Exposure that flips on
            # a probability wobbling either side of a single number would trade
            # the noise in the estimate rather than the market, and each flip
            # costs a spread.
            if held <= 0:
                if reading.probability >= entry_probability and regime >= regime_floor:
                    signals.append(
                        StrategySignal(
                            instrument_id=instrument.id,
                            side=OrderSide.BUY,
                            conviction=reading.probability,
                            reason=(
                                f"Index model {reading.probability:.0%} "
                                f"(>= {entry_probability:.0%}) that the next 20 days are "
                                f"positive; regime {regime:.2f} (>= {regime_floor:.2f}) — "
                                f"{self._explain(reading)}"
                            ),
                            metrics=metrics,
                        )
                    )
            elif reading.probability < exit_probability or regime < regime_floor:
                cause = (
                    f"regime {regime:.2f} below {regime_floor:.2f}"
                    if regime < regime_floor
                    else f"model {reading.probability:.0%} below {exit_probability:.0%}"
                )
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=f"Index exit: {cause}",
                        target_quantity=held,
                        metrics=metrics,
                    )
                )
        return signals

    @staticmethod
    def _explain(reading: IndexReading, top: int = 3) -> str:
        ranked = sorted(reading.contributions.items(), key=lambda kv: -abs(kv[1]))[:top]
        if not ranked:
            return "no features contributed"
        return ", ".join(f"{name} {value:+.2f} log-odds" for name, value in ranked)
