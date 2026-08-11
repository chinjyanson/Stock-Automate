"""Individual stocks, entered on a fitted probability (§8).

Two layers, unchanged. The **scanner** decides *what* is worth owning; its
absolute, fundamentals-first score ranks the catalogue and the top names become
this strategy's universe. This decides only *when*.

**What changed is how "when" is answered.** The strategy this replaces summed
three hand-weighted readings of the same dislocation and compared the total to a
hand-chosen threshold. Every weight in it was picked by eye and swept one at a
time, and nine months of that produced no measurable edge. Here a logistic model
sets the weights instead, and returns something the old score never was: a
calibrated probability that *this trade reaches its target before its stop* —
the exact question the position is a bet on.

**The features are the ones that survived measurement**, not the ones that
sounded right. Ranked by Spearman IC against a 20-day forward return, per
instrument demeaned so they measure *when to buy* rather than *which stock*, and
required to hold their sign across an out-of-sample fold:

    discount_sma200   +0.096 / +0.084
    rsi_14            -0.084 / -0.071
    sma200_slope      -0.076 / -0.070
    atr_pct           +0.039 / +0.054

plus Kronos's forecast return, its probability of finishing up, and the spread
across its sampled paths — the last being the model's own uncertainty, which is
worth more than the point forecast because a confident wrong answer and an
unsure one are otherwise identical.

**What stayed a gate, and why.** The model answers "is this likely to work". It
does not answer "should this ever be bought", and blending those lets an
attractive enough setup buy its way past a safety rule. So two absolute gates
survive:

  * **ATR floor.** A stock whose true range is a rounding error has no snapback
    worth trading and would get a meaningless stop from the risk engine. This is
    tradeability, not prediction, and no probability should overrule it.
  * **Post-earnings drift.** A dip the market is still repricing is not a dip.
    It comes from the earnings table rather than from prices, so the feature set
    cannot see it.

The 200-day slope is deliberately *not* a gate any more. It is a feature now,
and having it in both places would count the same fact twice — once weighted by
the fit and once absolutely.

**Exits are untouched.** Insider selling still forces an early exit, the middle
Bollinger band is still the target, and the risk engine still owns the stop, the
size and the holding cap. Only the entry moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.backtest.features import MIN_BARS as FEATURE_MIN_BARS
from app.backtest.features import compute as compute_features
from app.indicators import functions as ind
from app.indicators.series import PriceSeries
from app.models.enums import Interval, OrderSide, StrategyKind
from app.models_ml.logistic import FittedModel
from app.strategies.base import Strategy, StrategyContext, StrategySignal

#: The features the model is fitted on, in no particular order — the stored
#: model carries its own ordering and this is only what the strategy offers it.
#: A name here that the model does not use is ignored; one the model wants and
#: this does not supply imputes to the training mean and contributes nothing.
PRICE_FEATURES = ("discount_sma200", "rsi_14", "sma200_slope", "atr_pct")

#: Bars fetched. `sma200_slope` needs 200 plus its 21-bar fit window, which is
#: far more than the Bollinger target asks for.
PREFERRED_BARS = 300

#: Below this `features.compute` returns nothing at all, so the model has no
#: input and the question is unanswerable. Set from the feature module rather
#: than guessed, so the two cannot drift: a lower number here would make the
#: engine silently evaluate names it can say nothing about instead of recording
#: them as skipped, and a data gap would look like a night that found no setup.
REQUIRED_BARS = FEATURE_MIN_BARS


@dataclass(frozen=True, slots=True)
class StockReading:
    """One instrument's model inputs and the probability they produce."""

    probability: float
    features: dict[str, float]
    contributions: dict[str, float]
    middle: float
    atr: float
    atr_pct: float
    close: float


def read_features(series: PriceSeries, kronos: dict[str, float] | None) -> dict[str, float]:
    """Point-in-time features for the last bar of `series`.

    Uses `backtest.features.compute` — the same function the fit ran on. That
    sharing is the whole safeguard against the failure this design is most
    exposed to: a model fitted on one definition of `rsi_14` and served on
    another produces plausible probabilities that are simply wrong, and nothing
    anywhere errors.

    The caller is responsible for passing only bars that had closed at the time
    being evaluated; this cannot check that.
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
    if kronos:
        out.update(kronos)
    return out


def read_stock(
    series: PriceSeries,
    model: FittedModel,
    kronos: dict[str, float] | None,
    *,
    bb_period: int,
    bb_std: float,
    atr_period: int,
) -> StockReading | None:
    """Evaluate the model and the exit target together, or None if unreadable.

    None means "no opinion is possible" — too few bars, a flat window with no
    band, an unusable price. That is a different answer from a low probability,
    and the caller treats it differently.
    """
    if series.length < REQUIRED_BARS:
        return None
    bands = ind.bollinger_bands(series.close, bb_period, bb_std)
    if bands is None:
        return None
    _, middle, _ = bands

    atr = ind.average_true_range(series.high, series.low, series.close, period=atr_period)
    last = float(series.close[-1])
    if atr is None or last <= 0:
        return None

    features = read_features(series, kronos)
    # No feature computed at all is "no opinion", not "the base rate".
    #
    # `backtest.features.compute` returns nothing below MIN_BARS, so a series
    # under ~220 bars yields an empty reading — and the model would then answer
    # `sigmoid(intercept)` for every such name, identically, having seen nothing
    # about any of them. If that base rate happened to sit above the entry
    # probability the strategy would buy every short-history instrument in the
    # universe on no information whatever. Missing *some* features imputes to
    # the training mean, as everywhere else; missing all of them means the
    # question is unanswerable.
    if not features:
        return None

    return StockReading(
        probability=model.probability(features),
        features=features,
        contributions=model.contributions(features),
        middle=middle,
        atr=atr,
        atr_pct=atr / last,
        close=last,
    )


class LogisticStockStrategy(Strategy):
    kind = StrategyKind.LOGISTIC_STOCK
    interval = Interval.D1

    async def evaluate(self, ctx: StrategyContext) -> list[StrategySignal]:
        model = ctx.stock_model
        if model is None:
            # No fitted model means no opinion — deliberately not a fallback to
            # some default weighting, which would be a different strategy
            # trading under this one's name.
            return []

        entry_probability = float(self.param("entry_probability", 0.55))
        min_atr_pct = float(self.param("min_atr_pct", 0.02))
        pead_veto_below = float(self.param("pead_veto_below", 40.0))
        insider_veto = float(self.param("insider_sell_veto", 0.10))
        insider_exit_max_drop = float(self.param("insider_exit_max_drop_atr", 1.0))
        bb_period = int(self.param("bb_period", 20))
        bb_std = float(self.param("bb_std", 2.0))
        atr_period = int(self.param("atr_period", 14))

        signals: list[StrategySignal] = []
        for instrument in ctx.instruments:
            series = await ctx.series(
                instrument.id,
                self.read_interval,
                limit=PREFERRED_BARS,
                required=REQUIRED_BARS,
            )
            if series is None:
                continue

            reading = read_stock(
                series,
                model,
                ctx.kronos_features(instrument.id),
                bb_period=bb_period,
                bb_std=bb_std,
                atr_period=atr_period,
            )
            if reading is None:
                continue

            held = ctx.held_quantity(instrument.id)
            pressure = ctx.sell_pressure(instrument.id)
            pead = ctx.pead_score(instrument.id)

            metrics = {
                "probability": reading.probability,
                "atr": reading.atr,
                "atr_pct": reading.atr_pct,
                "close": reading.close,
                "bb_middle": reading.middle,
                **{f"feature_{k}": v for k, v in reading.features.items()},
                **{f"logodds_{k}": v for k, v in reading.contributions.items()},
            }
            if pead is not None:
                metrics["pead_score"] = pead

            # Leave *before* the fall, or not at all. Unchanged from the
            # strategy this replaces: a chief officer choosing to sell precedes
            # about -6.28% excess return over the following month, so a position
            # held into that is worth closing early — but only while the market
            # has not already acted, since once the stock has dropped, selling
            # only realises the loss at the bottom. Signed move, not magnitude:
            # a stock that has *risen* is the best case to leave, not a damped
            # one.
            insider_exit = (
                held > 0
                and pressure is not None
                and pressure.sell_penalty >= insider_veto
                and (pressure.move_atr is None or pressure.move_atr > -insider_exit_max_drop)
            )

            if insider_exit:
                assert pressure is not None  # narrowed by insider_exit
                moved = (
                    "unknown"
                    if pressure.move_atr is None
                    else f"{pressure.move_atr:+.1f} ATR since filing"
                )
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=(
                            f"Insider exit: chief-officer selling "
                            f"(pressure {pressure.sell_penalty:.0%} >= {insider_veto:.0%}), "
                            f"price {moved} — leaving before the drop"
                        ),
                        target_quantity=held,
                        metrics={
                            **metrics,
                            "insider_sell_pressure": pressure.sell_penalty,
                            "insider_move_atr": pressure.move_atr or 0.0,
                        },
                    )
                )
            elif held <= 0:
                atr_ok = reading.atr_pct >= min_atr_pct
                pead_ok = pead is None or pead > pead_veto_below
                if reading.probability >= entry_probability and atr_ok and pead_ok:
                    signals.append(
                        StrategySignal(
                            instrument_id=instrument.id,
                            side=OrderSide.BUY,
                            # A calibrated probability is exactly what conviction
                            # was always meant to hold. Still not read by sizing
                            # — the risk engine works from ATR and equity alone.
                            conviction=reading.probability,
                            reason=(
                                f"Model probability {reading.probability:.0%} "
                                f"(>= {entry_probability:.0%}) that this reaches target "
                                f"before stop — {self._explain(reading)}; "
                                f"ATR {reading.atr_pct:.1%} of price "
                                f"(>= {min_atr_pct:.1%})"
                            ),
                            metrics=metrics,
                        )
                    )
            elif reading.close >= reading.middle:
                signals.append(
                    StrategySignal(
                        instrument_id=instrument.id,
                        side=OrderSide.SELL,
                        conviction=1.0,
                        reason=(
                            f"Exit: close {reading.close:.2f} recovered to the "
                            f"middle band {reading.middle:.2f}"
                        ),
                        target_quantity=held,
                        metrics=metrics,
                    )
                )
        return signals

    @staticmethod
    def _explain(reading: StockReading, top: int = 3) -> str:
        """The features that actually moved this probability, largest first.

        A probability with no account of where it came from is unreviewable, and
        these are the numbers a human would want when a trade goes wrong.
        """
        ranked = sorted(reading.contributions.items(), key=lambda kv: -abs(kv[1]))[:top]
        if not ranked:
            return "no features contributed"
        return ", ".join(f"{name} {value:+.2f} log-odds" for name, value in ranked)
