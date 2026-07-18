"""Scanner scoring (§6).

Turns a `PriceSeries` (and optional fundamentals) into a 100-point core score
across five categories, plus a *separate* optional fundamental score.

Two rules from §6 are non-negotiable and are the reason this module is written
the way it is:

  1. **Missing optional data never lowers the core score** (acceptance 7). Each
     sub-signal reports whether it could be computed; a signal that could not is
     dropped from both the numerator and the denominator of its category, so
     the category is scored on what is known, not penalised for what is not.
     Absence lowers *confidence*, not score.

  2. **No output asserts an instrument is a good investment** (§0). Scores are
     framed as "passes the configured screen" bands. The signal explanations use
     neutral, mechanical language ("price is above its 200-day average"), never
     recommendations.

Every threshold and weight is configurable; the constants here are only the
defaults §6 specifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.indicators import functions as ind
from app.indicators.series import PriceSeries
from app.models.scanner import Classification

# -- Default weights (§6). Sum to 100. --------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 25.0,
    "momentum": 20.0,
    "risk": 20.0,
    "liquidity": 20.0,
    "positioning": 15.0,
}

DEFAULT_THRESHOLDS: dict[str, float] = {"screening": 75.0, "watchlist": 60.0}

#: Preferred minimum history (§6). Scoring proceeds below this with reduced
#: confidence rather than refusing — a shorter series is still informative,
#: it is just less certain, and that uncertainty is reported.
PREFERRED_HISTORY_DAYS = ind.TRADING_DAYS_PER_YEAR


@dataclass(slots=True)
class SubSignal:
    """One measurable component of a category score.

    `available` is the whole point: an unavailable signal contributes nothing to
    either side of its category average, so missing data cannot drag a score
    down. `value` is the 0..1 normalised strength when available.
    """

    name: str
    available: bool
    value: float = 0.0
    #: Human-readable, mechanical description for the UI when this signal is
    #: strongly positive or negative.
    explanation: str = ""
    positive: bool = False


@dataclass(slots=True)
class CategoryScore:
    name: str
    points: float  # already weighted, i.e. out of the category's max
    max_points: float
    signals_available: int
    signals_total: int

    @property
    def coverage(self) -> float:
        return self.signals_available / self.signals_total if self.signals_total else 0.0


@dataclass(slots=True)
class ValueResult:
    """A separate valuation lens (0-100): how *cheap* the instrument looks.

    Deliberately independent of the momentum core score. The two answer different
    questions — "is this strong?" (momentum) and "is this cheap?" (value) — and
    an instrument can be high on one and low on the other. Neither is a buy
    signal on its own (§0); they are two inputs a human weighs.

    A high value score means the price is depressed relative to its own history
    and (where fundamentals exist) its earnings/book — i.e. potentially
    undervalued. A momentum screen and a value screen pointing the same way is
    the notable case; that is for the reader to judge, not the tool.
    """

    value_score: float
    price_value_score: float
    fundamental_value_score: float | None
    positive_signals: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreResult:
    core_score: float
    categories: dict[str, CategoryScore]
    fundamental_score: float | None
    classification: Classification
    data_completeness: float
    confidence: float
    candles_used: int
    #: The valuation lens, computed alongside momentum (never folded into it).
    value: ValueResult | None = None
    positive_signals: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _category_points(signals: list[SubSignal], max_points: float, name: str) -> CategoryScore:
    """Average the available signals and scale to the category's max.

    The average is over *available* signals only. A category with three of five
    signals available is scored on those three — never diluted toward zero by
    the two that could not be computed.
    """
    available = [s for s in signals if s.available]
    if not available:
        # Nothing to score. Award the neutral midpoint rather than zero: absence
        # of evidence is not evidence of a failing instrument (§6).
        return CategoryScore(
            name=name,
            points=max_points * 0.5,
            max_points=max_points,
            signals_available=0,
            signals_total=len(signals),
        )
    mean_strength = sum(s.value for s in available) / len(available)
    return CategoryScore(
        name=name,
        points=max_points * mean_strength,
        max_points=max_points,
        signals_available=len(available),
        signals_total=len(signals),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_series(
    series: PriceSeries,
    *,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    benchmark: PriceSeries | None = None,
    fundamentals: dict[str, Decimal | None] | None = None,
) -> ScoreResult:
    """Score one instrument's series. Pure — no I/O, fully deterministic."""
    weights = weights or DEFAULT_WEIGHTS
    thresholds = thresholds or DEFAULT_THRESHOLDS
    closes = series.preferred_close
    volumes = series.volume
    metrics: dict[str, Any] = {}

    trend = _score_trend(series, closes, metrics)
    momentum = _score_momentum(closes, benchmark, metrics)
    risk = _score_risk(closes, metrics)
    liquidity = _score_liquidity(closes, volumes, metrics)
    positioning = _score_positioning(closes, metrics)

    categories = {
        "trend": _category_points(trend, weights.get("trend", 25.0), "trend"),
        "momentum": _category_points(momentum, weights.get("momentum", 20.0), "momentum"),
        "risk": _category_points(risk, weights.get("risk", 20.0), "risk"),
        "liquidity": _category_points(liquidity, weights.get("liquidity", 20.0), "liquidity"),
        "positioning": _category_points(
            positioning, weights.get("positioning", 15.0), "positioning"
        ),
    }
    core_score = sum(c.points for c in categories.values())

    all_signals = trend + momentum + risk + liquidity + positioning
    available = sum(1 for s in all_signals if s.available)
    completeness = available / len(all_signals) if all_signals else 0.0

    # Confidence blends data completeness with history depth: a full signal set
    # over 40 days is less trustworthy than the same set over 300.
    history_factor = _clamp01(series.length / PREFERRED_HISTORY_DAYS)
    confidence = _clamp01(0.5 * completeness + 0.5 * history_factor)

    fundamental_score = _score_fundamentals(fundamentals) if fundamentals else None
    value = score_value(series, fundamentals=fundamentals)

    positives = [s.explanation for s in all_signals if s.available and s.positive and s.explanation]
    negatives = [
        s.explanation for s in all_signals if s.available and not s.positive and s.explanation
    ]
    missing = [s.name for s in all_signals if not s.available]

    return ScoreResult(
        core_score=round(core_score, 2),
        categories=categories,
        fundamental_score=round(fundamental_score, 2) if fundamental_score is not None else None,
        classification=classify(core_score, thresholds),
        data_completeness=round(completeness, 4),
        confidence=round(confidence, 4),
        candles_used=series.length,
        value=value,
        positive_signals=positives,
        negative_signals=negatives,
        missing_information=missing,
        metrics=metrics,
    )


def combine_primary_score(
    momentum_core: float,
    value_score: float | None,
    *,
    momentum_weight: float,
    value_weight: float,
) -> float:
    """Blend the momentum core and value scores into the primary ranking score.

    The primary score is what classification and default ordering use. A
    momentum-primary run (value_weight 0) returns the momentum core unchanged; a
    "buy low" run (value_weight high) makes value lead while momentum still
    contributes as a secondary term.

    A missing value score contributes as the neutral midpoint (50), so an
    instrument without a computable value reading is neither rewarded nor
    punished on the value axis. A degenerate all-zero weighting falls back to
    momentum rather than dividing by zero.
    """
    total = momentum_weight + value_weight
    if total <= 0:
        return momentum_core
    effective_value = value_score if value_score is not None else 50.0
    return (momentum_weight * momentum_core + value_weight * effective_value) / total


def classify(core_score: float, thresholds: dict[str, float] | None = None) -> Classification:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if core_score >= thresholds.get("screening", 75.0):
        return Classification.SCREENING_CANDIDATE
    if core_score >= thresholds.get("watchlist", 60.0):
        return Classification.WATCHLIST_CANDIDATE
    return Classification.DOES_NOT_PASS


# -- Category scorers -------------------------------------------------------
#
# Each returns a list of SubSignals. Normalisation maps a raw indicator to a
# 0..1 strength; the mappings are intentionally simple and monotonic so the
# score is explainable, not a black box.


def _score_trend(series: PriceSeries, closes: Any, metrics: dict[str, Any]) -> list[SubSignal]:
    price = float(closes[-1]) if closes.size else None
    sma50 = ind.simple_moving_average(closes, 50)
    sma200 = ind.simple_moving_average(closes, 200)
    slope = ind.sma_slope(closes, 200, slope_window=21)
    metrics.update({"sma50": sma50, "sma200": sma200, "sma200_slope": slope})

    signals: list[SubSignal] = []

    if price is not None and sma50 is not None:
        above = price > sma50
        signals.append(
            SubSignal(
                "price_above_sma50",
                True,
                1.0 if above else 0.0,
                "Price is above its 50-day average"
                if above
                else "Price is below its 50-day average",
                positive=above,
            )
        )
    else:
        signals.append(SubSignal("price_above_sma50", False))

    if price is not None and sma200 is not None:
        above = price > sma200
        signals.append(
            SubSignal(
                "price_above_sma200",
                True,
                1.0 if above else 0.0,
                "Price is above its 200-day average"
                if above
                else "Price is below its 200-day average",
                positive=above,
            )
        )
    else:
        signals.append(SubSignal("price_above_sma200", False))

    if sma50 is not None and sma200 is not None:
        golden = sma50 > sma200
        signals.append(
            SubSignal(
                "sma50_above_sma200",
                True,
                1.0 if golden else 0.0,
                "50-day average is above the 200-day average"
                if golden
                else "50-day average is below the 200-day average",
                positive=golden,
            )
        )
    else:
        signals.append(SubSignal("sma50_above_sma200", False))

    if slope is not None:
        # Normalise a fractional daily slope; ~0.1%/day maps to full strength.
        strength = _clamp01(0.5 + slope / 0.002)
        signals.append(
            SubSignal(
                "sma200_slope",
                True,
                strength,
                "Long-term trend is rising" if slope > 0 else "Long-term trend is falling",
                positive=slope > 0,
            )
        )
    else:
        signals.append(SubSignal("sma200_slope", False))

    return signals


def _score_momentum(
    closes: Any, benchmark: PriceSeries | None, metrics: dict[str, Any]
) -> list[SubSignal]:
    windows = {
        "1m": ind.TRADING_DAYS_PER_MONTH,
        "3m": ind.TRADING_DAYS_PER_MONTH * 3,
        "6m": ind.TRADING_DAYS_PER_MONTH * 6,
        "12m": ind.TRADING_DAYS_PER_YEAR,
    }
    signals: list[SubSignal] = []
    for label, days in windows.items():
        ret = ind.trailing_return(closes, days)
        metrics[f"return_{label}"] = ret
        if ret is None:
            signals.append(SubSignal(f"return_{label}", False))
            continue
        # A +20% move over the window maps to full strength; losses map toward 0.
        strength = _clamp01(0.5 + ret / 0.4)
        signals.append(
            SubSignal(
                f"return_{label}",
                True,
                strength,
                f"{label} return is {ret:+.1%}",
                positive=ret > 0,
            )
        )

    if benchmark is not None:
        rel = ind.relative_momentum(closes, benchmark.preferred_close, ind.TRADING_DAYS_PER_YEAR)
        metrics["relative_momentum_12m"] = rel
        if rel is None:
            signals.append(SubSignal("relative_momentum_12m", False))
        else:
            strength = _clamp01(0.5 + rel / 0.4)
            signals.append(
                SubSignal(
                    "relative_momentum_12m",
                    True,
                    strength,
                    f"12-month return vs benchmark is {rel:+.1%}",
                    positive=rel > 0,
                )
            )
    else:
        signals.append(SubSignal("relative_momentum_12m", False))

    return signals


def _score_risk(closes: Any, metrics: dict[str, Any]) -> list[SubSignal]:
    vol20 = ind.annualised_volatility(closes, 20)
    vol60 = ind.annualised_volatility(closes, 60)
    dd = ind.max_drawdown(closes, ind.TRADING_DAYS_PER_YEAR)
    downside = ind.downside_deviation(closes, 60)
    worst = ind.largest_daily_loss(closes, ind.TRADING_DAYS_PER_YEAR)
    metrics.update(
        {
            "volatility_20d": vol20,
            "volatility_60d": vol60,
            "max_drawdown_1y": dd,
            "downside_deviation_60d": downside,
            "largest_daily_loss_1y": worst,
        }
    )

    signals: list[SubSignal] = []

    # Lower risk scores higher. Each measure is inverted against a scale where
    # the "bad" end maps to 0.
    def _inverse(name: str, value: float | None, bad_at: float, label: str) -> SubSignal:
        if value is None:
            return SubSignal(name, False)
        strength = _clamp01(1.0 - value / bad_at)
        return SubSignal(
            name,
            True,
            strength,
            f"{label} is {value:.1%}",
            positive=strength >= 0.5,
        )

    signals.append(_inverse("volatility_20d", vol20, 0.60, "20-day volatility"))
    signals.append(_inverse("volatility_60d", vol60, 0.60, "60-day volatility"))
    signals.append(_inverse("max_drawdown_1y", dd, 0.50, "1-year max drawdown"))
    signals.append(_inverse("downside_deviation_60d", downside, 0.40, "downside deviation"))
    signals.append(_inverse("largest_daily_loss_1y", worst, 0.20, "largest 1-day loss"))
    return signals


def _score_liquidity(closes: Any, volumes: Any, metrics: dict[str, Any]) -> list[SubSignal]:
    avg_vol = ind.average_volume(volumes, 20)
    avg_value = ind.average_traded_value(closes, volumes, 20)
    zero_days = ind.zero_volume_days(volumes, 20)
    stale_days = ind.stale_price_days(closes, 20)
    metrics.update(
        {
            "avg_volume_20d": avg_vol,
            "avg_traded_value_20d": avg_value,
            "zero_volume_days_20d": zero_days,
            "stale_price_days_20d": stale_days,
        }
    )

    signals: list[SubSignal] = []

    if avg_value is not None:
        # Full strength at ~£1m/day traded value; scales down logarithmically.
        import math

        strength = _clamp01(math.log10(max(avg_value, 1.0)) / 6.0)
        signals.append(
            SubSignal(
                "avg_traded_value_20d",
                True,
                strength,
                f"20-day average traded value is {avg_value:,.0f}",
                positive=strength >= 0.5,
            )
        )
    else:
        signals.append(SubSignal("avg_traded_value_20d", False))

    if avg_vol is not None:
        signals.append(
            SubSignal(
                "avg_volume_20d",
                True,
                _clamp01(avg_vol / 1_000_000),
                f"20-day average volume is {avg_vol:,.0f}",
                positive=avg_vol >= 100_000,
            )
        )
    else:
        signals.append(SubSignal("avg_volume_20d", False))

    # Zero-volume and stale days are always computable from whatever we have.
    signals.append(
        SubSignal(
            "zero_volume_days_20d",
            True,
            _clamp01(1.0 - zero_days / 5.0),
            f"{zero_days} zero-volume day(s) in 20",
            positive=zero_days == 0,
        )
    )
    signals.append(
        SubSignal(
            "stale_price_days_20d",
            True,
            _clamp01(1.0 - stale_days / 5.0),
            f"{stale_days} stale-price day(s) in 20",
            positive=stale_days <= 1,
        )
    )
    return signals


def _score_positioning(closes: Any, metrics: dict[str, Any]) -> list[SubSignal]:
    dist_high = ind.distance_from_high(closes)
    dist_low = ind.distance_from_low(closes)
    pos = ind.position_in_range(closes)
    metrics.update(
        {
            "distance_from_52w_high": dist_high,
            "distance_from_52w_low": dist_low,
            "position_in_52w_range": pos,
        }
    )

    signals: list[SubSignal] = []

    if dist_high is not None:
        # Momentum orientation: nearer the 52-week high scores higher, peaking at
        # the high itself (strength 1.0 at the high, ~0.5 at 25% below, 0 at 50%+
        # below). This is trend-following, not value — a stock near its high is
        # "strong", not "cheap". A value/mean-reversion screen would invert this.
        strength = _clamp01(1.0 - dist_high / 0.5)
        signals.append(
            SubSignal(
                "distance_from_52w_high",
                True,
                strength,
                f"{dist_high:.1%} below the 52-week high",
                positive=dist_high < 0.25,
            )
        )
    else:
        signals.append(SubSignal("distance_from_52w_high", False))

    if pos is not None:
        signals.append(
            SubSignal(
                "position_in_52w_range",
                True,
                pos,
                f"At the {pos:.0%} mark of its 52-week range",
                positive=pos >= 0.5,
            )
        )
    else:
        signals.append(SubSignal("position_in_52w_range", False))

    if dist_low is not None:
        signals.append(
            SubSignal(
                "distance_from_52w_low",
                True,
                _clamp01(dist_low / 1.0),
                f"{dist_low:.1%} above the 52-week low",
                positive=dist_low > 0.20,
            )
        )
    else:
        signals.append(SubSignal("distance_from_52w_low", False))

    return signals


def score_value(
    series: PriceSeries, *, fundamentals: dict[str, Decimal | None] | None = None
) -> ValueResult:
    """Valuation lens (§6 extension): how cheap does this look? 0-100.

    Higher = more depressed / potentially undervalued. This is the mirror image
    of the momentum core: where momentum rewards a stock near its highs, value
    rewards one that has pulled back, sits low in its range, trades below its
    long-term average, and is oversold (low RSI). Where fundamentals exist, a
    high earnings yield, low price-to-book and high dividend yield add to it.

    Price value and fundamental value are blended when both are present, so the
    score does not swing on fundamentals that most instruments lack; price value
    alone carries it otherwise. Missing signals are dropped, never scored as
    zero — the same discipline as the core (acceptance criterion 7).
    """
    closes = series.preferred_close
    metrics: dict[str, Any] = {}
    price_signals: list[SubSignal] = []

    # -- Price-based value: cheapness relative to the instrument's own history --

    dist_high = ind.distance_from_high(closes)
    if dist_high is not None:
        # A deeper pullback from the 52-week high reads as cheaper. Full strength
        # around a 40% drawdown; at the high, value is ~0.
        price_signals.append(
            SubSignal(
                "pullback_from_high",
                True,
                _clamp01(dist_high / 0.4),
                f"{dist_high:.1%} below its 52-week high",
                positive=dist_high > 0.15,
            )
        )
        metrics["pullback_from_high"] = dist_high
    else:
        price_signals.append(SubSignal("pullback_from_high", False))

    pos = ind.position_in_range(closes)
    if pos is not None:
        # Low in the 52-week range = cheap. Invert the momentum reading.
        price_signals.append(
            SubSignal(
                "low_in_range",
                True,
                _clamp01(1.0 - pos),
                f"At the {pos:.0%} mark of its 52-week range",
                positive=pos < 0.5,
            )
        )
        metrics["position_in_52w_range"] = pos
    else:
        price_signals.append(SubSignal("low_in_range", False))

    sma200 = ind.simple_moving_average(closes, 200)
    price = float(closes[-1]) if closes.size else None
    if sma200 is not None and price is not None and sma200 > 0:
        # Trading below the 200-day average — potentially cheap vs its trend.
        discount = (sma200 - price) / sma200  # positive when below the average
        price_signals.append(
            SubSignal(
                "below_200d_average",
                True,
                _clamp01(0.5 + discount / 0.4),
                (
                    f"{discount:.1%} below its 200-day average"
                    if discount > 0
                    else f"{-discount:.1%} above its 200-day average"
                ),
                positive=discount > 0,
            )
        )
        metrics["discount_to_sma200"] = discount
    else:
        price_signals.append(SubSignal("below_200d_average", False))

    rsi = ind.relative_strength_index(closes, 14)
    if rsi is not None:
        # Oversold (low RSI) = mean-reversion candidate. RSI 30 → strong value,
        # RSI 70 → none.
        price_signals.append(
            SubSignal(
                "oversold_rsi",
                True,
                _clamp01((70.0 - rsi) / 40.0),
                f"14-day RSI is {rsi:.0f}"
                + (" (oversold)" if rsi < 30 else " (overbought)" if rsi > 70 else ""),
                positive=rsi < 40,
            )
        )
        metrics["rsi_14"] = rsi
    else:
        price_signals.append(SubSignal("oversold_rsi", False))

    available_price = [s for s in price_signals if s.available]
    price_value = (
        100.0 * sum(s.value for s in available_price) / len(available_price)
        if available_price
        else 50.0
    )

    # -- Fundamental value: cheap vs earnings / book / yield --------------------

    fundamental_value = _score_fundamental_value(fundamentals) if fundamentals else None

    # Blend: fundamentals get a third of the weight when present, since most
    # instruments (ETFs, many lines) lack them and price value must carry.
    if fundamental_value is not None:
        blended = 0.65 * price_value + 0.35 * fundamental_value
    else:
        blended = price_value

    positives = [s.explanation for s in price_signals if s.available and s.positive]
    negatives = [s.explanation for s in price_signals if s.available and not s.positive]

    return ValueResult(
        value_score=round(blended, 2),
        price_value_score=round(price_value, 2),
        fundamental_value_score=(
            round(fundamental_value, 2) if fundamental_value is not None else None
        ),
        positive_signals=positives,
        negative_signals=negatives,
        metrics=metrics,
    )


def _score_fundamental_value(fundamentals: dict[str, Decimal | None]) -> float | None:
    """Fundamental cheapness: earnings yield, price-to-book, dividend yield.

    Distinct from `_score_fundamentals` (which measures business *quality* —
    margins, growth). This measures *cheapness*. A stock can be cheap and low
    quality (a value trap) or expensive and high quality; keeping the two scores
    separate is what lets a reader tell them apart.
    """
    signals: list[SubSignal] = []

    pe = fundamentals.get("trailing_pe")
    if pe is not None and pe > 0:
        # Earnings yield = 1/PE. A PE of 10 (10% yield) is cheap; 40 is not.
        earnings_yield = 1.0 / float(pe)
        signals.append(SubSignal("earnings_yield", True, _clamp01(earnings_yield / 0.10)))

    ptb = fundamentals.get("price_to_book")
    if ptb is not None and ptb > 0:
        # P/B of 1 or below is cheap; above ~5 is not.
        signals.append(SubSignal("price_to_book", True, _clamp01(1.0 - (float(ptb) - 1.0) / 4.0)))

    dy = fundamentals.get("dividend_yield")
    if dy is not None and dy >= 0:
        # A 4%+ yield reads as value; scales to full strength there.
        signals.append(SubSignal("dividend_yield", True, _clamp01(float(dy) / 0.04)))

    available = [s for s in signals if s.available]
    if not available:
        return None
    return 100.0 * sum(s.value for s in available) / len(available)


def _score_fundamentals(fundamentals: dict[str, Decimal | None]) -> float | None:
    """Optional, separate fundamental score (§6).

    Never feeds the core score. Every field is optional, and a field that is
    absent is simply not scored — the same missing-data discipline as the core.
    Returns None when nothing at all was available.
    """
    signals: list[SubSignal] = []

    pe = fundamentals.get("trailing_pe")
    if pe is not None and pe > 0:
        # Lower P/E scores higher; full strength around 10, fading past 40.
        strength = _clamp01(1.0 - (float(pe) - 10.0) / 30.0)
        signals.append(SubSignal("trailing_pe", True, strength))

    margin = fundamentals.get("profit_margin")
    if margin is not None:
        signals.append(SubSignal("profit_margin", True, _clamp01(float(margin) / 0.30)))

    growth = fundamentals.get("revenue_growth")
    if growth is not None:
        signals.append(SubSignal("revenue_growth", True, _clamp01(0.5 + float(growth) / 0.4)))

    dte = fundamentals.get("debt_to_equity")
    if dte is not None and dte >= 0:
        signals.append(SubSignal("debt_to_equity", True, _clamp01(1.0 - float(dte) / 200.0)))

    available = [s for s in signals if s.available]
    if not available:
        return None
    return 100.0 * sum(s.value for s in available) / len(available)
