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
# `sector` scores the health of the instrument's own industry (via its sector
# ETF); the others were trimmed to make room while keeping the 0-100 scale, so
# the screening/watchlist thresholds still mean the same thing.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 20.0,
    "momentum": 20.0,
    "risk": 15.0,
    "liquidity": 15.0,
    "positioning": 10.0,
    "sector": 20.0,
}

DEFAULT_THRESHOLDS: dict[str, float] = {"screening": 75.0, "watchlist": 60.0}

#: Window for the rate-sensitivity and beta reads. Matches the risk engine's
#: `correlation_window_short` default so the scanner and the sizing stage are
#: talking about the same period.
RATE_CORRELATION_WINDOW = 60

#: Absolute rate correlation at which the rate-sensitivity signal scores zero.
#: Set at the risk engine's `correlation_threshold` default: past this point a
#: holding is more a bet on bond yields than on the company.
RATE_CORRELATION_BAD_AT = 0.80

# -- Final-score factor weights. Fundamentals-first: intrinsic value + P/E lead,
# with price cheapness and the reversal/quality/sector signals in support. Sum to
# 1.0; a missing factor drops out and the rest re-normalise (see combine_final_score).
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "fundamental_value": 0.30,
    # Raised from 0.17 at the expense of `reversal`. This is the factor that
    # rewards being *at* a low, which is what the mean-reversion entry needs.
    "price_cheapness": 0.29,
    # Cut from 0.17. `reversal` rewards "RSI rising" and "reclaimed the 20-day
    # average" — and that average *is* the Bollinger middle band the strategy
    # sells at. At 0.17 the ranking was actively selecting for stocks sitting in
    # the strategy's exit zone: of a top 20, eleven were at or above the middle
    # band and none below the lower one. Kept non-zero because a stock turning
    # up is still real information for a human reading the table; it just should
    # not drive the ranking a dip-buyer depends on.
    "reversal": 0.05,
    "quality": 0.12,
    "sector": 0.09,
    # Insider *buying* only; selling is a penalty, not a factor (see
    # DEFAULT_INSIDER_SELL_PENALTY). Held to 0.15 because this is the one factor
    # absent for most instruments — Form 4 has no UK equivalent, so ~60% of a
    # Trading 212 universe can never carry it — and raising it further would let
    # a single buy signal lift an otherwise ordinary company over a better one
    # that simply files in the wrong jurisdiction. The other five were shaved
    # proportionally, so their balance relative to each other is unchanged.
    "insider": 0.15,
}

#: A stock with no fundamentals at all loses this fraction of its final score — a
#: mild, deliberate disadvantage (we cannot confirm the value is real), not a cliff.
DEFAULT_FUNDAMENTALS_PENALTY = 0.10

#: Ceiling on the insider-selling penalty. Expressed as a penalty rather than as
#: a negative factor deliberately: a penalty only ever subtracts, so it can be
#: this aggressive without disadvantaging the majority of the catalogue that SEC
#: Form 4 does not cover. Applies only to discretionary chief-officer selling —
#: the one cohort a 686-event backtest found a real effect for (-6.28% excess at
#: one month, t = -2.43). See app.services.insider.
DEFAULT_INSIDER_SELL_PENALTY = 0.40

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
    # -- The five factors the *primary* (final) score blends (each 0-100, or None
    # when uncomputable). See combine_final_score.
    fundamental_value: float | None = None  # intrinsic value + P/E (heaviest)
    price_cheapness: float | None = None  # price pulled back / oversold
    reversal: float | None = None  # turning up from a decline
    quality: float | None = None  # fundamentals soundness + low-risk + liquidity
    sector_factor: float | None = None  # sector-ETF health, 0-100
    #: Insider buying/selling, 0-100 with 50 neutral. None when the instrument
    #: has no Form 4 filings in the window — which is the common case, and why
    #: it must be None rather than 50: a neutral value would dilute every other
    #: factor with a non-observation.
    insider: float | None = None
    #: Fraction of the final score to remove for insider selling (0..0.30).
    #: Separate from `insider` because buying and selling act through different
    #: mechanisms: buying lifts a weighted factor, selling discounts the total.
    insider_sell_penalty: float = 0.0


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
    sector: PriceSeries | None = None,
    rates: PriceSeries | None = None,
    pead: float | None = None,
    fundamentals: dict[str, Decimal | None] | None = None,
) -> ScoreResult:
    """Score one instrument's series. Pure — no I/O, fully deterministic.

    `sector`, when given, is the price series of the instrument's sector proxy
    (a sector ETF). It feeds two things: a relative-strength-vs-sector momentum
    signal, and a dedicated `sector` category measuring the industry's own
    health. Absent (untagged instrument, or a sector with no proxy) it drops out
    with no penalty, exactly like `benchmark`.

    `rates` is the bond-price proxy for the rate-sensitivity risk signal. It
    must be a *price* series, not a yield series — yields move inversely to
    prices, so passing one would silently invert the correlation's meaning.
    Absent, the signal drops out with no penalty like the rest.
    """
    weights = weights or DEFAULT_WEIGHTS
    thresholds = thresholds or DEFAULT_THRESHOLDS
    closes = series.preferred_close
    volumes = series.volume
    metrics: dict[str, Any] = {}

    trend = _score_trend(series, closes, metrics)
    momentum = _score_momentum(closes, benchmark, sector, pead, metrics)
    risk = _score_risk(closes, rates, metrics)
    liquidity = _score_liquidity(closes, volumes, metrics)
    positioning = _score_positioning(closes, metrics)
    sector_signals = _score_sector(closes, sector, metrics)

    categories = {
        "trend": _category_points(trend, weights.get("trend", 20.0), "trend"),
        "momentum": _category_points(momentum, weights.get("momentum", 20.0), "momentum"),
        "risk": _category_points(risk, weights.get("risk", 15.0), "risk"),
        "liquidity": _category_points(liquidity, weights.get("liquidity", 15.0), "liquidity"),
        "positioning": _category_points(
            positioning, weights.get("positioning", 10.0), "positioning"
        ),
        "sector": _category_points(sector_signals, weights.get("sector", 20.0), "sector"),
    }
    core_score = sum(c.points for c in categories.values())

    all_signals = trend + momentum + risk + liquidity + positioning + sector_signals
    available = sum(1 for s in all_signals if s.available)
    completeness = available / len(all_signals) if all_signals else 0.0

    # Confidence blends data completeness with history depth: a full signal set
    # over 40 days is less trustworthy than the same set over 300.
    history_factor = _clamp01(series.length / PREFERRED_HISTORY_DAYS)
    confidence = _clamp01(0.5 * completeness + 0.5 * history_factor)

    fundamental_score = _score_fundamentals(fundamentals) if fundamentals else None
    value = score_value(series, fundamentals=fundamentals)

    # -- The five factors the final score blends. Reuse the signals already
    # computed here (risk/liquidity for soundness, the value lens for
    # cheapness/intrinsic), and add the reversal read.
    reversal_score = score_reversal(series)
    quality_score = _score_quality(risk, liquidity, fundamentals)
    sector_factor = _factor_from_signals(sector_signals)

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
        fundamental_value=value.fundamental_value_score if value else None,
        price_cheapness=value.price_value_score if value else None,
        reversal=round(reversal_score, 2) if reversal_score is not None else None,
        quality=round(quality_score, 2),
        sector_factor=round(sector_factor, 2) if sector_factor is not None else None,
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


def combine_final_score(
    result: ScoreResult,
    *,
    weights: dict[str, float] | None = None,
    fundamentals_penalty: float = DEFAULT_FUNDAMENTALS_PENALTY,
) -> float:
    """The absolute 0-100 ranking score: a fundamentals-first weighted blend.

    Weighted average of the six factors, over *available* factors only (a
    missing factor drops out and the remainder re-normalise), so the result is
    always a clean 0-100 and absence is neutral — not zero. Because the factors
    are different axes (intrinsic value / price level / turn / soundness /
    sector) rather than inverses, the blend reinforces instead of cancelling.

    Two multiplicative penalties apply after the blend. A stock with no
    fundamentals at all (its Fundamental Value factor is None) loses
    `fundamentals_penalty` of its score: a deliberate, mild
    disadvantage since its value cannot be confirmed. The score is *absolute* —
    it does not depend on the rest of the scanned batch — so it is comparable run
    to run.
    """
    weights = weights or DEFAULT_FACTOR_WEIGHTS
    factors: dict[str, float | None] = {
        "fundamental_value": result.fundamental_value,
        "price_cheapness": result.price_cheapness,
        "reversal": result.reversal,
        "quality": result.quality,
        "sector": result.sector_factor,
        "insider": result.insider,
    }

    weighted = 0.0
    total_weight = 0.0
    for name, score in factors.items():
        if score is None:
            continue
        w = weights.get(name, 0.0)
        weighted += w * score
        total_weight += w
    if total_weight <= 0:
        return 0.0

    final = weighted / total_weight
    if result.fundamental_value is None:
        final *= 1.0 - fundamentals_penalty
    # Insider selling discounts the whole score rather than competing as a
    # factor. A stock with no filings has a penalty of 0.0 and is untouched.
    if result.insider_sell_penalty:
        final *= 1.0 - min(max(result.insider_sell_penalty, 0.0), 1.0)
    return round(_clamp01(final / 100.0) * 100.0, 2)


def _factor_from_signals(signals: list[SubSignal]) -> float | None:
    """A 0-100 factor score from raw sub-signals, or None if none are available."""
    available = [s.value for s in signals if s.available]
    if not available:
        return None
    return 100.0 * sum(available) / len(available)


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
    closes: Any,
    benchmark: PriceSeries | None,
    sector: PriceSeries | None,
    pead: float | None,
    metrics: dict[str, Any],
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
        # CAPM beta against the market benchmark. Reported, never scored: a high
        # beta is not by itself good or bad, and the risk engine already sizes
        # against volatility directly. It earns a signal only if it proves out.
        metrics["beta_vs_benchmark"] = ind.beta(
            ind.daily_returns(closes),
            ind.daily_returns(benchmark.preferred_close),
            RATE_CORRELATION_WINDOW,
        )
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

    # Relative strength vs the instrument's own sector — beating your peers is a
    # stronger signal than beating the broad market. Same shape as the benchmark
    # signal; drops out when there is no sector proxy.
    if sector is not None:
        rel_sector = ind.relative_momentum(
            closes, sector.preferred_close, ind.TRADING_DAYS_PER_YEAR
        )
        metrics["relative_momentum_vs_sector_12m"] = rel_sector
        if rel_sector is None:
            signals.append(SubSignal("relative_momentum_vs_sector", False))
        else:
            strength = _clamp01(0.5 + rel_sector / 0.4)
            signals.append(
                SubSignal(
                    "relative_momentum_vs_sector",
                    True,
                    strength,
                    f"12-month return vs its sector is {rel_sector:+.1%}",
                    positive=rel_sector > 0,
                )
            )
    else:
        signals.append(SubSignal("relative_momentum_vs_sector", False))

    # Post-earnings announcement drift. A momentum signal, but an event-driven
    # one: the other measures here read a trailing window of price, whereas this
    # reads how the market reacted to a specific piece of news and bets the
    # repricing is not finished. Computed outside `score_series` (it needs the
    # earnings table) and passed in already scored 0-100.
    metrics["pead_score"] = pead
    if pead is None:
        # No report inside the drift window, or no coverage for this listing —
        # Form-10-style earnings dates are thin outside the US. Drops out of the
        # category average rather than marking the stock down for a data gap.
        signals.append(SubSignal("earnings_drift", False))
    else:
        strength = _clamp01(pead / 100.0)
        signals.append(
            SubSignal(
                "earnings_drift",
                True,
                strength,
                f"post-earnings drift score {pead:.0f}",
                positive=pead >= 50.0,
            )
        )

    return signals


def _score_risk(closes: Any, rates: PriceSeries | None, metrics: dict[str, Any]) -> list[SubSignal]:
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

    # Rate sensitivity. Scored in the *risk* category because a holding that
    # tracks bond yields carries a macro exposure the other risk measures cannot
    # see — a low-volatility REIT and a low-volatility staples name look alike on
    # drawdown and deviation, and behave nothing alike when yields move.
    #
    # Magnitude, not direction: a strongly *negatively* rate-correlated holding
    # is just as much a bet on rates as a positively correlated one, so the
    # signal is symmetric around zero. Independence from rates scores high.
    rate_corr = (
        ind.rolling_correlation(
            ind.daily_returns(closes),
            ind.daily_returns(rates.preferred_close),
            RATE_CORRELATION_WINDOW,
        )
        if rates is not None
        else None
    )
    metrics["rate_correlation_60d"] = rate_corr
    if rate_corr is None:
        # No rates proxy ingested yet, or too little history to correlate. Drops
        # out of the category average rather than scoring zero.
        signals.append(SubSignal("rate_sensitivity", False))
    else:
        strength = _clamp01(1.0 - abs(rate_corr) / RATE_CORRELATION_BAD_AT)
        signals.append(
            SubSignal(
                "rate_sensitivity",
                True,
                strength,
                f"60-day correlation to rates is {rate_corr:+.2f}",
                positive=strength >= 0.5,
            )
        )
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


def _score_sector(
    closes: Any, sector: PriceSeries | None, metrics: dict[str, Any]
) -> list[SubSignal]:
    """Health of the instrument's own sector, via its sector-ETF proxy (§6).

    Rewards being in an industry that is itself in favour: the sector index above
    its 200-day average, that average rising, and positive medium-term sector
    momentum. Entirely dropped (never penalised) when the instrument has no
    sector tag or its sector has no proxy series — every sub-signal reports
    unavailable and `_category_points` awards the neutral midpoint, the same
    missing-data discipline as any other category.
    """
    if sector is None:
        return [
            SubSignal("sector_above_200d", False),
            SubSignal("sector_trend_rising", False),
            SubSignal("sector_momentum_3m", False),
            SubSignal("sector_momentum_6m", False),
        ]

    sclose = sector.preferred_close
    price = float(sclose[-1]) if sclose.size else None
    sma200 = ind.simple_moving_average(sclose, 200)
    slope = ind.sma_slope(sclose, 200, slope_window=21)
    ret3m = ind.trailing_return(sclose, ind.TRADING_DAYS_PER_MONTH * 3)
    ret6m = ind.trailing_return(sclose, ind.TRADING_DAYS_PER_MONTH * 6)
    metrics.update(
        {
            "sector_sma200": sma200,
            "sector_sma200_slope": slope,
            "sector_return_3m": ret3m,
            "sector_return_6m": ret6m,
        }
    )

    signals: list[SubSignal] = []

    if price is not None and sma200 is not None:
        above = price > sma200
        signals.append(
            SubSignal(
                "sector_above_200d",
                True,
                1.0 if above else 0.0,
                "Sector is above its 200-day average"
                if above
                else "Sector is below its 200-day average",
                positive=above,
            )
        )
    else:
        signals.append(SubSignal("sector_above_200d", False))

    if slope is not None:
        signals.append(
            SubSignal(
                "sector_trend_rising",
                True,
                _clamp01(0.5 + slope / 0.002),
                "Sector trend is rising" if slope > 0 else "Sector trend is falling",
                positive=slope > 0,
            )
        )
    else:
        signals.append(SubSignal("sector_trend_rising", False))

    for label, ret in (("3m", ret3m), ("6m", ret6m)):
        if ret is None:
            signals.append(SubSignal(f"sector_momentum_{label}", False))
        else:
            signals.append(
                SubSignal(
                    f"sector_momentum_{label}",
                    True,
                    _clamp01(0.5 + ret / 0.4),
                    f"Sector {label} return is {ret:+.1%}",
                    positive=ret > 0,
                )
            )

    return signals


def score_reversal(series: PriceSeries) -> float | None:
    """How strongly a beaten-down instrument is *turning up* (0-100), or None.

    The "predicted to recover" signal: a stock that fell but is now stabilising
    and inflecting higher. Distinct from cheapness (a price *level*) and from
    momentum (already up) — it reads the recent *derivative*. Missing sub-signals
    drop out; None when history is too short to read a turn.
    """
    closes = series.preferred_close
    if closes.size < 30:
        return None
    price = float(closes[-1])
    signals: list[SubSignal] = []

    # RSI rising, especially off an oversold trough ~10 sessions ago.
    rsi_now = ind.relative_strength_index(closes, 14)
    rsi_prev = ind.relative_strength_index(closes[:-10], 14) if closes.size > 24 else None
    if rsi_now is not None and rsi_prev is not None:
        strength = _clamp01(0.5 + (rsi_now - rsi_prev) / 20.0)
        if rsi_prev < 40 and rsi_now < 60:
            strength = _clamp01(strength + 0.2)
        signals.append(SubSignal("rsi_turning_up", True, strength))

    # Reclaimed the 20-day average (short-term trend regained).
    sma20 = ind.simple_moving_average(closes, 20)
    if sma20 is not None and sma20 > 0:
        signals.append(
            SubSignal("reclaimed_20d", True, _clamp01(0.5 + (price - sma20) / sma20 / 0.10))
        )

    # Short-term up while the medium term is still down = a bottom forming.
    ret1m = ind.trailing_return(closes, ind.TRADING_DAYS_PER_MONTH)
    ret6m = ind.trailing_return(closes, ind.TRADING_DAYS_PER_MONTH * 6)
    if ret1m is not None and ret6m is not None and ret6m < 0:
        signals.append(SubSignal("momentum_inflection", True, _clamp01(ret1m / 0.10)))

    # Bounced off the 52-week low, but not yet recovered (sweet spot ~15% up).
    dist_low = ind.distance_from_low(closes)
    if dist_low is not None:
        off_low = (
            _clamp01(dist_low / 0.15)
            if dist_low <= 0.15
            else _clamp01(1.0 - (dist_low - 0.15) / 0.35)
        )
        signals.append(SubSignal("off_the_low", True, off_low))

    # Recent selling pressure easing vs the longer window.
    dd20 = ind.downside_deviation(closes, 20)
    dd60 = ind.downside_deviation(closes, 60)
    if dd20 is not None and dd60 is not None and dd60 > 0:
        signals.append(SubSignal("downside_easing", True, _clamp01(1.0 - dd20 / dd60)))

    return _factor_from_signals(signals)


def _score_quality(
    risk_signals: list[SubSignal],
    liquidity_signals: list[SubSignal],
    fundamentals: dict[str, Decimal | None] | None,
) -> float:
    """Is this a sound business, not a falling knife (0-100)?

    Blends fundamental quality (margins, growth, low debt — when available) with
    always-computable soundness from candles (low volatility, contained drawdown,
    liquidity). The soundness half comes from prices, so quality is always
    scorable even with no fundamentals.
    """
    parts: list[float] = []
    fund_q = _score_fundamentals(fundamentals) if fundamentals else None
    if fund_q is not None:
        parts.append(fund_q / 100.0)
    for signals in (risk_signals, liquidity_signals):
        available = [s.value for s in signals if s.available]
        if available:
            parts.append(sum(available) / len(available))
    if not parts:
        return 50.0
    return 100.0 * sum(parts) / len(parts)


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
    ptb = fundamentals.get("price_to_book")

    if pe is not None and pe > 0:
        # Earnings yield = 1/PE. A PE of 10 (10% yield) is cheap; 40 is not.
        earnings_yield = 1.0 / float(pe)
        signals.append(SubSignal("earnings_yield", True, _clamp01(earnings_yield / 0.10)))

    if ptb is not None and ptb > 0:
        # P/B of 1 or below is cheap; above ~5 is not.
        signals.append(SubSignal("price_to_book", True, _clamp01(1.0 - (float(ptb) - 1.0) / 4.0)))

    # Graham intrinsic value: fair when P/E·P/B = 22.5, so the margin of safety
    # below that fair value is 1 - √(P/E·P/B / 22.5). ~50%+ below reads as a full
    # signal; at or above Graham fair value it is zero. Needs both ratios.
    if pe is not None and pe > 0 and ptb is not None and ptb > 0:
        import math

        margin_of_safety = 1.0 - math.sqrt(float(pe) * float(ptb) / 22.5)
        signals.append(SubSignal("graham_margin_of_safety", True, _clamp01(margin_of_safety / 0.5)))

    # PEG: P/E relative to earnings growth. Cheap vs growth when ≤ 1.
    growth = fundamentals.get("earnings_growth")
    if pe is not None and pe > 0 and growth is not None and float(growth) > 0:
        peg = float(pe) / (float(growth) * 100.0)
        signals.append(SubSignal("peg", True, _clamp01((2.0 - peg) / 1.5)))

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
