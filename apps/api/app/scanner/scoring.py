"""Scanner scoring (§6).

Turns a `PriceSeries` (and optional fundamentals) into **one** absolute 0-100
score: how much is this worth owning?

Three rules are non-negotiable and are the reason this module is written the way
it is:

  1. **Missing optional data never lowers the score** (acceptance 7). Each
     sub-signal reports whether it could be computed; a signal that could not is
     dropped from both the numerator and the denominator of its group, and a
     group with nothing at all drops out of the blend with its weight removed
     from the divisor. Absence lowers *confidence*, not score.

  2. **The score is absolute, never cohort-relative.** A batch of 400 scored
     tonight is directly comparable with a different 400 scored tomorrow, which
     is what makes a rotation over ~20,000 instruments mean anything.

  3. **No output asserts an instrument is a good investment** (§0). Scores are
     framed as "passes the configured screen" bands, and the explanations use
     neutral, mechanical language ("29.0% below its 52-week high"), never
     recommendations.

**Why there is no momentum here.** The scanner answers "what is worth owning";
the strategy (`app.strategies.mean_reversion`) answers "when to buy it". Fast
signals belong to the second question, and not only on grounds of tidiness: the
scanner rotates 200-2000 names a night against a catalogue of ~20,000, so any
given score is 10-100 days old when it is compared against a fresh one. A P/E or
a debt ratio survives that staleness; a one-month return does not. Trend and
momentum readings are still *computed* into `metrics` for the results table —
they are free from candles already loaded — but they are deliberately not scored.

That split is also what keeps the score internally consistent. Reading a fact
once, in one direction, is only possible when every group shares an orientation:
here, cheap and sound scores high. `distance_from_52w_high` cannot be a virtue in
one group and a fault in another, because the group that wanted it to be a virtue
now lives in the strategy.

Every threshold and weight is configurable; the constants here are only the
defaults §6 specifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.indicators import functions as ind
from app.indicators.series import PriceSeries
from app.models.scanner import Classification

# -- Default group weights. Sum to 100. -------------------------------------
#
# Fundamentals-first, and deliberately less cheapness-led than the first cut:
# intrinsic value leads, with insider buying and business soundness given equal
# second billing, then price level, then sector health.
#
# Retuned 2026-08-09, in two passes. Cheapness fell 30 -> 24 and quality rose
# 13 -> 21 to close the gap the original weights left open: a stock that is cheap
# *because the business is deteriorating* scored far too close to a sound one.
# Measured on a value trap (cheapness 90, quality 35) against a compounder
# (cheapness 45, quality 85), the trap's lead fell from +13.8 to +5.1 points.
#
# The second pass also took insider back from 18 to 15, because a single lookup
# carrying 18 points made *having US filings at all* too strong a differentiator
# — see the note on that key below.
DEFAULT_WEIGHTS: dict[str, float] = {
    # Graham margin of safety, earnings yield, price-to-book, PEG, dividend
    # yield. The heaviest group, and the one most instruments lack entirely —
    # which is why a missing group must renormalise rather than score zero.
    "value": 30.0,
    # Where the price sits against its own year. Still substantial, because this
    # is the group that rewards being *at* a low, which is what the
    # mean-reversion entry downstream needs — but no longer able to carry a
    # failing business into the top ranks on price alone.
    "cheapness": 24.0,
    # Insider *buying* only; selling is a penalty, not a group (see
    # DEFAULT_INSIDER_SELL_PENALTY).
    #
    # Note what this weight means in practice: the group holds a *single*
    # measurement, so all 15 points rest on one lookup — still the heaviest
    # per-measurement weight in the system, at 1.9x cheapness and 15x one risk
    # signal, and able to swing a score 15 points end to end.
    #
    # That is why it came back down from 18. Form 4 has no UK equivalent, so
    # ~60% of a Trading 212 universe can never carry this group at all; the
    # heavier this key, the more the ranking rewards *being US-listed* rather
    # than being a good business. `test_buying_cannot_dominate_the_ranking` pins
    # the absent-vs-best gap under 12 points (currently 6.0). Still worth
    # watching the top ranks for a US skew.
    "insider": 15.0,
    # Is this a sound business in a sound market, or a falling knife? The
    # heaviest group after value, and deliberately ahead of both cheapness and
    # insider: whether the business is deteriorating decides whether a low price
    # is an opportunity or a warning, so it should outrank both the price itself
    # and who happens to be buying.
    "quality": 21.0,
    # Health of the instrument's own industry, via its sector ETF.
    "sector": 10.0,
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

#: Polarity at which news sentiment scores its worst (0.0) and its best (1.0).
#: Not the full [-1, +1] range: a polarity of exactly -1 needs every tone word in
#: a week's headlines to be negative, which in practice only happens when a
#: company generated one bleak article and nothing else. Saturating at ±0.5 makes
#: the signal responsive over the range real companies actually occupy instead of
#: bunching almost everything around the midpoint.
SENTIMENT_SATURATION = 0.50

#: A stock with no fundamentals at all loses this fraction of its score — a mild,
#: deliberate disadvantage (we cannot confirm the value is real), not a cliff.
DEFAULT_FUNDAMENTALS_PENALTY = 0.10

#: Ceiling on the insider-selling penalty. Expressed as a penalty rather than as
#: a negative group deliberately: a penalty only ever subtracts, so it can be
#: this aggressive without disadvantaging the majority of the catalogue that SEC
#: Form 4 does not cover. Applies only to discretionary chief-officer selling —
#: the one cohort a 686-event backtest found a real effect for (-6.28% excess at
#: one month, t = -2.43). See app.services.insider.
DEFAULT_INSIDER_SELL_PENALTY = 0.40

#: Preferred minimum history (§6). Scoring proceeds below this with reduced
#: confidence rather than refusing — a shorter series is still informative,
#: it is just less certain, and that uncertainty is reported.
PREFERRED_HISTORY_DAYS = ind.TRADING_DAYS_PER_YEAR

#: The five groups, in the order they are reported.
GROUP_NAMES = ("value", "cheapness", "insider", "quality", "sector")


@dataclass(slots=True)
class SubSignal:
    """One measurable component of a group score.

    `available` is the whole point: an unavailable signal contributes nothing to
    either side of its group average, so missing data cannot drag a score down.
    `value` is the 0..1 normalised strength when available.
    """

    name: str
    available: bool
    value: float = 0.0
    #: Human-readable, mechanical description for the UI when this signal is
    #: strongly positive or negative.
    explanation: str = ""
    positive: bool = False


@dataclass(slots=True)
class GroupScore:
    """One weighted group's contribution, as a 0-100 reading.

    `score` is None when nothing in the group could be computed. That is
    materially different from zero: a None group is removed from the blend
    *along with its weight*, so absence is neutral rather than damning. See
    `combine_score`.
    """

    name: str
    score: float | None
    weight: float
    signals: list[SubSignal] = field(default_factory=list)

    @property
    def signals_available(self) -> int:
        return sum(1 for s in self.signals if s.available)

    @property
    def coverage(self) -> float:
        return self.signals_available / len(self.signals) if self.signals else 0.0


@dataclass(slots=True)
class ScoreResult:
    """Everything one instrument's scan produced."""

    #: The absolute 0-100 ranking score. The only score this module emits.
    score: float
    groups: dict[str, GroupScore]
    classification: Classification
    data_completeness: float
    confidence: float
    candles_used: int
    positive_signals: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    #: Computed-but-unscored readings, including every trend and momentum
    #: measure. Reported for a human reading the results table; they contribute
    #: nothing to `score` — see the module docstring.
    metrics: dict[str, Any] = field(default_factory=dict)
    #: Fraction of the score removed for insider selling (0..0.40). Separate
    #: from the `insider` group because buying and selling act through different
    #: mechanisms: buying lifts a weighted group, selling discounts the total.
    insider_sell_penalty: float = 0.0

    def group_score(self, name: str) -> float | None:
        group = self.groups.get(name)
        return group.score if group is not None else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _group(name: str, signals: list[SubSignal], weight: float) -> GroupScore:
    """Average the available signals into a 0-100 group score.

    The average is over *available* signals only. A group with three of five
    signals available is scored on those three — never diluted toward zero by
    the two that could not be computed. With none available the score is None,
    which drops the group out of the blend entirely.
    """
    available = [s.value for s in signals if s.available]
    score = 100.0 * sum(available) / len(available) if available else None
    return GroupScore(name=name, score=score, weight=weight, signals=signals)


def score_series(
    series: PriceSeries,
    *,
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    benchmark: PriceSeries | None = None,
    sector: PriceSeries | None = None,
    rates: PriceSeries | None = None,
    sentiment: float | None = None,
    fundamentals: dict[str, Decimal | None] | None = None,
    insider: float | None = None,
    insider_sell_penalty: float = 0.0,
    fundamentals_penalty: float = DEFAULT_FUNDAMENTALS_PENALTY,
) -> ScoreResult:
    """Score one instrument. Pure — no I/O, fully deterministic.

    `benchmark` and `sector` are the market and sector-ETF proxy series. Neither
    is scored directly any more; `sector` drives the sector group and both feed
    reported-only relative-strength metrics. Absent, they drop out with no
    penalty.

    `rates` is the bond-*price* proxy for the rate-sensitivity signal. It must be
    a price series, not a yield series — yields move inversely to prices, so
    passing one would silently invert the correlation's meaning.

    `sentiment` is the Loughran-McDonald news polarity in [-1, +1], read from the
    stored snapshot rather than computed here so this function stays pure.

    `insider` is a 0-100 reading of Form 4 buying, looked up rather than derived
    from the series, and None for most of the catalogue.
    """
    weights = weights or DEFAULT_WEIGHTS
    thresholds = thresholds or DEFAULT_THRESHOLDS
    closes = series.preferred_close
    volumes = series.volume
    metrics: dict[str, Any] = {}

    # Risk and liquidity are scored as signal lists because Quality blends their
    # averages, but they are not groups in their own right.
    risk_signals = _score_risk(closes, rates, sentiment, metrics)
    liquidity_signals = _score_liquidity(closes, volumes, metrics)

    # A partial `weights` dict falls back to the defaults key by key, rather than
    # to a literal repeated here — two copies of a weight would eventually
    # disagree, and the one that lost would do so silently.
    def _w(name: str) -> float:
        return weights.get(name, DEFAULT_WEIGHTS[name])

    groups: dict[str, GroupScore] = {
        "value": _group("value", _score_value(fundamentals), _w("value")),
        "cheapness": _group("cheapness", _score_cheapness(closes, metrics), _w("cheapness")),
        "insider": _insider_group(insider, _w("insider")),
        "quality": _quality_group(fundamentals, risk_signals, liquidity_signals, _w("quality")),
        "sector": _group("sector", _score_sector(sector, metrics), _w("sector")),
    }

    # Unscored context for the results table. Computed last so it cannot be
    # mistaken for an input to anything above.
    _report_trend_and_momentum(closes, benchmark, sector, metrics)

    score = combine_score(
        groups,
        fundamentals_penalty=fundamentals_penalty,
        insider_sell_penalty=insider_sell_penalty,
    )

    scored_signals = (
        [s for name in ("value", "cheapness", "quality", "sector") for s in groups[name].signals]
        + risk_signals
        + liquidity_signals
    )
    # Risk and liquidity appear once each: they are Quality's inputs, and
    # `groups["quality"].signals` is left empty precisely so they are not
    # double-counted in the completeness and explanation tallies below.
    available = sum(1 for s in scored_signals if s.available)
    completeness = available / len(scored_signals) if scored_signals else 0.0

    # Confidence blends data completeness with history depth: a full signal set
    # over 40 days is less trustworthy than the same set over 300.
    history_factor = _clamp01(series.length / PREFERRED_HISTORY_DAYS)
    confidence = _clamp01(0.5 * completeness + 0.5 * history_factor)

    positives = [
        s.explanation for s in scored_signals if s.available and s.positive and s.explanation
    ]
    negatives = [
        s.explanation for s in scored_signals if s.available and not s.positive and s.explanation
    ]
    missing = [s.name for s in scored_signals if not s.available]
    if groups["insider"].score is None:
        missing.append("insider_activity")

    return ScoreResult(
        score=score,
        groups=groups,
        classification=classify(score, thresholds),
        data_completeness=round(completeness, 4),
        confidence=round(confidence, 4),
        candles_used=series.length,
        positive_signals=positives,
        negative_signals=negatives,
        missing_information=missing,
        metrics=metrics,
        insider_sell_penalty=insider_sell_penalty,
    )


def combine_score(
    groups: dict[str, GroupScore],
    *,
    fundamentals_penalty: float = DEFAULT_FUNDAMENTALS_PENALTY,
    insider_sell_penalty: float = 0.0,
) -> float:
    """Weighted mean of the available groups, then two multiplicative penalties.

    The divisor is **the summed weight of the groups that produced a score**, not
    the full 100. That single detail is what makes absence neutral: a UK stock
    with no Form 4 filings divides by 84, not 100, so it is neither rewarded nor
    punished on an axis nobody can measure for it. Scoring a missing group as a
    "neutral" 50 instead would drag every non-US company toward the middle, and
    roughly 60% of a tradable catalogue is non-US.

    Because the groups are different axes (intrinsic value / price level /
    soundness / sector / insider) rather than inverses of one another, the blend
    reinforces instead of cancelling.
    """
    weighted = 0.0
    total_weight = 0.0
    for group in groups.values():
        if group.score is None:
            continue
        weighted += group.weight * group.score
        total_weight += group.weight
    if total_weight <= 0:
        return 0.0

    score = weighted / total_weight
    # No fundamentals at all: a mild disadvantage, since the value cannot be
    # confirmed. Not a cliff — plenty of tradable lines (ETFs especially) have
    # none and are not thereby bad.
    value_group = groups.get("value")
    if value_group is None or value_group.score is None:
        score *= 1.0 - fundamentals_penalty
    # Insider selling discounts the whole score rather than competing as a
    # group. A stock with no filings has a penalty of 0.0 and is untouched.
    if insider_sell_penalty:
        score *= 1.0 - min(max(insider_sell_penalty, 0.0), 1.0)
    return round(_clamp01(score / 100.0) * 100.0, 2)


def classify(score: float, thresholds: dict[str, float] | None = None) -> Classification:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if score >= thresholds.get("screening", 75.0):
        return Classification.SCREENING_CANDIDATE
    if score >= thresholds.get("watchlist", 60.0):
        return Classification.WATCHLIST_CANDIDATE
    return Classification.DOES_NOT_PASS


# -- Group scorers ----------------------------------------------------------
#
# Each returns a list of SubSignals (or, for the two that are not signal lists,
# a GroupScore directly). Normalisation maps a raw indicator to a 0..1 strength;
# the mappings are intentionally simple and monotonic so the score is
# explainable, not a black box.
#
# Every group shares one orientation: cheap and sound scores high. No fact is
# read twice in opposite directions.


def _score_value(fundamentals: dict[str, Decimal | None] | None) -> list[SubSignal]:
    """Intrinsic value: is this cheap against its earnings, book and growth?

    Distinct from `cheapness`, which measures the price against its own recent
    history. A stock can be low in its 52-week range and still expensive on
    earnings; keeping the two apart is what lets a reader tell those cases apart.
    """
    if not fundamentals:
        return [
            SubSignal("earnings_yield", False),
            SubSignal("price_to_book", False),
            SubSignal("graham_margin_of_safety", False),
            SubSignal("peg", False),
            SubSignal("dividend_yield", False),
        ]

    signals: list[SubSignal] = []
    pe = fundamentals.get("trailing_pe")
    ptb = fundamentals.get("price_to_book")

    if pe is not None and pe > 0:
        # Earnings yield = 1/PE. A PE of 10 (10% yield) is cheap; 40 is not.
        # This is the only place P/E is read — `quality` deliberately does not
        # score it again, since a low P/E is a statement about price, not about
        # whether the business is sound.
        earnings_yield = 1.0 / float(pe)
        signals.append(
            SubSignal(
                "earnings_yield",
                True,
                _clamp01(earnings_yield / 0.10),
                f"Earnings yield is {earnings_yield:.1%} (P/E {float(pe):.1f})",
                positive=earnings_yield >= 0.05,
            )
        )
    else:
        signals.append(SubSignal("earnings_yield", False))

    if ptb is not None and ptb > 0:
        # P/B of 1 or below is cheap; above ~5 is not.
        signals.append(
            SubSignal(
                "price_to_book",
                True,
                _clamp01(1.0 - (float(ptb) - 1.0) / 4.0),
                f"Price-to-book is {float(ptb):.2f}",
                positive=float(ptb) <= 2.0,
            )
        )
    else:
        signals.append(SubSignal("price_to_book", False))

    # Graham intrinsic value: fair when P/E·P/B = 22.5, so the margin of safety
    # below that fair value is 1 - √(P/E·P/B / 22.5). ~50%+ below reads as a full
    # signal; at or above Graham fair value it is zero. Needs both ratios.
    if pe is not None and pe > 0 and ptb is not None and ptb > 0:
        margin_of_safety = 1.0 - math.sqrt(float(pe) * float(ptb) / 22.5)
        signals.append(
            SubSignal(
                "graham_margin_of_safety",
                True,
                _clamp01(margin_of_safety / 0.5),
                f"Graham margin of safety is {margin_of_safety:+.1%}",
                positive=margin_of_safety > 0,
            )
        )
    else:
        signals.append(SubSignal("graham_margin_of_safety", False))

    # PEG: P/E relative to earnings growth. Cheap vs growth when ≤ 1.
    growth = fundamentals.get("earnings_growth")
    if pe is not None and pe > 0 and growth is not None and float(growth) > 0:
        peg = float(pe) / (float(growth) * 100.0)
        signals.append(
            SubSignal(
                "peg",
                True,
                _clamp01((2.0 - peg) / 1.5),
                f"PEG is {peg:.2f}",
                positive=peg <= 1.5,
            )
        )
    else:
        signals.append(SubSignal("peg", False))

    dy = fundamentals.get("dividend_yield")
    if dy is not None and dy >= 0:
        # A 4%+ yield reads as value; scales to full strength there.
        signals.append(
            SubSignal(
                "dividend_yield",
                True,
                _clamp01(float(dy) / 0.04),
                f"Dividend yield is {float(dy):.1%}",
                positive=float(dy) >= 0.02,
            )
        )
    else:
        signals.append(SubSignal("dividend_yield", False))

    return signals


def _score_cheapness(closes: Any, metrics: dict[str, Any]) -> list[SubSignal]:
    """Where the price sits against its own year. Lower is cheaper.

    Note what is *not* here: the RSI level. Being oversold is a timing fact, and
    the mean-reversion strategy already gates on `RSI ≤ 35` against candles that
    are fresh tonight rather than up to 100 days old. Scoring it here as well
    would count the same fact twice, in two layers, one of them stale.
    """
    signals: list[SubSignal] = []

    dist_high = ind.distance_from_high(closes)
    if dist_high is not None:
        # A deeper pullback from the 52-week high reads as cheaper. Full strength
        # around a 40% drawdown; at the high, cheapness is ~0.
        signals.append(
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
        signals.append(SubSignal("pullback_from_high", False))

    pos = ind.position_in_range(closes)
    if pos is not None:
        signals.append(
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
        signals.append(SubSignal("low_in_range", False))

    sma200 = ind.simple_moving_average(closes, 200)
    price = float(closes[-1]) if closes.size else None
    if sma200 is not None and price is not None and sma200 > 0:
        # Trading below the 200-day average — potentially cheap vs its trend.
        # The *level* is read here; the *slope* of that same average is the
        # strategy's falling-knife filter, which is why the two compose rather
        # than contradict: cheap relative to a still-rising long-term average.
        discount = (sma200 - price) / sma200  # positive when below the average
        signals.append(
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
        signals.append(SubSignal("below_200d_average", False))

    return signals


def _insider_group(insider: float | None, weight: float) -> GroupScore:
    """Form 4 buying as a group of its own.

    A single looked-up 0-100 reading rather than a signal list. None — the
    common case, since Form 4 has no UK equivalent — leaves the group scoreless
    so it drops out of the blend with its weight, which is the whole reason this
    is a group and not a neutral-50 default.
    """
    signals = [
        SubSignal(
            "insider_buying",
            True,
            _clamp01(insider / 100.0),
            f"Insider buying score {insider:.0f}",
            positive=insider >= 50.0,
        )
        if insider is not None
        else SubSignal("insider_buying", False)
    ]
    return GroupScore(name="insider", score=insider, weight=weight, signals=signals)


def _quality_group(
    fundamentals: dict[str, Decimal | None] | None,
    risk_signals: list[SubSignal],
    liquidity_signals: list[SubSignal],
    weight: float,
) -> GroupScore:
    """Is this a sound business in a sound market, not a falling knife? (0-100)

    A **three-part mean**, not a flat average of its fourteen inputs, and
    deliberately so. Flattening would give the seven risk signals half the
    group instead of a third, tripling the influence of measures that were tuned
    at their current weight — and would mean every future risk signal silently
    diluted the business fundamentals. The three parts are:

      * business soundness — margins, growth, leverage (never P/E; that is a
        price statement and belongs to `value`);
      * market risk — volatility, drawdown, rate sensitivity, news tone;
      * tradability — volume, traded value, staleness.

    Parts with nothing available are dropped, so a stock with no fundamentals is
    still scored on the two parts that come from candles. `signals` is left empty
    on the returned group: the risk and liquidity signals are tallied once by
    `score_series` and attaching them here as well would double-count them in
    the completeness figure.
    """
    parts: list[float] = []

    business = _quality_fundamentals(fundamentals) if fundamentals else None
    if business is not None:
        parts.append(business)
    for signals in (risk_signals, liquidity_signals):
        available = [s.value for s in signals if s.available]
        if available:
            parts.append(sum(available) / len(available))

    score = 100.0 * sum(parts) / len(parts) if parts else None
    return GroupScore(name="quality", score=score, weight=weight, signals=[])


def _quality_fundamentals(fundamentals: dict[str, Decimal | None]) -> float | None:
    """Business soundness from the accounts: margins, growth, leverage.

    Returns a 0..1 strength, or None when none of the three could be read.
    Deliberately excludes P/E — `value` reads it as an earnings yield, and a
    cheap price is not evidence that a business is well run.
    """
    strengths: list[float] = []

    margin = fundamentals.get("profit_margin")
    if margin is not None:
        strengths.append(_clamp01(float(margin) / 0.30))

    growth = fundamentals.get("revenue_growth")
    if growth is not None:
        strengths.append(_clamp01(0.5 + float(growth) / 0.4))

    dte = fundamentals.get("debt_to_equity")
    if dte is not None and dte >= 0:
        strengths.append(_clamp01(1.0 - float(dte) / 200.0))

    if not strengths:
        return None
    return sum(strengths) / len(strengths)


def _score_risk(
    closes: Any,
    rates: PriceSeries | None,
    sentiment: float | None,
    metrics: dict[str, Any],
) -> list[SubSignal]:
    """Market risk — one of Quality's three parts. Lower risk scores higher."""
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

    # Each measure is inverted against a scale where the "bad" end maps to 0.
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

    # Rate sensitivity. A holding that tracks bond yields carries a macro
    # exposure the other risk measures cannot see — a low-volatility REIT and a
    # low-volatility staples name look alike on drawdown and deviation, and
    # behave nothing alike when yields move.
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
        # No rates proxy ingested yet, or too little history to correlate.
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

    # News tone. A hazard the price series has not necessarily shown yet, which
    # is why it sits with the risk measures rather than anywhere that would imply
    # good press is a reason to buy.
    metrics["news_sentiment"] = sentiment
    if sentiment is None:
        # No headlines, no coverage, or a sweep that has not reached this name.
        # Drops out rather than scoring zero — most of a UK-tradable catalogue
        # will never have news coverage, and scoring that absence would rank
        # companies by how famous they are.
        signals.append(SubSignal("news_sentiment", False))
    else:
        strength = _clamp01(0.5 + sentiment / (2.0 * SENTIMENT_SATURATION))
        signals.append(
            SubSignal(
                "news_sentiment",
                True,
                strength,
                f"news tone is {sentiment:+.2f}",
                positive=strength >= 0.5,
            )
        )
    return signals


def _score_liquidity(closes: Any, volumes: Any, metrics: dict[str, Any]) -> list[SubSignal]:
    """Tradability — one of Quality's three parts."""
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


def _score_sector(sector: PriceSeries | None, metrics: dict[str, Any]) -> list[SubSignal]:
    """Health of the instrument's own sector, via its sector-ETF proxy (§6).

    Rewards being in an industry that is itself in favour: the sector index above
    its 200-day average, that average rising, and positive medium-term sector
    momentum.

    Momentum is scored *here* and nowhere else, and that is not a contradiction
    of the module docstring. A sector ETF's trend is a slow, structural fact
    about an industry — it does not flip week to week the way a single stock's
    one-month return does — so it survives the scanner's rotation staleness in a
    way an individual name's momentum does not.

    Entirely dropped (never penalised) when the instrument has no sector tag or
    its sector has no proxy series.
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


# -- Reported, never scored -------------------------------------------------


def _report_trend_and_momentum(
    closes: Any,
    benchmark: PriceSeries | None,
    sector: PriceSeries | None,
    metrics: dict[str, Any],
) -> None:
    """Compute the trend and momentum readings into `metrics` without scoring.

    These are free — the candles are already loaded — and a human reading the
    results table wants to know whether a cheap stock is falling or steadying.
    But they decide nothing here, for the reason in the module docstring: the
    scanner's rotation makes a one-month return stale before it is compared with
    a fresh one, and the mean-reversion strategy reads the same facts nightly
    against candles from last night. `sma200_slope` in particular is now the
    strategy's falling-knife entry gate.

    `beta_vs_benchmark` set the precedent: measured, reported, never scored,
    because a high beta is not by itself good or bad.
    """
    metrics["sma50"] = ind.simple_moving_average(closes, 50)
    metrics["sma200"] = ind.simple_moving_average(closes, 200)
    metrics["sma200_slope"] = ind.sma_slope(closes, 200, slope_window=21)

    for label, days in (
        ("1m", ind.TRADING_DAYS_PER_MONTH),
        ("3m", ind.TRADING_DAYS_PER_MONTH * 3),
        ("6m", ind.TRADING_DAYS_PER_MONTH * 6),
        ("12m", ind.TRADING_DAYS_PER_YEAR),
    ):
        metrics[f"return_{label}"] = ind.trailing_return(closes, days)

    if benchmark is not None:
        metrics["beta_vs_benchmark"] = ind.beta(
            ind.daily_returns(closes),
            ind.daily_returns(benchmark.preferred_close),
            RATE_CORRELATION_WINDOW,
        )
        metrics["relative_momentum_12m"] = ind.relative_momentum(
            closes, benchmark.preferred_close, ind.TRADING_DAYS_PER_YEAR
        )
    if sector is not None:
        metrics["relative_momentum_vs_sector_12m"] = ind.relative_momentum(
            closes, sector.preferred_close, ind.TRADING_DAYS_PER_YEAR
        )
