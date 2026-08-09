"""Scanner scoring (§6, §20).

The most important test here is acceptance criterion 7: missing optional data
must not reduce the score. It is easy to get wrong in two different places —
dividing a group by its full signal count instead of its available count, or
dividing the blend by the full 100 instead of the weight that actually scored —
and both failures are invisible, since sparse instruments just quietly rank low.
Both are pinned from several angles.

Two structural guards also live here, because the design they protect is easy to
erode one convenient addition at a time: no fact may be scored by two groups
(`TestNoInversions`), and no momentum reading may be scored at all
(`TestMomentumIsNotScored`).
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from app.indicators.series import PriceSeries
from app.models.scanner import Classification
from app.scanner import scoring


def _series_from_closes(closes: list[float], volume: float = 500_000) -> PriceSeries:
    arr = np.array(closes, dtype=np.float64)
    n = arr.size
    return PriceSeries(
        open=arr,
        high=arr * 1.01,
        low=arr * 0.99,
        close=arr,
        adjusted_close=np.full(n, np.nan),
        volume=np.full(n, volume),
    )


def _rising_series(n: int = 300, start: float = 100.0, daily: float = 0.001) -> PriceSeries:
    closes = [start * (1 + daily) ** i for i in range(n)]
    return _series_from_closes(closes)


def _falling_series(n: int = 300, start: float = 100.0, daily: float = -0.002) -> PriceSeries:
    closes = [start * (1 + daily) ** i for i in range(n)]
    return _series_from_closes(closes)


def _series_from_returns(pattern: list[float], n: int, start: float = 100.0) -> PriceSeries:
    """A series whose daily returns cycle through `pattern`.

    Lets a test state a correlation exactly rather than approximately: two
    repeating patterns over a whole number of cycles correlate to a known value.
    """
    closes = [start]
    for i in range(n):
        closes.append(closes[-1] * (1 + pattern[i % len(pattern)]))
    return _series_from_closes(closes)


def _rate_signal(closes: np.ndarray, rates: PriceSeries) -> scoring.SubSignal:
    """The `rate_sensitivity` sub-signal alone, out of the risk readings."""
    signals = scoring._score_risk(closes, rates, None, {})
    return next(s for s in signals if s.name == "rate_sensitivity")


def _sentiment_signal(polarity: float) -> scoring.SubSignal:
    """The `news_sentiment` sub-signal alone, out of the risk readings."""
    closes = _rising_series().preferred_close
    signals = scoring._score_risk(closes, None, polarity, {})
    return next(s for s in signals if s.name == "news_sentiment")


CHEAP_FUNDAMENTALS: dict[str, Decimal | None] = {
    "trailing_pe": Decimal("8"),
    "price_to_book": Decimal("0.9"),
    "profit_margin": Decimal("0.22"),
    "revenue_growth": Decimal("0.15"),
    "earnings_growth": Decimal("0.20"),
    "debt_to_equity": Decimal("20"),
    "dividend_yield": Decimal("0.05"),
}

EXPENSIVE_FUNDAMENTALS: dict[str, Decimal | None] = {
    "trailing_pe": Decimal("60"),
    "price_to_book": Decimal("9"),
    "profit_margin": Decimal("0.01"),
    "revenue_growth": Decimal("-0.10"),
    "earnings_growth": Decimal("0.01"),
    "debt_to_equity": Decimal("300"),
    "dividend_yield": Decimal("0"),
}


def _group(name: str, score: float | None, weight: float) -> scoring.GroupScore:
    return scoring.GroupScore(name=name, score=score, weight=weight)


def _full_groups(**overrides: float | None) -> dict[str, scoring.GroupScore]:
    """Five groups all scoring 60 unless overridden, at the default weights."""
    scores: dict[str, float | None] = dict.fromkeys(scoring.GROUP_NAMES, 60.0)
    scores.update(overrides)
    return {
        name: _group(name, scores[name], scoring.DEFAULT_WEIGHTS[name])
        for name in scoring.GROUP_NAMES
    }


# -- Shape ------------------------------------------------------------------


class TestScoreRange:
    def test_score_is_within_zero_to_one_hundred(self) -> None:
        result = scoring.score_series(_rising_series())
        assert 0.0 <= result.score <= 100.0

    def test_weights_sum_to_one_hundred(self) -> None:
        assert sum(scoring.DEFAULT_WEIGHTS.values()) == pytest.approx(100.0)

    def test_there_are_exactly_five_groups(self) -> None:
        assert set(scoring.DEFAULT_WEIGHTS) == set(scoring.GROUP_NAMES)
        result = scoring.score_series(_rising_series())
        assert set(result.groups) == set(scoring.GROUP_NAMES)

    def test_every_group_score_is_zero_to_one_hundred_or_none(self) -> None:
        result = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        for group in result.groups.values():
            assert group.score is None or 0.0 <= group.score <= 100.0, group.name

    def test_extremes_do_not_escape_the_scale(self) -> None:
        """A stock that is bad on every measurable axis still lands inside 0-100."""
        result = scoring.score_series(
            _rising_series(),  # near its high, so cheapness scores badly
            fundamentals=EXPENSIVE_FUNDAMENTALS,
            insider=0.0,
            insider_sell_penalty=0.40,
        )
        assert 0.0 <= result.score <= 100.0


class TestWeightBalance:
    """The tuning decisions behind DEFAULT_WEIGHTS, stated as behaviour.

    Pinning the constants themselves would just restate them. What is worth
    protecting is what they were chosen to *do*, so that a future retune has to
    break a claim about outcomes rather than only change a number.
    """

    def test_soundness_counts_for_at_least_as_much_as_price_level(self) -> None:
        """Retuned 2026-08-09: cheapness 30 → 24, quality 13 → 18.

        The old balance let a cheap price carry a deteriorating business a long
        way up the ranking. Quality is now weighted at least as heavily as
        cheapness, which is the whole point of that change.
        """
        assert scoring.DEFAULT_WEIGHTS["quality"] >= scoring.DEFAULT_WEIGHTS["cheapness"] * 0.7

    def test_a_value_trap_no_longer_runs_far_ahead_of_a_sound_business(self) -> None:
        """A cheap, deteriorating stock against a dearer, sound one.

        The trap is still allowed to win — it *is* cheaper, and this is a value
        screen. What the weights control is by how much. Under the pre-retune
        balance the gap was +13.8 points, which is more than a whole
        classification band.
        """
        trap = {"value": 80.0, "cheapness": 90.0, "insider": None, "quality": 35.0, "sector": 50.0}
        solid = {"value": 60.0, "cheapness": 45.0, "insider": None, "quality": 85.0, "sector": 70.0}
        gap = scoring.combine_score(_full_groups(**trap)) - scoring.combine_score(
            _full_groups(**solid)
        )
        assert 0 < gap < 10.0

    def test_insider_is_the_heaviest_single_measurement(self) -> None:
        """Its group holds exactly one lookup, so its weight is undiluted.

        Worth stating because it is easy to miss from the weight table alone: at
        18 the insider group is only the joint-second *group*, but it is by some
        way the heaviest single *measurement* — value's 30 is split five ways.
        """
        per_measurement = {
            "value": scoring.DEFAULT_WEIGHTS["value"] / 5,
            "cheapness": scoring.DEFAULT_WEIGHTS["cheapness"] / 3,
            "insider": scoring.DEFAULT_WEIGHTS["insider"] / 1,
            "sector": scoring.DEFAULT_WEIGHTS["sector"] / 4,
        }
        assert max(per_measurement, key=lambda k: per_measurement[k]) == "insider"


class TestClassification:
    def test_bands(self) -> None:
        assert scoring.classify(80.0) is Classification.SCREENING_CANDIDATE
        assert scoring.classify(65.0) is Classification.WATCHLIST_CANDIDATE
        assert scoring.classify(40.0) is Classification.DOES_NOT_PASS

    def test_boundaries_are_inclusive_at_the_lower_edge(self) -> None:
        assert scoring.classify(75.0) is Classification.SCREENING_CANDIDATE
        assert scoring.classify(60.0) is Classification.WATCHLIST_CANDIDATE
        assert scoring.classify(59.99) is Classification.DOES_NOT_PASS

    def test_thresholds_are_configurable(self) -> None:
        loose = {"screening": 50.0, "watchlist": 30.0}
        assert scoring.classify(55.0, loose) is Classification.SCREENING_CANDIDATE

    def test_the_result_classifies_on_its_own_score(self) -> None:
        """There is one score, so there can only be one classification.

        The old two-score design computed a classification from the momentum
        core and then threw it away, while the engine recomputed a different one
        from the blend. Anything reading `result.classification` was reading the
        discarded answer.
        """
        result = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        assert result.classification is scoring.classify(result.score)


# -- Acceptance criterion 7 -------------------------------------------------


class TestMissingDataDoesNotPenalise:
    def test_unavailable_signals_leave_their_group_average_alone(self) -> None:
        """Three available signals at 0.5 score 50, whether or not two are missing."""
        present = [scoring.SubSignal(f"s{i}", True, 0.5) for i in range(3)]
        padded = [*present, scoring.SubSignal("gone_a", False), scoring.SubSignal("gone_b", False)]
        assert scoring._group("g", present, 30.0).score == pytest.approx(50.0)
        assert scoring._group("g", padded, 30.0).score == pytest.approx(50.0)

    def test_a_group_with_nothing_available_scores_none_not_zero(self) -> None:
        empty = [scoring.SubSignal("a", False), scoring.SubSignal("b", False)]
        assert scoring._group("g", empty, 30.0).score is None

    def test_a_missing_group_shrinks_the_divisor(self) -> None:
        """The single most important line in the module.

        Four groups at 60 with the fifth absent must still score 60 — not
        60 x (weight that scored / 100), which is what dividing by the full
        weight would give.
        """
        groups = _full_groups(insider=None)
        assert scoring.combine_score(groups) == pytest.approx(60.0)

    def test_a_missing_group_is_not_scored_as_a_neutral_fifty(self) -> None:
        """Absence must be *removed*, not replaced with a midpoint.

        A neutral 50 would drag every high scorer down and lift every low one,
        and since ~60% of a tradable catalogue can never carry Form 4 data, the
        drag would fall almost entirely on non-US listings.
        """
        absent = scoring.combine_score(_full_groups(insider=None))
        neutral = scoring.combine_score(_full_groups(insider=50.0))
        assert absent == pytest.approx(60.0)
        assert neutral < absent

    def test_every_group_missing_scores_zero_rather_than_dividing_by_zero(self) -> None:
        nothing = {name: _group(name, None, 20.0) for name in scoring.GROUP_NAMES}
        assert scoring.combine_score(nothing) == 0.0

    def test_absence_lowers_confidence_not_score(self) -> None:
        rich = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        sparse = scoring.score_series(_rising_series())
        assert sparse.confidence < rich.confidence
        assert sparse.data_completeness < rich.data_completeness

    def test_missing_signals_are_named_not_silent(self) -> None:
        result = scoring.score_series(_rising_series())
        assert "earnings_yield" in result.missing_information
        assert "insider_activity" in result.missing_information


class TestPenalties:
    def test_no_fundamentals_costs_a_tenth(self) -> None:
        groups = _full_groups(value=None)
        with_penalty = scoring.combine_score(groups, fundamentals_penalty=0.10)
        without = scoring.combine_score(groups, fundamentals_penalty=0.0)
        assert with_penalty == pytest.approx(without * 0.90, rel=1e-3)

    def test_the_penalty_does_not_apply_when_value_scored(self) -> None:
        groups = _full_groups()
        assert scoring.combine_score(groups, fundamentals_penalty=0.10) == pytest.approx(60.0)

    def test_insider_selling_discounts_the_whole_score(self) -> None:
        groups = _full_groups()
        assert scoring.combine_score(groups, insider_sell_penalty=0.40) == pytest.approx(36.0)

    def test_the_sell_penalty_is_clamped(self) -> None:
        groups = _full_groups()
        assert scoring.combine_score(groups, insider_sell_penalty=5.0) == 0.0
        assert scoring.combine_score(groups, insider_sell_penalty=-1.0) == pytest.approx(60.0)


# -- Structural guards ------------------------------------------------------


def _all_scored_signal_names() -> list[str]:
    """Every signal name that contributes to the score, across all groups."""
    series = _rising_series()
    closes = series.preferred_close
    names = [
        *(s.name for s in scoring._score_value(CHEAP_FUNDAMENTALS)),
        *(s.name for s in scoring._score_cheapness(closes, {})),
        *(s.name for s in scoring._score_sector(series, {})),
        *(s.name for s in scoring._score_risk(closes, None, None, {})),
        *(s.name for s in scoring._score_liquidity(closes, series.volume, {})),
        "insider_buying",
    ]
    return names


class TestNoInversions:
    def test_no_fact_is_scored_by_two_groups(self) -> None:
        """The guard against reintroducing the old design's contradictions.

        The two-score scanner read the same fact in opposite directions four
        times over — `price_above_sma200` against `below_200d_average`,
        `distance_from_52w_high` against `pullback_from_high`, and so on. Every
        group now shares one orientation (cheap and sound scores high), which is
        only sustainable if each measurement appears exactly once.
        """
        names = _all_scored_signal_names()
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, f"scored twice: {sorted(duplicates)}"

    def test_the_rsi_level_is_not_scored(self) -> None:
        """It is the strategy's entry condition, checked nightly on fresh candles.

        Scoring it here as well would count one fact in two layers, and the
        scanner's copy would be up to 100 days stale by the time it was compared
        against a freshly-scanned name.
        """
        assert not any("rsi" in n for n in _all_scored_signal_names())

    def test_pe_is_read_as_an_earnings_yield_and_nowhere_else(self) -> None:
        """`value` prices it; `quality` must not also reward it as soundness."""
        quality_inputs = scoring._quality_fundamentals(CHEAP_FUNDAMENTALS)
        without_pe = scoring._quality_fundamentals(
            {**CHEAP_FUNDAMENTALS, "trailing_pe": Decimal("60")}
        )
        assert quality_inputs == without_pe


class TestMomentumIsNotScored:
    #: Everything the old momentum and trend categories measured.
    MOMENTUM_NAMES = frozenset(
        {
            "price_above_sma50",
            "price_above_sma200",
            "sma50_above_sma200",
            "sma200_slope",
            "return_1m",
            "return_3m",
            "return_6m",
            "return_12m",
            "relative_momentum_12m",
            "relative_momentum_vs_sector",
            "earnings_drift",
        }
    )

    def test_no_momentum_signal_contributes_to_the_score(self) -> None:
        assert self.MOMENTUM_NAMES.isdisjoint(_all_scored_signal_names())

    def test_a_rising_and_a_falling_stock_score_the_same_on_quality(self) -> None:
        """Trend direction must not leak into the score through a side door."""
        rising = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        falling = scoring.score_series(_falling_series(), fundamentals=CHEAP_FUNDAMENTALS)
        # Cheapness legitimately differs — the falling stock *is* cheaper — but
        # nothing in the value group reads price at all.
        assert rising.group_score("value") == falling.group_score("value")

    def test_momentum_is_still_reported_as_a_metric(self) -> None:
        """Unscored is not unmeasured: the results table still shows the trend."""
        result = scoring.score_series(_rising_series())
        for key in ("sma50", "sma200", "sma200_slope", "return_1m", "return_12m"):
            assert key in result.metrics, key
        assert result.metrics["sma200_slope"] is not None

    def test_the_sector_trend_is_scored_and_that_is_deliberate(self) -> None:
        """An industry's trend is a slow structural fact, not a fast one.

        It survives the scanner's rotation staleness in a way a single stock's
        one-month return does not, which is why it is the one trend reading that
        still earns a score.
        """
        names = [s.name for s in scoring._score_sector(_rising_series(), {})]
        assert "sector_trend_rising" in names


# -- Individual groups ------------------------------------------------------


class TestValueGroup:
    def test_cheap_fundamentals_score_above_expensive_ones(self) -> None:
        cheap = scoring._group("value", scoring._score_value(CHEAP_FUNDAMENTALS), 32.0)
        dear = scoring._group("value", scoring._score_value(EXPENSIVE_FUNDAMENTALS), 32.0)
        assert cheap.score is not None and dear.score is not None
        assert cheap.score > dear.score

    def test_no_fundamentals_means_no_score(self) -> None:
        assert scoring._group("value", scoring._score_value(None), 32.0).score is None

    def test_partial_fundamentals_still_score(self) -> None:
        partial: dict[str, Decimal | None] = {"dividend_yield": Decimal("0.04")}
        group = scoring._group("value", scoring._score_value(partial), 32.0)
        assert group.score == pytest.approx(100.0)
        assert group.signals_available == 1

    def test_graham_needs_both_ratios(self) -> None:
        only_pe: dict[str, Decimal | None] = {"trailing_pe": Decimal("10")}
        signals = {s.name: s for s in scoring._score_value(only_pe)}
        assert signals["earnings_yield"].available
        assert not signals["graham_margin_of_safety"].available

    def test_a_negative_pe_is_not_treated_as_cheap(self) -> None:
        """A loss-making company has no earnings yield, not an enormous one."""
        loss: dict[str, Decimal | None] = {"trailing_pe": Decimal("-5")}
        signals = {s.name: s for s in scoring._score_value(loss)}
        assert not signals["earnings_yield"].available


class TestCheapnessGroup:
    def test_a_pulled_back_stock_scores_above_one_at_its_high(self) -> None:
        at_high = _rising_series()
        pulled_back = _series_from_closes(
            [100.0 * (1.001**i) for i in range(250)] + [70.0 - i * 0.05 for i in range(50)]
        )
        high_score = scoring._group(
            "c", scoring._score_cheapness(at_high.preferred_close, {}), 30.0
        )
        back_score = scoring._group(
            "c", scoring._score_cheapness(pulled_back.preferred_close, {}), 30.0
        )
        assert high_score.score is not None and back_score.score is not None
        assert back_score.score > high_score.score

    def test_it_reads_three_things(self) -> None:
        names = [s.name for s in scoring._score_cheapness(_rising_series().preferred_close, {})]
        assert names == ["pullback_from_high", "low_in_range", "below_200d_average"]

    def test_a_short_series_drops_the_200_day_signal(self) -> None:
        short = _series_from_closes([100.0 + i for i in range(40)])
        signals = {s.name: s for s in scoring._score_cheapness(short.preferred_close, {})}
        assert not signals["below_200d_average"].available
        assert signals["pullback_from_high"].available


class TestQualityGroup:
    def test_it_is_a_three_part_mean_not_a_flat_average(self) -> None:
        """Flattening would give the seven risk signals half the group.

        With business soundness at 1.0 and both candle-derived parts at 0.0, a
        three-part mean is 1/3. A flat average over 3 + 7 + 4 signals would be
        3/14, which is a materially more risk-driven group than the one that was
        tuned.
        """
        risk = [scoring.SubSignal(f"r{i}", True, 0.0) for i in range(7)]
        liquidity = [scoring.SubSignal(f"l{i}", True, 0.0) for i in range(4)]
        perfect: dict[str, Decimal | None] = {
            "profit_margin": Decimal("0.30"),
            "revenue_growth": Decimal("0.20"),
            "debt_to_equity": Decimal("0"),
        }
        group = scoring._quality_group(perfect, risk, liquidity, 13.0)
        assert group.score == pytest.approx(100.0 / 3.0)

    def test_it_still_scores_without_fundamentals(self) -> None:
        risk = [scoring.SubSignal("r", True, 0.6)]
        liquidity = [scoring.SubSignal("l", True, 0.8)]
        group = scoring._quality_group(None, risk, liquidity, 13.0)
        assert group.score == pytest.approx(70.0)

    def test_it_carries_no_signals_of_its_own(self) -> None:
        """Risk and liquidity are tallied once, by `score_series`.

        Attaching them to the quality group as well would double-count them in
        the completeness figure and in the explanation lists.
        """
        result = scoring.score_series(_rising_series())
        assert result.groups["quality"].signals == []


class TestInsiderGroup:
    def test_a_buy_score_becomes_the_group_score(self) -> None:
        result = scoring.score_series(_rising_series(), insider=82.0)
        assert result.group_score("insider") == pytest.approx(82.0)

    def test_no_filings_means_no_score(self) -> None:
        result = scoring.score_series(_rising_series(), insider=None)
        assert result.group_score("insider") is None

    def test_selling_is_a_penalty_not_a_group_score(self) -> None:
        clean = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        sold = scoring.score_series(
            _rising_series(), fundamentals=CHEAP_FUNDAMENTALS, insider_sell_penalty=0.40
        )
        assert sold.score < clean.score
        assert sold.group_score("insider") is None


class TestSectorGroup:
    def test_a_healthy_sector_scores_above_a_weak_one(self) -> None:
        strong = scoring._group("s", scoring._score_sector(_rising_series(), {}), 9.0)
        weak = scoring._group("s", scoring._score_sector(_falling_series(), {}), 9.0)
        assert strong.score is not None and weak.score is not None
        assert strong.score > weak.score

    def test_an_untagged_instrument_has_no_sector_score(self) -> None:
        """None, not a midpoint — the group drops out with its weight."""
        result = scoring.score_series(_rising_series(), sector=None)
        assert result.group_score("sector") is None

    def test_the_sector_group_is_on_the_same_scale_as_the_others(self) -> None:
        """It used to store category *points* (max 20) beside 0-100 columns.

        An untagged stock showed 10.0 in a results table where every neighbour
        ran to 100, which read as a very low sector score rather than as no
        sector score at all.
        """
        result = scoring.score_series(_rising_series(), sector=_rising_series())
        score = result.group_score("sector")
        assert score is not None and 0.0 <= score <= 100.0
        assert score > 20.0


# -- Risk readings (Quality's market-risk part) -----------------------------


class TestRateSensitivity:
    def test_an_uncorrelated_instrument_scores_high(self) -> None:
        closes = _series_from_returns([0.01, -0.01], 240).preferred_close
        rates = _series_from_returns([0.01, 0.01, -0.01, -0.01], 240)
        signal = _rate_signal(closes, rates)
        assert signal.available
        assert signal.value > 0.9

    def test_a_rate_tracking_instrument_scores_low(self) -> None:
        rates = _series_from_returns([0.012, -0.008, 0.004, -0.011], 240)
        signal = _rate_signal(rates.preferred_close, rates)
        assert signal.available
        assert signal.value == pytest.approx(0.0, abs=0.01)

    def test_negative_correlation_is_penalised_just_as_much(self) -> None:
        """Moving hard *against* rates is still a bet on rates."""
        pattern = [0.012, -0.008, 0.004, -0.011]
        rates = _series_from_returns(pattern, 240)
        inverse = _series_from_returns([-r for r in pattern], 240)
        assert _rate_signal(inverse.preferred_close, rates).value == pytest.approx(0.0, abs=0.01)

    def test_no_rates_proxy_drops_the_signal(self) -> None:
        signals = scoring._score_risk(_rising_series().preferred_close, None, None, {})
        rate = next(s for s in signals if s.name == "rate_sensitivity")
        assert not rate.available


class TestNewsSentiment:
    def test_neutral_tone_sits_at_the_midpoint(self) -> None:
        assert _sentiment_signal(0.0).value == pytest.approx(0.5)

    def test_bad_news_scores_low_and_good_news_high(self) -> None:
        assert _sentiment_signal(-0.4).value < 0.5
        assert _sentiment_signal(0.4).value > 0.5

    def test_it_saturates_rather_than_running_to_the_extremes(self) -> None:
        assert _sentiment_signal(-0.5).value == pytest.approx(0.0)
        assert _sentiment_signal(0.5).value == pytest.approx(1.0)
        assert _sentiment_signal(-1.0).value == pytest.approx(0.0)

    def test_no_coverage_drops_the_signal(self) -> None:
        signals = scoring._score_risk(_rising_series().preferred_close, None, None, {})
        news = next(s for s in signals if s.name == "news_sentiment")
        assert not news.available

    def test_its_influence_on_the_score_is_small_by_design(self) -> None:
        """One of seven risk signals, in one of three quality parts, at weight 13.

        Worth roughly 0.6% of the final score — bounded and local, which is the
        whole point of admitting new signals at the group layer.
        """
        series = _rising_series()
        best = scoring.score_series(series, fundamentals=CHEAP_FUNDAMENTALS, sentiment=0.5)
        worst = scoring.score_series(series, fundamentals=CHEAP_FUNDAMENTALS, sentiment=-0.5)
        assert 0.0 < best.score - worst.score < 3.0


# -- Provenance -------------------------------------------------------------


class TestConfidenceAndCompleteness:
    def test_a_full_input_set_is_more_complete_than_a_bare_one(self) -> None:
        bare = scoring.score_series(_series_from_closes([100.0 + i for i in range(40)]))
        full = scoring.score_series(
            _rising_series(),
            benchmark=_rising_series(),
            sector=_rising_series(),
            rates=_rising_series(),
            sentiment=0.1,
            fundamentals=CHEAP_FUNDAMENTALS,
            insider=60.0,
        )
        assert full.data_completeness > bare.data_completeness
        assert full.confidence > bare.confidence

    def test_withholding_an_optional_input_lowers_completeness(self) -> None:
        full = scoring.score_series(
            _rising_series(),
            benchmark=_rising_series(),
            sector=_rising_series(),
            rates=_rising_series(),
            sentiment=0.1,
            fundamentals=CHEAP_FUNDAMENTALS,
        )
        without = scoring.score_series(
            _rising_series(),
            benchmark=_rising_series(),
            sector=_rising_series(),
            rates=None,
            sentiment=0.1,
            fundamentals=CHEAP_FUNDAMENTALS,
        )
        assert without.data_completeness < full.data_completeness

    def test_short_history_lowers_confidence_even_when_complete(self) -> None:
        short = scoring.score_series(
            _series_from_closes([100.0 + i for i in range(60)]),
            fundamentals=CHEAP_FUNDAMENTALS,
        )
        long = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        assert short.confidence < long.confidence

    def test_candles_used_is_reported(self) -> None:
        assert scoring.score_series(_rising_series(n=250)).candles_used == 250


class TestExplanations:
    def test_signals_carry_mechanical_language(self) -> None:
        result = scoring.score_series(_rising_series(), fundamentals=CHEAP_FUNDAMENTALS)
        text = " ".join(result.positive_signals + result.negative_signals).lower()
        for word in ("buy", "sell", "recommend", "should"):
            assert word not in text

    def test_both_sides_are_reported(self) -> None:
        result = scoring.score_series(_rising_series(), fundamentals=EXPENSIVE_FUNDAMENTALS)
        assert result.positive_signals or result.negative_signals
        assert all(isinstance(s, str) and s for s in result.negative_signals)

    def test_no_signal_is_listed_twice(self) -> None:
        result = scoring.score_series(
            _rising_series(), fundamentals=CHEAP_FUNDAMENTALS, sector=_rising_series()
        )
        listed = result.positive_signals + result.negative_signals
        assert len(listed) == len(set(listed))


# -- Absoluteness -----------------------------------------------------------


class TestScoreIsAbsolute:
    def test_score_is_absolute_not_cohort_relative(self) -> None:
        """The same inputs must score the same however the batch is composed.

        Batch-relative ranking would make a rotation over ~20,000 instruments
        meaningless: tonight's 400 would be scored against each other rather than
        against a fixed standard, so a score could not be compared with one from
        a different night.
        """
        series = _rising_series()
        alone = scoring.score_series(series, fundamentals=CHEAP_FUNDAMENTALS)
        for _ in range(5):
            scoring.score_series(_falling_series(), fundamentals=EXPENSIVE_FUNDAMENTALS)
        again = scoring.score_series(series, fundamentals=CHEAP_FUNDAMENTALS)
        assert alone.score == again.score

    def test_scoring_is_deterministic(self) -> None:
        series = _rising_series()
        first = scoring.score_series(series, fundamentals=CHEAP_FUNDAMENTALS, insider=55.0)
        second = scoring.score_series(series, fundamentals=CHEAP_FUNDAMENTALS, insider=55.0)
        assert first.score == second.score
        assert first.data_completeness == second.data_completeness
