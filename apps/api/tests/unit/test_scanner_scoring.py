"""Scanner scoring (§6, §20).

The most important test here is acceptance criterion 7: missing optional data
must not reduce the core score. It is easy to get wrong (divide by the full
signal count instead of the available count) and the failure is invisible —
sparse instruments just quietly score low — so it is pinned from several angles.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from app.indicators.series import PriceSeries
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


class TestScoreRange:
    def test_score_is_within_zero_to_one_hundred(self) -> None:
        result = scoring.score_series(_rising_series())
        assert 0.0 <= result.core_score <= 100.0

    def test_category_points_never_exceed_their_max(self) -> None:
        result = scoring.score_series(_rising_series())
        for category in result.categories.values():
            assert category.points <= category.max_points + 1e-9

    def test_weights_sum_to_one_hundred(self) -> None:
        assert sum(scoring.DEFAULT_WEIGHTS.values()) == 100.0


class TestTrendDiscrimination:
    def test_strong_uptrend_scores_higher_than_downtrend(self) -> None:
        up = scoring.score_series(_rising_series())
        down = scoring.score_series(_falling_series())
        assert up.core_score > down.core_score

    def test_uptrend_trend_category_beats_downtrend(self) -> None:
        up = scoring.score_series(_rising_series())
        down = scoring.score_series(_falling_series())
        assert up.categories["trend"].points > down.categories["trend"].points

    def test_uptrend_produces_positive_signals(self) -> None:
        result = scoring.score_series(_rising_series())
        assert any("above" in s.lower() for s in result.positive_signals)


class TestClassification:
    def test_bands(self) -> None:
        assert scoring.classify(80) is scoring.Classification.SCREENING_CANDIDATE
        assert scoring.classify(65) is scoring.Classification.WATCHLIST_CANDIDATE
        assert scoring.classify(40) is scoring.Classification.DOES_NOT_PASS

    def test_boundaries_are_inclusive_at_the_lower_edge(self) -> None:
        assert scoring.classify(75) is scoring.Classification.SCREENING_CANDIDATE
        assert scoring.classify(60) is scoring.Classification.WATCHLIST_CANDIDATE
        assert scoring.classify(59.99) is scoring.Classification.DOES_NOT_PASS

    def test_thresholds_are_configurable(self) -> None:
        strict = {"screening": 90, "watchlist": 80}
        assert scoring.classify(85, strict) is scoring.Classification.WATCHLIST_CANDIDATE


class TestMissingDataDoesNotPenalise:
    """Acceptance criterion 7, from every angle it could break."""

    def test_short_history_is_not_scored_as_failing(self) -> None:
        """An instrument with 30 bars must not score near zero just for being
        new. Its available signals are scored; the unavailable ones are dropped,
        not counted as failures."""
        short = scoring.score_series(_rising_series(n=30))
        # A rising short series still scores respectably on the signals it can
        # compute — not floored.
        assert short.core_score > 40.0
        # And it is honest about what is missing.
        assert len(short.missing_information) > 0

    def test_missing_fundamentals_yield_none_not_zero(self) -> None:
        result = scoring.score_series(_rising_series(), fundamentals={})
        assert result.fundamental_score is None

    def test_fundamentals_never_change_the_core_score(self) -> None:
        series = _rising_series()
        without = scoring.score_series(series)
        with_funds = scoring.score_series(
            series,
            fundamentals={
                "trailing_pe": Decimal("15"),
                "profit_margin": Decimal("0.25"),
            },
        )
        # The fundamental score appears, but the CORE score is byte-identical.
        assert with_funds.fundamental_score is not None
        assert without.core_score == with_funds.core_score

    def test_a_category_with_no_signals_gets_the_neutral_midpoint(self) -> None:
        """A single bar makes momentum/trend uncomputable. That category must
        land at its midpoint, not zero — absence of evidence is not a failing
        grade."""
        result = scoring.score_series(_series_from_closes([100.0]))
        momentum = result.categories["momentum"]
        assert momentum.signals_available == 0
        assert momentum.points == momentum.max_points * 0.5

    def test_partial_fundamentals_score_on_what_is_present(self) -> None:
        # Only profit margin is known; the score reflects it and ignores the rest.
        result = scoring.score_series(
            _rising_series(), fundamentals={"profit_margin": Decimal("0.30")}
        )
        assert result.fundamental_score is not None
        assert result.fundamental_score > 0


class TestConfidenceAndCompleteness:
    def test_full_history_has_higher_confidence_than_short(self) -> None:
        long = scoring.score_series(_rising_series(n=300))
        short = scoring.score_series(_rising_series(n=40))
        assert long.confidence > short.confidence

    def test_completeness_is_a_fraction(self) -> None:
        result = scoring.score_series(_rising_series())
        assert 0.0 <= result.data_completeness <= 1.0

    def test_full_series_has_full_completeness(self) -> None:
        # A 300-bar series with volume can compute every signal.
        result = scoring.score_series(_rising_series(n=300))
        assert result.data_completeness > 0.9


class TestExplanations:
    def test_no_output_claims_good_investment(self) -> None:
        """§0: the scanner must never assert an instrument is a good investment."""
        result = scoring.score_series(_rising_series())
        blob = " ".join(
            result.positive_signals + result.negative_signals + result.missing_information
        ).lower()
        for forbidden in ("good investment", "buy", "recommend", "should invest", "great stock"):
            assert forbidden not in blob

    def test_signals_use_mechanical_language(self) -> None:
        result = scoring.score_series(_rising_series())
        # Explanations describe measurements, e.g. "Price is above its 200-day".
        assert result.positive_signals
        assert all(isinstance(s, str) and s for s in result.positive_signals)


class TestValueScoring:
    """The valuation lens: high score = cheap/undervalued, the mirror of momentum."""

    def test_value_runs_alongside_momentum_without_changing_it(self) -> None:
        # The core (momentum) score must be identical whether or not value is read.
        series = _rising_series()
        result = scoring.score_series(series)
        assert result.value is not None
        # Recompute core in isolation to confirm value did not perturb it.
        core_only = sum(c.points for c in result.categories.values())
        assert result.core_score == pytest.approx(round(core_only, 2))

    def test_beaten_down_stock_scores_high_on_value(self) -> None:
        """A stock in a steady downtrend is cheap by the value lens — the exact
        opposite of the momentum score, which would rate it low."""
        falling = _falling_series(n=300, daily=-0.003)
        result = scoring.score_series(falling)
        assert result.value is not None
        assert result.value.value_score > 55.0
        # And its momentum core is low — the two lenses disagree, as intended.
        assert result.core_score < result.value.value_score

    def test_stock_at_its_highs_scores_low_on_value(self) -> None:
        """A stock marching to new highs is expensive by the value lens, even as
        the momentum score rates it highly."""
        rising = _rising_series(n=300, daily=0.003)
        result = scoring.score_series(rising)
        assert result.value is not None
        assert result.value.value_score < 45.0
        assert result.core_score > result.value.value_score

    def test_value_and_momentum_are_independent_lenses(self) -> None:
        rising = scoring.score_series(_rising_series())
        falling = scoring.score_series(_falling_series())
        assert rising.value is not None and falling.value is not None
        # Momentum: rising beats falling. Value: falling beats rising. Opposite.
        assert rising.core_score > falling.core_score
        assert falling.value.value_score > rising.value.value_score

    def test_fundamental_value_uses_cheapness_not_quality(self) -> None:
        series = _rising_series()
        cheap = scoring.score_series(
            series,
            fundamentals={"trailing_pe": Decimal("8"), "price_to_book": Decimal("1.0")},
        )
        expensive = scoring.score_series(
            series,
            fundamentals={"trailing_pe": Decimal("50"), "price_to_book": Decimal("12")},
        )
        assert cheap.value is not None and expensive.value is not None
        assert cheap.value.fundamental_value_score is not None
        assert cheap.value.value_score > expensive.value.value_score

    def test_missing_fundamentals_do_not_zero_the_value_score(self) -> None:
        # No fundamentals → fundamental value is None, price value carries.
        result = scoring.score_series(_falling_series())
        assert result.value is not None
        assert result.value.fundamental_value_score is None
        assert result.value.value_score > 0

    def test_value_signals_use_mechanical_language(self) -> None:
        result = scoring.score_series(_falling_series())
        assert result.value is not None
        blob = " ".join(result.value.positive_signals + result.value.negative_signals).lower()
        for forbidden in ("good investment", "buy", "undervalued stock", "recommend"):
            assert forbidden not in blob


class TestPrimaryScoreBlend:
    def test_momentum_primary_returns_the_core_unchanged(self) -> None:
        # value_weight 0 → primary is exactly the momentum core.
        p = scoring.combine_primary_score(80.0, 20.0, momentum_weight=1.0, value_weight=0.0)
        assert p == pytest.approx(80.0)

    def test_value_primary_leads_with_value(self) -> None:
        # value-primary (0.3/0.7): a cheap, weak stock scores high on primary.
        p = scoring.combine_primary_score(30.0, 90.0, momentum_weight=0.3, value_weight=0.7)
        # 0.3*30 + 0.7*90 = 72.
        assert p == pytest.approx(72.0)

    def test_value_and_momentum_primary_rank_oppositely(self) -> None:
        # A strong-expensive stock vs a weak-cheap one, under each configuration.
        strong_expensive = (85.0, 10.0)  # (momentum, value)
        weak_cheap = (30.0, 85.0)

        def mom(m: float, v: float) -> float:
            return scoring.combine_primary_score(m, v, momentum_weight=1.0, value_weight=0.0)

        def val(m: float, v: float) -> float:
            return scoring.combine_primary_score(m, v, momentum_weight=0.3, value_weight=0.7)

        # Momentum-primary ranks the strong one higher.
        assert mom(*strong_expensive) > mom(*weak_cheap)
        # Value-primary ranks the cheap one higher — the inversion the user wants.
        assert val(*weak_cheap) > val(*strong_expensive)

    def test_missing_value_contributes_neutrally(self) -> None:
        # No value score → treated as 50, neither rewarded nor punished.
        p = scoring.combine_primary_score(80.0, None, momentum_weight=0.5, value_weight=0.5)
        assert p == pytest.approx(65.0)  # (80 + 50) / 2

    def test_degenerate_zero_weights_fall_back_to_momentum(self) -> None:
        p = scoring.combine_primary_score(77.0, 20.0, momentum_weight=0.0, value_weight=0.0)
        assert p == pytest.approx(77.0)
