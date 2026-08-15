"""Technical indicator correctness.

These use closed-form or hand-computable inputs so the expected value is known,
not merely reproduced from the implementation. An indicator that is subtly wrong
produces a score that looks authoritative and is not — so the bar here is
known-answer, not smoke.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.indicators import functions as ind


class TestMovingAverage:
    def test_sma_of_known_values(self) -> None:
        assert ind.simple_moving_average(np.array([1.0, 2, 3, 4, 5]), 5) == 3.0

    def test_sma_uses_only_the_last_period(self) -> None:
        # Last 3 of [1..5] = [3,4,5] → 4.
        assert ind.simple_moving_average(np.array([1.0, 2, 3, 4, 5]), 3) == 4.0

    def test_sma_none_when_insufficient(self) -> None:
        assert ind.simple_moving_average(np.array([1.0, 2]), 5) is None

    def test_sma_series_is_nan_until_the_window_fills(self) -> None:
        series = ind.sma_series(np.array([1.0, 2, 3, 4]), 2)
        assert np.isnan(series[0])
        assert series[1] == 1.5
        assert series[3] == 3.5


class TestReturns:
    def test_trailing_return_is_simple_ratio(self) -> None:
        # 100 → 110 over 1 bar = +10%.
        assert ind.trailing_return(np.array([100.0, 110.0]), 1) == pytest.approx(0.10)

    def test_trailing_return_spans_the_lookback(self) -> None:
        closes = np.array([100.0, 105, 110, 120])
        # 3 bars back is 100, latest 120 → +20%.
        assert ind.trailing_return(closes, 3) == pytest.approx(0.20)

    def test_trailing_return_none_when_too_short(self) -> None:
        assert ind.trailing_return(np.array([100.0, 110.0]), 5) is None

    def test_daily_returns_length_is_one_less(self) -> None:
        rets = ind.daily_returns(np.array([100.0, 110, 99]))
        assert rets.size == 2
        assert rets[0] == pytest.approx(0.10)
        assert rets[1] == pytest.approx(-0.10)


class TestVolatility:
    def test_zero_volatility_for_constant_prices(self) -> None:
        closes = np.full(30, 100.0)
        assert ind.annualised_volatility(closes, 20) == pytest.approx(0.0)

    def test_volatility_is_annualised(self) -> None:
        # Daily returns alternate +1%/-1%-ish; check it scales by sqrt(252).
        rng = np.random.default_rng(42)
        daily = rng.normal(0, 0.01, 300)
        closes = 100 * np.cumprod(1 + daily)
        vol = ind.annualised_volatility(closes, 250)
        assert vol is not None
        # ~1% daily → ~15.9% annual (0.01 * sqrt(252)).
        assert vol == pytest.approx(0.159, abs=0.03)

    def test_volatility_none_when_insufficient(self) -> None:
        assert ind.annualised_volatility(np.array([100.0, 101]), 20) is None


class TestDownsideDeviation:
    def test_zero_when_no_down_days(self) -> None:
        closes = np.array([100.0 * (1.01**i) for i in range(30)])
        dd = ind.downside_deviation(closes, 20)
        assert dd == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_there_are_losses(self) -> None:
        rng = np.random.default_rng(7)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))
        dd = ind.downside_deviation(closes, 40)
        assert dd is not None and dd > 0


class TestDrawdown:
    def test_known_drawdown(self) -> None:
        # Peak 100, trough 70 → 30% drawdown, returned positive.
        closes = np.array([80.0, 100, 90, 70, 85])
        assert ind.max_drawdown(closes) == pytest.approx(0.30)

    def test_no_drawdown_on_monotonic_rise(self) -> None:
        closes = np.array([1.0, 2, 3, 4, 5])
        assert ind.max_drawdown(closes) == pytest.approx(0.0)

    def test_largest_daily_loss_is_positive_magnitude(self) -> None:
        # Worst single day is 200 → 170 = -15%, returned as 0.15. (The loss is
        # measured against the *previous* close, not any earlier peak.)
        closes = np.array([100.0, 200, 170, 180])
        assert ind.largest_daily_loss(closes) == pytest.approx(0.15)

    def test_largest_daily_loss_zero_when_never_down(self) -> None:
        assert ind.largest_daily_loss(np.array([1.0, 2, 3])) == pytest.approx(0.0)


class TestLiquidity:
    def test_average_traded_value_multiplies_price_and_volume(self) -> None:
        closes = np.array([10.0, 10, 10])
        volumes = np.array([100.0, 200, 300])
        # (1000 + 2000 + 3000) / 3 = 2000.
        assert ind.average_traded_value(closes, volumes, 3) == pytest.approx(2000.0)

    def test_zero_volume_days_counted(self) -> None:
        assert ind.zero_volume_days(np.array([100.0, 0, 50, 0, 0])) == 3

    def test_stale_price_days_counts_unchanged_closes(self) -> None:
        # 100→100 (stale), 100→101, 101→101 (stale) → 2.
        assert ind.stale_price_days(np.array([100.0, 100, 101, 101])) == 2


class TestPositioning:
    def test_distance_from_high_at_the_high_is_zero(self) -> None:
        closes = np.array([90.0, 95, 100])
        assert ind.distance_from_high(closes, 3) == pytest.approx(0.0)

    def test_distance_from_high_below_the_high(self) -> None:
        # High 100, latest 75 → 25% below.
        closes = np.array([100.0, 80, 75])
        assert ind.distance_from_high(closes, 3) == pytest.approx(0.25)

    def test_position_in_range_endpoints(self) -> None:
        # Latest at the high → 1.0; construct low=50, high=100, latest=75 → 0.5.
        assert ind.position_in_range(np.array([100.0, 50, 75]), 3) == pytest.approx(0.5)


class TestATR:
    def test_atr_of_constant_range(self) -> None:
        # Every bar spans exactly 2 (high-low), no gaps → ATR 2.
        highs = np.array([11.0, 11, 11, 11, 11])
        lows = np.array([9.0, 9, 9, 9, 9])
        closes = np.array([10.0, 10, 10, 10, 10])
        assert ind.average_true_range(highs, lows, closes, period=3) == pytest.approx(2.0)

    def test_atr_captures_a_gap(self) -> None:
        # A gap up makes |high - prev_close| exceed the bar's own range.
        highs = np.array([10.0, 10, 20])
        lows = np.array([9.0, 9, 19])
        closes = np.array([9.5, 9.5, 19.5])
        atr = ind.average_true_range(highs, lows, closes, period=2)
        assert atr is not None and atr > 1.0


class TestBollingerBands:
    def test_known_answer_on_a_hand_computable_series(self) -> None:
        # Four values, mean 10, population sd = sqrt(((2)^2+(1)^2+(1)^2+(2)^2)/4)
        # = sqrt(10/4) = 1.5811. Bands at 2 sd: 10 +/- 3.1623.
        closes = np.array([8.0, 9.0, 11.0, 12.0])
        bands = ind.bollinger_bands(closes, period=4, num_std=2.0)
        assert bands is not None
        lower, middle, upper = bands
        assert middle == pytest.approx(10.0)
        assert lower == pytest.approx(10.0 - 3.16227766, rel=1e-6)
        assert upper == pytest.approx(10.0 + 3.16227766, rel=1e-6)

    def test_uses_population_not_sample_deviation(self) -> None:
        """ddof=0, matching Bollinger's definition and every charting package.

        Sample deviation (ddof=1) would widen these bands by ~15% on a 4-bar
        window and disagree with any chart the operator compares against.
        """
        closes = np.array([8.0, 9.0, 11.0, 12.0])
        bands = ind.bollinger_bands(closes, period=4, num_std=1.0)
        assert bands is not None
        lower, _, _ = bands
        assert lower == pytest.approx(10.0 - float(np.std(closes)), rel=1e-9)
        assert lower != pytest.approx(10.0 - float(np.std(closes, ddof=1)), rel=1e-9)

    def test_none_when_history_is_short(self) -> None:
        assert ind.bollinger_bands(np.array([1.0, 2.0]), period=20) is None

    def test_none_on_a_flat_window(self) -> None:
        """Zero width would put price on both edges at once — no opinion, not
        'maximally oversold and overbought simultaneously'."""
        assert ind.bollinger_bands(np.array([5.0] * 30), period=20) is None

    def test_position_maps_the_bands_to_zero_and_one(self) -> None:
        """0.0 sits on the lower band, 1.0 on the upper, 0.5 on the mean."""
        closes = np.array([8.0, 9.0, 11.0, 12.0])
        bands = ind.bollinger_bands(closes, period=4, num_std=2.0)
        assert bands is not None
        lower, middle, upper = bands
        pos = ind.bollinger_position(closes, period=4, num_std=2.0)
        assert pos is not None
        expected = (float(closes[-1]) - lower) / (upper - lower)
        assert pos == pytest.approx(expected)
        # The mean is the midpoint of a symmetric band by construction.
        assert (middle - lower) / (upper - lower) == pytest.approx(0.5)

    def test_position_exceeds_one_when_price_breaks_out(self) -> None:
        """Deliberately unclamped: a close beyond the band is the signal."""
        closes = np.array([*[10.0, 11.0, 9.0, 10.0] * 5, 40.0])
        pos = ind.bollinger_position(closes, period=20, num_std=2.0)
        assert pos is not None and pos > 1.0


class TestRelativeMomentum:
    def test_zero_when_matching_the_benchmark(self) -> None:
        closes = np.array([100.0, 110, 120])
        assert ind.relative_momentum(closes, closes, 2) == pytest.approx(0.0)

    def test_positive_when_outperforming(self) -> None:
        own = np.array([100.0, 120])  # +20%
        bench = np.array([100.0, 110])  # +10%
        assert ind.relative_momentum(own, bench, 1) == pytest.approx(0.10)


class TestCorrelation:
    def test_perfect_positive_correlation(self) -> None:
        a = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        assert ind.rolling_correlation(a, a, 5) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        a = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        assert ind.rolling_correlation(a, -a, 5) == pytest.approx(-1.0)

    def test_none_when_a_series_is_flat(self) -> None:
        a = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        flat = np.zeros(5)
        assert ind.rolling_correlation(a, flat, 5) is None


class TestAnchoredVWAP:
    def test_a_constant_price_series_averages_to_that_price(self) -> None:
        flat = np.full(10, 50.0)
        assert ind.anchored_vwap(flat, flat, flat, np.full(10, 100.0), 0) == pytest.approx(50.0)

    def test_it_weights_by_volume_not_by_bar(self) -> None:
        # Typical price is (h+l+c)/3, so with h=l=c it is the close itself.
        # Bars at 10 and 20, but the 20 traded four times the volume:
        # (10*100 + 20*400) / 500 = 18, not the unweighted 15.
        closes = np.array([10.0, 20.0])
        volumes = np.array([100.0, 400.0])
        assert ind.anchored_vwap(closes, closes, closes, volumes, 0) == pytest.approx(18.0)

    def test_it_uses_the_typical_price_of_each_bar(self) -> None:
        # (12 + 6 + 9) / 3 = 9.
        highs, lows, closes = np.array([12.0]), np.array([6.0]), np.array([9.0])
        assert ind.anchored_vwap(highs, lows, closes, np.array([100.0]), 0) == pytest.approx(9.0)

    def test_the_anchor_excludes_earlier_bars(self) -> None:
        closes = np.array([1.0, 1.0, 20.0])
        volumes = np.full(3, 100.0)
        assert ind.anchored_vwap(closes, closes, closes, volumes, 2) == pytest.approx(20.0)

    def test_none_when_nothing_traded(self) -> None:
        # No volume means no average price *paid*; 0.0 would be a fabrication.
        flat = np.full(5, 50.0)
        assert ind.anchored_vwap(flat, flat, flat, np.zeros(5), 0) is None

    def test_none_when_the_anchor_is_out_of_range(self) -> None:
        flat = np.full(5, 50.0)
        assert ind.anchored_vwap(flat, flat, flat, np.full(5, 10.0), 9) is None


class TestLowestCloseIndex:
    def test_it_finds_the_trough(self) -> None:
        assert ind.lowest_close_index(np.array([10.0, 3.0, 8.0]), 3) == 1

    def test_it_indexes_the_whole_array_not_the_window(self) -> None:
        # The global low is at 0 but outside a 3-bar window; the windowed low is
        # the 2.0 at index 3, and the index returned must be usable directly.
        closes = np.array([1.0, 9.0, 8.0, 2.0, 7.0])
        assert ind.lowest_close_index(closes, 3) == 3

    def test_none_on_empty_input(self) -> None:
        assert ind.lowest_close_index(np.array([]), 10) is None


class TestBeta:
    def test_beta_against_itself_is_one(self) -> None:
        a = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        assert ind.beta(a, a, 5) == pytest.approx(1.0)

    def test_beta_of_a_doubled_series_is_two(self) -> None:
        # Same shape, twice the amplitude: correlation 1, twice the deviation.
        b = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        assert ind.beta(b * 2, b, 5) == pytest.approx(2.0)

    def test_beta_is_negative_when_the_series_move_against_each_other(self) -> None:
        b = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        assert ind.beta(-b, b, 5) == pytest.approx(-1.0)

    def test_beta_none_when_the_reference_is_flat(self) -> None:
        # No benchmark variance means the regression slope is undefined — the
        # answer is "unknown", never 0.0.
        a = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        assert ind.beta(a, np.zeros(5), 5) is None

    def test_beta_none_when_insufficient_history(self) -> None:
        a = np.array([0.01, -0.02, 0.03])
        assert ind.beta(a, a, 60) is None


class TestSlope:
    def test_rising_average_has_positive_slope(self) -> None:
        closes = np.arange(1.0, 60.0)  # steadily rising
        slope = ind.sma_slope(closes, period=10, slope_window=20)
        assert slope is not None and slope > 0

    def test_falling_average_has_negative_slope(self) -> None:
        closes = np.arange(60.0, 1.0, -1.0)
        slope = ind.sma_slope(closes, period=10, slope_window=20)
        assert slope is not None and slope < 0


class TestRSI:
    def test_rsi_of_only_gains_is_100(self) -> None:
        # Monotonic rise → no losses → RSI pinned at 100.
        closes = np.arange(1.0, 40.0)
        assert ind.relative_strength_index(closes, 14) == pytest.approx(100.0)

    def test_rsi_of_only_losses_is_near_zero(self) -> None:
        closes = np.arange(40.0, 1.0, -1.0)
        rsi = ind.relative_strength_index(closes, 14)
        assert rsi is not None and rsi < 1.0

    def test_rsi_is_bounded_0_to_100(self) -> None:
        rng = np.random.default_rng(3)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 100))
        rsi = ind.relative_strength_index(closes, 14)
        assert rsi is not None and 0.0 <= rsi <= 100.0

    def test_rsi_none_when_insufficient(self) -> None:
        assert ind.relative_strength_index(np.array([1.0, 2, 3]), 14) is None
