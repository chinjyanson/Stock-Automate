"""The overlay's decision rule, which is the part that moves money."""

from __future__ import annotations

import numpy as np
import pytest

from app.signals.crash_features import (
    CALIBRATION_MIN,
    FEATURES,
    OverlayState,
    label_fall,
    momentum,
    rows,
    step,
    trigger_from,
)

DEFENSIVE = 0.3
TIMEOUT = 20


def _step(
    state: OverlayState,
    probability: float | None,
    trigger: float | None = 0.5,
    close: float = 100.0,
):
    return step(
        state,
        probability=probability,
        trigger=trigger,
        close=close,
        defensive=DEFENSIVE,
        timeout=TIMEOUT,
    )


class TestFailsInvested:
    """A missing feed must never be able to sell the portfolio."""

    def test_no_probability_holds_fully_invested(self) -> None:
        state, reason = _step(OverlayState(), None)
        assert state.exposure == 1.0
        assert not state.is_defensive
        assert "no reading" in reason

    def test_no_trigger_holds_fully_invested(self) -> None:
        # Before two years of model output the trigger is undefined, and an
        # undefined trigger means "do not warn" — never "warn".
        state, _ = _step(OverlayState(), 0.99, trigger=None)
        assert state.exposure == 1.0

    def test_losing_the_feed_while_defensive_returns_the_position(self) -> None:
        aside = OverlayState(exposure=DEFENSIVE, exit_price=100.0, days_out=3)
        state, reason = _step(aside, None)
        assert state.exposure == 1.0
        assert not state.is_defensive
        assert "returned" in reason


class TestSteppingAside:
    def test_warning_cuts_exposure_and_records_the_price(self) -> None:
        state, reason = _step(OverlayState(), 0.6, trigger=0.5, close=4200.0)
        assert state.exposure == DEFENSIVE
        assert state.exit_price == 4200.0
        assert state.days_out == 0
        assert "warning fired" in reason

    def test_probability_exactly_at_the_trigger_warns(self) -> None:
        # The trigger is a percentile the day is ranked against; a day sitting
        # exactly on it is inside the alarming fraction, not outside.
        state, _ = _step(OverlayState(), 0.5, trigger=0.5)
        assert state.exposure == DEFENSIVE

    def test_below_the_trigger_stays_fully_invested(self) -> None:
        state, reason = _step(OverlayState(), 0.49, trigger=0.5)
        assert state.exposure == 1.0
        assert "no warning" in reason


class TestComingBack:
    def test_all_clear_reinvests_in_full_the_very_next_day(self) -> None:
        """The whole point of the all-clear rule: no waiting on price."""
        aside = OverlayState(exposure=DEFENSIVE, exit_price=100.0, days_out=0)
        state, reason = _step(aside, 0.1, trigger=0.5, close=80.0)
        assert state.exposure == 1.0
        assert not state.is_defensive
        assert "all clear" in reason

    def test_reinvests_even_when_the_price_has_risen_past_the_exit(self) -> None:
        # The rule this replaced could only re-enter on *further falls*, so a
        # market that rallied off the alarm left the position pinned at the
        # defensive weight for a month. That is the regression this guards.
        aside = OverlayState(exposure=DEFENSIVE, exit_price=100.0, days_out=1)
        state, _ = _step(aside, 0.1, trigger=0.5, close=115.0)
        assert state.exposure == 1.0

    def test_standing_warning_keeps_the_position_aside_and_counts_days(self) -> None:
        state = OverlayState(exposure=DEFENSIVE, exit_price=100.0, days_out=0)
        for expected_day in (1, 2, 3):
            state, reason = _step(state, 0.9, trigger=0.5)
            assert state.exposure == DEFENSIVE
            assert state.days_out == expected_day
            assert "still standing" in reason

    def test_timeout_returns_the_position_even_while_warning(self) -> None:
        state = OverlayState(exposure=DEFENSIVE, exit_price=100.0, days_out=TIMEOUT - 1)
        state, reason = _step(state, 0.99, trigger=0.5)
        assert state.exposure == 1.0
        assert not state.is_defensive
        assert "timed out" in reason

    def test_exit_price_is_carried_while_aside_and_dropped_on_return(self) -> None:
        state, _ = _step(OverlayState(), 0.9, trigger=0.5, close=4200.0)
        state, _ = _step(state, 0.9, trigger=0.5, close=4100.0)
        assert state.exit_price == 4200.0
        state, _ = _step(state, 0.1, trigger=0.5, close=4150.0)
        assert state.exit_price is None


class TestTrigger:
    def test_refuses_to_calibrate_on_too_little_history(self) -> None:
        assert trigger_from(np.linspace(0, 1, CALIBRATION_MIN - 1), 0.10) is None

    def test_is_the_requested_upper_percentile(self) -> None:
        past = np.linspace(0.0, 1.0, CALIBRATION_MIN)
        assert trigger_from(past, 0.10) == pytest.approx(0.9, abs=0.01)

    def test_ignores_missing_values_rather_than_counting_them(self) -> None:
        past = np.concatenate([np.full(10, np.nan), np.linspace(0.0, 1.0, CALIBRATION_MIN)])
        assert trigger_from(past, 0.10) == pytest.approx(0.9, abs=0.01)

    def test_too_few_finite_values_is_still_refused(self) -> None:
        past = np.concatenate([np.linspace(0, 1, 100), np.full(CALIBRATION_MIN, np.nan)])
        assert trigger_from(past, 0.10) is None


class TestLabel:
    def test_marks_the_day_before_a_fall(self) -> None:
        close = np.array([100.0, 97.0, 97.0])
        assert label_fall(close, 0, fall=0.02, horizon=1) == 1.0

    def test_does_not_mark_a_shallower_fall(self) -> None:
        close = np.array([100.0, 99.0, 99.0])
        assert label_fall(close, 0, fall=0.02, horizon=1) == 0.0

    def test_looks_no_further_than_the_horizon(self) -> None:
        # The crash is two days out; a one-day horizon must not see it.
        close = np.array([100.0, 100.0, 90.0])
        assert label_fall(close, 0, fall=0.02, horizon=1) == 0.0
        assert label_fall(close, 0, fall=0.02, horizon=2) == 1.0


class TestFeatureMatrix:
    def test_rows_are_ordered_by_the_feature_list_not_the_dict(self) -> None:
        signals = {name: np.array([float(i), 0.0]) for i, name in enumerate(FEATURES)}
        matrix = rows(signals, np.array([0]), FEATURES)
        assert matrix.shape == (1, len(FEATURES))
        assert list(matrix[0]) == [float(i) for i in range(len(FEATURES))]

    def test_momentum_is_undefined_before_its_window(self) -> None:
        out = momentum(np.arange(1.0, 11.0), window=3)
        assert np.all(np.isnan(out[:3]))
        assert np.isfinite(out[3:]).all()
