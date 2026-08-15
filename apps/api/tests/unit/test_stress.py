"""Whole-book stress testing.

The property that matters, and the reason this is a bootstrap rather than a
covariance model, is that resampling *dates* preserves the correlation between
holdings. A stress test that quietly treated positions as independent would
report a comfortable number for exactly the book that is most dangerous — six
positions that are really one bet — so that is pinned from both directions:
perfectly correlated holdings must compound, perfectly opposed ones must cancel.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from app.risk import stress


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _alternating(magnitude: float, n: int = 120) -> np.ndarray:
    """A return series of fixed magnitude that alternates sign."""
    return np.array([magnitude if i % 2 == 0 else -magnitude for i in range(n)])


class TestBootstrapStress:
    def test_a_riskless_book_has_no_stress_loss(self) -> None:
        result = stress.bootstrap_stress({"a": np.zeros(120)}, {"a": 1.0}, rng=_rng())
        assert result is not None
        assert result.portfolio_loss_pct == 0.0

    def test_a_volatile_book_has_a_positive_stress_loss(self) -> None:
        result = stress.bootstrap_stress({"a": _alternating(0.03)}, {"a": 1.0}, rng=_rng())
        assert result is not None
        assert result.portfolio_loss_pct > 0.0

    def test_identical_holdings_stress_the_same_as_one_of_their_total_weight(self) -> None:
        """The correlation test, from the dangerous direction.

        Four holdings at 10% each that move identically are a 40% position in
        one thing. If the simulation resampled each holding independently their
        losses would partly cancel and the book would look safer than it is.
        """
        series = _alternating(0.02)
        split = stress.bootstrap_stress(
            {k: series.copy() for k in "abcd"},
            dict.fromkeys("abcd", 0.1),
            rng=_rng(),
        )
        single = stress.bootstrap_stress({"a": series.copy()}, {"a": 0.40}, rng=_rng())
        assert split is not None and single is not None
        assert split.portfolio_loss_pct == pytest.approx(single.portfolio_loss_pct)

    def test_opposed_holdings_cancel(self) -> None:
        """And from the safe direction: a genuine hedge must show as one."""
        series = _alternating(0.02)
        result = stress.bootstrap_stress(
            {"long": series, "short": -series},
            {"long": 0.5, "short": 0.5},
            rng=_rng(),
        )
        assert result is not None
        assert result.portfolio_loss_pct == pytest.approx(0.0)

    def test_doubling_weight_roughly_doubles_the_loss(self) -> None:
        series = _alternating(0.02)
        small = stress.bootstrap_stress({"a": series}, {"a": 0.25}, rng=_rng())
        large = stress.bootstrap_stress({"a": series}, {"a": 0.50}, rng=_rng())
        assert small is not None and large is not None
        # Compounding makes this not exactly 2x, but it must be close.
        assert large.portfolio_loss_pct == pytest.approx(2 * small.portfolio_loss_pct, rel=0.05)

    def test_the_same_book_always_produces_the_same_verdict(self) -> None:
        """A trade must not be approved or refused by luck of the draw."""
        series = _alternating(0.02)
        first = stress.bootstrap_stress({"a": series}, {"a": 0.5}, rng=_rng())
        second = stress.bootstrap_stress({"a": series}, {"a": 0.5}, rng=_rng())
        assert first is not None and second is not None
        assert first.portfolio_loss_pct == second.portfolio_loss_pct

    def test_none_when_there_is_nothing_to_simulate(self) -> None:
        assert stress.bootstrap_stress({}, {}, rng=_rng()) is None

    def test_none_when_history_is_too_short(self) -> None:
        """Unknown is not the same as zero, and must not be reported as zero."""
        short = np.full(stress.MIN_RETURNS - 1, 0.01)
        assert stress.bootstrap_stress({"a": short}, {"a": 1.0}, rng=_rng()) is None

    def test_zero_weight_holdings_are_ignored(self) -> None:
        series = _alternating(0.02)
        with_zero = stress.bootstrap_stress(
            {"a": series, "b": series.copy()}, {"a": 0.5, "b": 0.0}, rng=_rng()
        )
        without = stress.bootstrap_stress({"a": series}, {"a": 0.5}, rng=_rng())
        assert with_zero is not None and without is not None
        assert with_zero.portfolio_loss_pct == pytest.approx(without.portfolio_loss_pct)

    def test_none_when_a_series_carries_nan(self) -> None:
        series = _alternating(0.02)
        series[5] = np.nan
        assert stress.bootstrap_stress({"a": series}, {"a": 1.0}, rng=_rng()) is None


class TestDrawdownScaling:
    def test_a_loss_within_the_limit_leaves_the_size_alone(self) -> None:
        assert stress.drawdown_scaled_quantity(Decimal("100"), 0.10, 0.15) == Decimal("100")

    def test_a_breach_scales_the_size_down_proportionally(self) -> None:
        # 20% stressed loss against a 15% limit → three quarters of the size.
        assert stress.drawdown_scaled_quantity(Decimal("100"), 0.20, 0.15) == Decimal("75")

    def test_an_unmeasurable_loss_leaves_the_size_alone(self) -> None:
        # Never tighten on a number we do not have.
        assert stress.drawdown_scaled_quantity(Decimal("100"), 0.0, 0.15) == Decimal("100")

    def test_a_disabled_limit_leaves_the_size_alone(self) -> None:
        assert stress.drawdown_scaled_quantity(Decimal("100"), 0.20, 0.0) == Decimal("100")
