"""Logistic regression with a per-feature prior.

Three of these classes exist because the prior is the part of this design that
can fail silently rather than loudly.

`TestPriorWithoutData` pins the behaviour the whole index model rests on: with
no observations the fit must return the prior *exactly* and report a shrinkage
of zero. If it drifted instead, a coefficient nobody measured would look like
one somebody did.

`TestClamp` pins the bound on being wrong. Dealer gamma stays prior-driven for
years on non-overlapping windows, so the safeguard cannot be patience — it has
to be that a wrong prior can tilt a decision and never drive it.

`TestServingIsSafe` pins the two ways a served feature can be meaningless: a
value that is absent, and a value whose scale was never learned. Both must
contribute exactly zero rather than a plausible-looking number.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from app.models_ml.logistic import (
    FittedModel,
    Prior,
    auc,
    auc_gap,
    average_ranks,
    fit,
)

LABEL = "1 if the forward 20-day return is positive"


def _sample(
    n: int, coefficients: list[float], intercept: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Draw features and labels from a known logistic model."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, len(coefficients)))
    logit = intercept + x @ np.array(coefficients)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0.0, 1.0, n) < p).astype(np.float64)
    return x, y


def _prior_driven_model(clamp: float | None, shrinkage: float = 0.02) -> FittedModel:
    """A model whose single feature is carried by its prior."""
    return FittedModel(
        feature_names=("gex",),
        coefficients=(0.15,),
        intercept=0.0,
        means=(0.0,),
        sds=(1.0,),
        scale_known=(True,),
        prior_means=(0.15,),
        prior_taus=(0.35,),
        shrinkage=(shrinkage,),
        standard_errors=(0.35,),
        n_observations=10,
        positive_rate=0.5,
        auc=0.5,
        brier=0.25,
        log_loss=0.69,
        label_definition=LABEL,
        low_shrinkage_clamp=clamp,
    )


class TestRecoversKnownCoefficients:
    def test_recovers_the_generating_model(self) -> None:
        x, y = _sample(20_000, [1.5, -0.8, 0.4], intercept=0.3, seed=1)
        model = fit(x, y, ["a", "b", "c"], label_definition=LABEL)

        # Features are drawn standard normal, so the fitted (standardised)
        # coefficients are directly comparable with the generating ones.
        assert model.coefficients[0] == pytest.approx(1.5, abs=0.1)
        assert model.coefficients[1] == pytest.approx(-0.8, abs=0.1)
        assert model.coefficients[2] == pytest.approx(0.4, abs=0.1)
        assert model.intercept == pytest.approx(0.3, abs=0.1)

    def test_standard_errors_shrink_with_more_data(self) -> None:
        x_small, y_small = _sample(500, [1.0], intercept=0.0, seed=2)
        x_large, y_large = _sample(20_000, [1.0], intercept=0.0, seed=3)
        small = fit(x_small, y_small, ["a"], label_definition=LABEL)
        large = fit(x_large, y_large, ["a"], label_definition=LABEL)
        assert large.standard_errors[0] < small.standard_errors[0]

    def test_auc_beats_a_coin_flip_on_a_real_signal(self) -> None:
        x, y = _sample(5_000, [2.0], intercept=0.0, seed=4)
        model = fit(x, y, ["a"], label_definition=LABEL)
        assert model.auc > 0.75

    def test_raises_when_labels_are_not_binary(self) -> None:
        x, _ = _sample(50, [1.0], intercept=0.0, seed=5)
        with pytest.raises(ValueError, match="0 or 1"):
            fit(x, np.full(50, 0.5), ["a"], label_definition=LABEL)


class TestPriorWithoutData:
    def test_no_observations_returns_the_prior_exactly(self) -> None:
        model = fit(
            np.zeros((0, 2)),
            np.zeros(0),
            ["gex", "charm"],
            priors={"gex": Prior(0.15, 0.35), "charm": Prior(0.05, 0.35)},
            label_definition=LABEL,
        )
        assert model.coefficients[0] == pytest.approx(0.15)
        assert model.coefficients[1] == pytest.approx(0.05)
        assert model.n_observations == 0

    def test_no_observations_reports_zero_shrinkage(self) -> None:
        model = fit(
            np.zeros((0, 1)),
            np.zeros(0),
            ["gex"],
            priors={"gex": Prior(0.15, 0.35)},
            label_definition=LABEL,
        )
        assert model.shrinkage[0] == pytest.approx(0.0)

    def test_an_all_missing_feature_keeps_its_prior(self) -> None:
        """The realistic shape: other features have data, this one has none."""
        x, y = _sample(2_000, [1.0], intercept=0.0, seed=6)
        x = np.hstack([x, np.full((x.shape[0], 1), np.nan)])
        model = fit(
            x,
            y,
            ["real", "gex"],
            priors={"gex": Prior(0.15, 0.35)},
            label_definition=LABEL,
        )
        assert model.coefficients[1] == pytest.approx(0.15, abs=1e-6)
        assert model.shrinkage[1] == pytest.approx(0.0)
        assert model.scale_known[1] is False
        # The fitted feature is unaffected by its unfittable neighbour.
        assert model.coefficients[0] == pytest.approx(1.0, abs=0.2)
        assert model.shrinkage[0] > 0.9

    def test_data_overwhelms_the_prior_when_there_is_enough_of_it(self) -> None:
        """A prior pointing the wrong way must lose to a clear signal."""
        x, y = _sample(20_000, [1.2], intercept=0.0, seed=7)
        model = fit(x, y, ["a"], priors={"a": Prior(-0.5, 0.35)}, label_definition=LABEL)
        assert model.coefficients[0] == pytest.approx(1.2, abs=0.15)
        assert model.shrinkage[0] > 0.99


class TestPriorBehavesAsRidge:
    def test_zero_mean_prior_shrinks_toward_zero(self) -> None:
        x, y = _sample(300, [2.0], intercept=0.0, seed=8)
        weak = fit(x, y, ["a"], priors={"a": Prior(0.0, 10.0)}, label_definition=LABEL)
        strong = fit(x, y, ["a"], priors={"a": Prior(0.0, 0.05)}, label_definition=LABEL)
        assert abs(strong.coefficients[0]) < abs(weak.coefficients[0])

    def test_separation_stays_finite(self) -> None:
        """Perfectly separating data sends an unpenalised coefficient to infinity."""
        x = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = fit(x, y, ["a"], label_definition=LABEL)
        assert math.isfinite(model.coefficients[0])
        assert abs(model.coefficients[0]) < 50.0


class TestClamp:
    def test_an_extreme_reading_cannot_move_the_probability_far(self) -> None:
        model = _prior_driven_model(clamp=0.35)
        base = model.probability({})  # feature absent -> imputed -> 0.5
        extreme = model.probability({"gex": 40.0})  # 40 standard deviations
        assert base == pytest.approx(0.5)
        assert abs(extreme - base) <= 0.088

    def test_the_clamp_is_symmetric(self) -> None:
        model = _prior_driven_model(clamp=0.35)
        high = model.probability({"gex": 40.0})
        low = model.probability({"gex": -40.0})
        assert high - 0.5 == pytest.approx(0.5 - low, abs=1e-9)

    def test_without_a_clamp_the_same_reading_dominates(self) -> None:
        """Shows the clamp does the work, not the smallness of the coefficient."""
        model = _prior_driven_model(clamp=None)
        assert model.probability({"gex": 40.0}) > 0.99

    def test_a_feature_that_has_earned_its_shrinkage_is_not_clamped(self) -> None:
        model = _prior_driven_model(clamp=0.35, shrinkage=0.9)
        assert model.probability({"gex": 40.0}) > 0.99


class TestServingIsSafe:
    def test_a_missing_feature_contributes_nothing(self) -> None:
        model = _prior_driven_model(clamp=None)
        assert model.contributions({})["gex"] == 0.0
        assert model.probability({}) == pytest.approx(0.5)

    def test_a_non_finite_value_contributes_nothing(self) -> None:
        model = _prior_driven_model(clamp=None)
        assert model.contributions({"gex": float("nan")})["gex"] == 0.0

    def test_a_feature_whose_scale_is_unknown_is_never_served(self) -> None:
        """The dangerous case: a real value, divided by a fabricated scale."""
        model = dataclasses.replace(_prior_driven_model(clamp=None), scale_known=(False,))
        assert model.contributions({"gex": 2.5})["gex"] == 0.0
        assert model.probability({"gex": 2.5}) == pytest.approx(0.5)

    def test_contributions_sum_to_the_logit_when_unclamped(self) -> None:
        x, y = _sample(1_000, [1.0, -0.5], intercept=0.2, seed=9)
        model = fit(x, y, ["a", "b"], label_definition=LABEL)
        reading = {"a": 0.7, "b": -1.3}
        assert sum(model.contributions(reading).values()) + model.intercept == pytest.approx(
            model.logit(reading)
        )


class TestRankStatistics:
    def test_ties_share_their_average_rank(self) -> None:
        ranks = average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
        assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]

    def test_auc_is_one_for_a_perfect_ordering(self) -> None:
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        assert auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    def test_auc_is_half_when_every_prediction_ties(self) -> None:
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        assert auc(labels, np.full(4, 0.5)) == pytest.approx(0.5)

    def test_auc_is_undefined_with_one_class(self) -> None:
        assert math.isnan(auc(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4])))


class TestPairedComparison:
    """`auc_gap` exists because the unpaired comparison is underpowered.

    The point is not that it gives a different number — the point estimate is
    the same subtraction either way. It is that resampling the two models
    *together* removes the noise they share, which on a short window with few
    events is most of the noise there is.
    """

    def test_the_difference_is_the_difference(self) -> None:
        rng = np.random.default_rng(3)
        labels = (rng.random(500) < 0.2).astype(float)
        left = labels + rng.normal(0, 0.8, 500)
        right = rng.normal(0, 1.0, 500)
        gap = auc_gap(labels, left, right, resamples=400)
        assert gap.difference == pytest.approx(auc(labels, left) - auc(labels, right))

    def test_a_model_against_itself_is_exactly_zero(self) -> None:
        rng = np.random.default_rng(4)
        labels = (rng.random(300) < 0.1).astype(float)
        scores = rng.normal(size=300)
        gap = auc_gap(labels, scores, scores, resamples=200)
        assert gap.difference == pytest.approx(0.0)
        assert gap.low == pytest.approx(0.0)
        assert gap.high == pytest.approx(0.0)
        assert not gap.real

    def test_it_is_antisymmetric(self) -> None:
        rng = np.random.default_rng(5)
        labels = (rng.random(400) < 0.15).astype(float)
        left = labels + rng.normal(0, 0.9, 400)
        right = labels + rng.normal(0, 1.4, 400)
        forward = auc_gap(labels, left, right, resamples=400)
        backward = auc_gap(labels, right, left, resamples=400)
        assert forward.difference == pytest.approx(-backward.difference)
        assert forward.share == pytest.approx(1.0 - backward.share, abs=0.05)

    def test_pairing_is_sharper_than_two_separate_intervals(self) -> None:
        """The reason this function exists, pinned as a property.

        Two models built from the same signal plus different noise are ranked
        the same way on almost every resample, so the paired interval is far
        narrower than either model's own interval. If a refactor lost the
        pairing — by resampling the two independently — this is what would
        catch it.
        """
        rng = np.random.default_rng(6)
        labels = (rng.random(600) < 0.1).astype(float)
        shared = labels + rng.normal(0, 1.0, 600)
        left = shared + rng.normal(0, 0.05, 600)
        right = shared

        paired = auc_gap(labels, left, right, resamples=600)
        spread = paired.high - paired.low

        # The same comparison done the naive way: each model resampled on its
        # own days, so the shared noise never cancels.
        rng = np.random.default_rng(7)
        draws = np.empty(600)
        for i in range(600):
            a = rng.integers(0, labels.size, labels.size)
            b = rng.integers(0, labels.size, labels.size)
            draws[i] = auc(labels[a], left[a]) - auc(labels[b], right[b])
        unpaired = float(np.quantile(draws, 0.95) - np.quantile(draws, 0.05))

        assert spread < unpaired / 3.0

    def test_it_survives_a_window_with_no_events(self) -> None:
        gap = auc_gap(np.zeros(50), np.random.default_rng(8).normal(size=50), np.zeros(50))
        assert math.isnan(gap.low) or math.isnan(gap.difference)
