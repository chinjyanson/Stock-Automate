"""The two fitted strategies, and the guarantees that keep them honest.

`TestTrainServeIdentity` is the most valuable test here. A model fitted on one
definition of a feature and served on another produces plausible probabilities
that are simply wrong, and nothing anywhere errors — it is the failure mode that
backtests well and trades badly. The guard is that the fit, the replay and the
live strategy all read features through one function; these tests pin that they
actually agree on a number rather than merely sharing an import.

`TestPriorFeaturesAreBounded` pins the safeguard the index model rests on.
Dealer gamma cannot be backfilled, so its coefficient is theory rather than
measurement, and on 20-day non-overlapping windows it stays that way for years.
The bound therefore cannot be patience — it has to be magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.backtest.entries import EveryBarReader, ModelReader
from app.backtest.features import MIN_BARS
from app.backtest.features import compute as compute_features
from app.indicators.series import PriceSeries
from app.models_ml.logistic import FittedModel, Prior, fit
from app.strategies.base import IndexConditions
from app.strategies.logistic_index import read_index, read_index_features
from app.strategies.logistic_stock import PRICE_FEATURES, read_features, read_stock


def _series(n: int = 400, seed: int = 0) -> PriceSeries:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, n))
    return PriceSeries(
        open=close.copy(),
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        adjusted_close=np.full(n, np.nan),
        volume=np.full(n, 1_000_000.0),
    )


def _model(
    names: tuple[str, ...] = PRICE_FEATURES,
    *,
    intercept: float = 0.0,
    clamp: float | None = None,
    shrinkage: tuple[float, ...] | None = None,
    coefficients: tuple[float, ...] | None = None,
) -> FittedModel:
    n = len(names)
    return FittedModel(
        feature_names=names,
        coefficients=coefficients or tuple(0.3 for _ in range(n)),
        intercept=intercept,
        means=(0.0,) * n,
        sds=(1.0,) * n,
        scale_known=(True,) * n,
        prior_means=(0.0,) * n,
        prior_taus=(1.0,) * n,
        shrinkage=shrinkage or (1.0,) * n,
        standard_errors=(0.1,) * n,
        n_observations=1_000,
        positive_rate=0.44,
        auc=0.55,
        brier=0.24,
        log_loss=0.68,
        label_definition="test",
        low_shrinkage_clamp=clamp,
    )


class TestTrainServeIdentity:
    def test_the_strategy_reads_the_same_numbers_the_fit_did(self) -> None:
        """The live path and the training path must agree bar for bar.

        The fit reads `compute(...)[name][i]` over a whole series; the strategy
        reads `compute(...)[name][-1]` over the series truncated at `i`. Those
        are the same quantity only if every feature is genuinely point-in-time,
        which is exactly what this asserts.
        """
        series = _series(400)
        columns = compute_features(
            series.open, series.high, series.low, series.close, series.volume
        )
        for i in (300, 350, 399):
            served = read_features(series.head(i + 1))
            for name in PRICE_FEATURES:
                assert served[name] == pytest.approx(float(columns[name][i]), rel=1e-12)

    def test_the_replay_reader_and_the_strategy_agree_on_a_probability(self) -> None:
        """The seam the sweep depends on: a threshold means one thing in both."""
        series = _series(400)
        model = _model()
        reader = ModelReader(model=model, threshold=0.5, min_atr_pct=0.0)

        reading = reader(series)
        live = read_stock(series, model, bb_period=20, bb_std=2.0, atr_period=14)
        assert reading is not None and live is not None
        assert reading.score == pytest.approx(live.probability, rel=1e-12)

    def test_rewriting_the_future_cannot_change_the_present(self) -> None:
        """Point-in-time, asserted rather than assumed."""
        series = _series(400)
        before = read_features(series.head(300))

        mutated = series.close.copy()
        mutated[300:] = 1_000.0
        rewritten = PriceSeries(
            open=mutated.copy(),
            high=mutated * 1.01,
            low=mutated * 0.99,
            close=mutated,
            adjusted_close=np.full(mutated.size, np.nan),
            volume=series.volume,
        )
        after = read_features(rewritten.head(300))
        assert before == pytest.approx(after)


class TestNoOpinionIsNotABaseRate:
    def test_a_series_too_short_for_any_feature_declines(self) -> None:
        """Below MIN_BARS the feature module returns nothing at all.

        Serving the base rate there would answer `sigmoid(intercept)` for every
        such instrument, identically, having seen nothing about any of them — so
        a confident intercept would buy the entire short-history universe on no
        information.
        """
        short = _series(MIN_BARS - 20)
        model = _model(intercept=5.0)  # would say 0.99 to anything
        assert read_stock(short, model, bb_period=20, bb_std=2.0, atr_period=14) is None

    def test_a_long_enough_series_does_have_an_opinion(self) -> None:
        model = _model(intercept=5.0)
        reading = read_stock(_series(400), model, bb_period=20, bb_std=2.0, atr_period=14)
        assert reading is not None
        assert reading.probability > 0.9


class TestPriorFeaturesAreBounded:
    """A wrong prior may tilt the index decision and must never drive it."""

    _NAMES = ("sma200_slope", "gamma_tilt")

    def _index_model(self, clamp: float | None) -> FittedModel:
        return _model(
            self._NAMES,
            clamp=clamp,
            # gamma_tilt is almost entirely prior; the price feature is fitted.
            shrinkage=(0.95, 0.02),
            coefficients=(0.3, 0.15),
        )

    def _conditions(self, tilt: float) -> IndexConditions:
        return IndexConditions(regime_factor=1.0, gamma_tilt=tilt, options_available=True)

    def test_an_extreme_reading_moves_the_probability_by_under_nine_points(self) -> None:
        series = _series(400)
        model = self._index_model(clamp=0.35)
        neutral = read_index(series, model, IndexConditions())
        extreme = read_index(series, model, self._conditions(50.0))
        assert neutral is not None and extreme is not None
        assert abs(extreme.probability - neutral.probability) <= 0.088

    def test_without_the_clamp_the_same_reading_dominates(self) -> None:
        """Proves the bound comes from the clamp, not from a small coefficient."""
        series = _series(400)
        model = self._index_model(clamp=None)
        neutral = read_index(series, model, IndexConditions())
        extreme = read_index(series, model, self._conditions(50.0))
        assert neutral is not None and extreme is not None
        assert abs(extreme.probability - neutral.probability) > 0.3

    def test_a_feature_whose_scale_is_unknown_is_not_served(self) -> None:
        """The prior supplies a coefficient; it does not supply a scale.

        Standardising a real reading against a fabricated mean and standard
        deviation gives a meaningless number that still looks like one.
        """
        import dataclasses

        series = _series(400)
        model = dataclasses.replace(self._index_model(clamp=None), scale_known=(True, False))
        neutral = read_index(series, model, IndexConditions())
        extreme = read_index(series, model, self._conditions(50.0))
        assert neutral is not None and extreme is not None
        assert neutral.probability == pytest.approx(extreme.probability)

    def test_option_features_are_offered_only_when_a_chain_was_read(self) -> None:
        series = _series(400)
        absent = read_index_features(series, IndexConditions(gamma_tilt=0.4))
        present = read_index_features(
            series, IndexConditions(gamma_tilt=0.4, options_available=True)
        )
        assert "gamma_tilt" not in absent
        assert present["gamma_tilt"] == pytest.approx(0.4)


class TestFittedPriorsSurviveARoundTrip:
    def test_a_prior_only_feature_keeps_its_prior_through_a_fit(self) -> None:
        """The whole index design in one assertion: a column with no history at
        all comes back with its prior intact, marked unfitted, while its
        neighbour is fitted normally."""
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, (500, 2))
        y = (x[:, 0] + rng.normal(0.0, 0.5, 500) > 0).astype(float)
        x[:, 1] = np.nan  # the un-backfillable column

        model = fit(
            x,
            y,
            ["price", "gamma_tilt"],
            priors={"gamma_tilt": Prior(0.15, 0.35)},
            label_definition="test",
            low_shrinkage_clamp=0.35,
        )
        assert model.coefficients[1] == pytest.approx(0.15, abs=1e-6)
        assert model.shrinkage[1] == pytest.approx(0.0)
        assert model.scale_known[1] is False
        assert model.shrinkage[0] > 0.9


class TestPrecomputingChangesNothing:
    """`ModelReader.prepare` is an optimisation, so it must be invisible.

    It computes the feature matrix once instead of once per bar, which turned a
    twenty-minute sweep into seconds. That is only sound because every feature
    is point-in-time; if it were not, this would quietly change the answers
    rather than fail, so the equality is asserted directly.
    """

    def test_a_prepared_reader_gives_the_same_reading(self) -> None:
        series = _series(400)
        model = _model()

        naive = ModelReader(model=model, threshold=0.5, min_atr_pct=0.0)
        prepared = ModelReader(model=model, threshold=0.5, min_atr_pct=0.0)
        prepared.prepare(series)

        for i in (300, 350, 399):
            window = series.head(i + 1)
            a = naive(window)
            b = prepared(window)
            assert a is not None and b is not None
            assert a.score == pytest.approx(b.score, rel=1e-12)
            assert a.admits == b.admits

    def test_a_prepared_replay_produces_the_same_trades(self) -> None:
        from app.backtest.engine import ReplayConfig, replay

        series = _series(400)
        model = _model()
        config = ReplayConfig(warmup_bars=260)

        # `replay` calls `prepare` itself, so this reader is the prepared one.
        prepared = replay(series, ModelReader(model=model, threshold=0.5, min_atr_pct=0.0), config)

        # A reader whose prepare is a no-op takes the per-bar path instead.
        naive = ModelReader(model=model, threshold=0.5, min_atr_pct=0.0)
        naive.prepare = lambda _series: None  # type: ignore[method-assign]
        per_bar = replay(series, naive, config)

        assert prepared.trade_count == per_bar.trade_count
        for a, b in zip(prepared.trades, per_bar.trades, strict=True):
            assert a.entry_index == b.entry_index
            assert a.exit_index == b.exit_index
            assert a.entry_price == pytest.approx(b.entry_price)
            assert a.exit_price == pytest.approx(b.exit_price)
            assert a.entry_score == pytest.approx(b.entry_score)
            assert a.exit_reason is b.exit_reason


class TestReadersDiffer:
    def test_the_labelling_reader_admits_what_a_fitted_one_refuses(self) -> None:
        """The training set must not be some model's opinion.

        The doubtful model here is doubtful by its intercept rather than by a
        high threshold: a threshold alone would depend on whether this
        particular fixture happened to look attractive, which is the fixture
        talking rather than the readers.
        """
        series = _series(400)
        every = EveryBarReader()(series)
        picky = ModelReader(model=_model(intercept=-20.0), threshold=0.55)(series)
        assert every is not None and picky is not None
        assert every.admits
        assert not picky.admits
