"""The scoring, which decides every conclusion in this directory.

If AUC is subtly wrong, every table here is wrong in the same direction and
nothing else would reveal it. The cases that matter are the ones with ties: a
detector that emits the same number on many days — a saturating one, or a
sampled probability quantised to multiples of 1/512, which is exactly what
`zeroshot.py` produces — must be neither rewarded nor punished for that.
"""

from __future__ import annotations

import numpy as np
import pytest

import metrics


class TestAucOnCasesWithKnownAnswers:
    def test_perfect_separation(self) -> None:
        score = np.array([0.1, 0.2, 0.8, 0.9])
        label = np.array([0.0, 0.0, 1.0, 1.0])
        assert metrics.auc(score, label) == pytest.approx(1.0)

    def test_exactly_backwards(self) -> None:
        score = np.array([0.9, 0.8, 0.2, 0.1])
        label = np.array([0.0, 0.0, 1.0, 1.0])
        assert metrics.auc(score, label) == pytest.approx(0.0)

    def test_all_tied_is_a_coin_toss(self) -> None:
        score = np.full(10, 0.3)
        label = np.array([1.0] * 3 + [0.0] * 7)
        assert metrics.auc(score, label) == pytest.approx(0.5)

    def test_one_tie_counts_as_half_a_win(self) -> None:
        # One positive and one negative share a score; the other pair separates
        # perfectly. So: one clean win, one half.
        score = np.array([0.0, 0.5, 0.5, 1.0])
        label = np.array([0.0, 0.0, 1.0, 1.0])
        assert metrics.auc(score, label) == pytest.approx(0.875)

    def test_undefined_without_both_classes(self) -> None:
        score = np.array([0.1, 0.2, 0.3])
        assert np.isnan(metrics.auc(score, np.zeros(3)))
        assert np.isnan(metrics.auc(score, np.ones(3)))

    def test_invariant_to_any_rising_rescaling(self) -> None:
        rng = np.random.default_rng(1)
        score = rng.normal(size=400)
        label = (rng.random(400) < 0.1).astype(float)
        plain = metrics.auc(score, label)
        assert metrics.auc(score * 7.0 + 3.0, label) == pytest.approx(plain)
        assert metrics.auc(1.0 / (1.0 + np.exp(-score)), label) == pytest.approx(plain)

    def test_matches_the_brute_force_definition(self) -> None:
        """Every positive against every negative, counted by hand."""
        rng = np.random.default_rng(11)
        score = np.round(rng.normal(size=120), 1)  # rounding manufactures ties
        label = (rng.random(120) < 0.2).astype(float)

        wins = 0.0
        positives = score[label > 0.5]
        negatives = score[label < 0.5]
        for high in positives:
            wins += float((high > negatives).sum()) + 0.5 * float((high == negatives).sum())
        assert metrics.auc(score, label) == pytest.approx(wins / (positives.size * negatives.size))


class TestLift:
    def test_a_perfect_detector_concentrates_every_event(self) -> None:
        label = np.zeros(100)
        label[:10] = 1.0
        score = -np.arange(100.0)
        ratio, rate = metrics.lift(score, label, fraction=0.10)
        assert rate == pytest.approx(1.0)
        assert ratio == pytest.approx(10.0)

    def test_a_useless_detector_finds_the_background_rate(self) -> None:
        rng = np.random.default_rng(3)
        label = (rng.random(4000) < 0.05).astype(float)
        ratio, _ = metrics.lift(rng.normal(size=4000), label, fraction=0.10)
        assert 0.6 < ratio < 1.6


class TestTheIntervalIsHonestAboutSmallSamples:
    def test_noise_covers_a_coin_toss(self) -> None:
        rng = np.random.default_rng(6)
        label = (rng.random(600) < 0.04).astype(float)
        score = metrics.evaluate("noise", rng.normal(size=600), label)
        assert score.auc_low < 0.5 < score.auc_high
        assert not score.informative

    def test_real_signal_clears_it(self) -> None:
        rng = np.random.default_rng(7)
        label = (rng.random(3000) < 0.1).astype(float)
        score = metrics.evaluate("signal", label + rng.normal(0, 0.6, 3000), label)
        assert score.informative
        assert score.auc_low > 0.5

    def test_fewer_events_gives_a_wider_interval(self) -> None:
        rng = np.random.default_rng(8)
        wide = metrics.evaluate("few", rng.normal(size=200), (rng.random(200) < 0.03).astype(float))
        narrow = metrics.evaluate(
            "many", rng.normal(size=4000), (rng.random(4000) < 0.3).astype(float)
        )
        assert (wide.auc_high - wide.auc_low) > (narrow.auc_high - narrow.auc_low)
